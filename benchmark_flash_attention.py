from datasets import load_dataset
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
from whisper_flash_attention import WhisperFlashAttentionForConditionalGeneration

import torch
from torch.utils.data import DataLoader
import numpy as np

import time
from tqdm import tqdm
import subprocess as sp

import csv

BATCH_SIZE = 1
NUM_BATCHES = 100
GENERATED_TOKENS = 25  # this is the typical output seq len for speech
USE_FLASH_ATTENTION = True

# MQA model from whisper-mqa or MHA model from transformers
whisper_cls = WhisperFlashAttentionForConditionalGeneration if USE_FLASH_ATTENTION else WhisperForConditionalGeneration

# benchmark on 100 samples from the LS dataset
librispeech = load_dataset("sanchit-gandhi/librispeech_asr_clean", split="train.100")
librispeech = librispeech.select(range(BATCH_SIZE * NUM_BATCHES))

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

def preprocess(batch):
    batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features[0]
    return batch

dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

dataloader = DataLoader(dataset_processed.with_format("torch"), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)


def get_gpu_memory():
    """Python equivalent of nvidia-smi, modified from https://stackoverflow.com/a/67722676"""
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    command = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(sp.check_output(command.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    memory_use_values = [int(x.split()[0]) for x in memory_use_info]
    return memory_use_values


# dicts to store our results
whisper_checkpoints = ["tiny.en", "base.en", "small.en", "medium.en"]
decoder_layer_results = {checkpoint: [] for checkpoint in whisper_checkpoints}
runtime_results = {checkpoint: [] for checkpoint in whisper_checkpoints}
param_results = {checkpoint: [] for checkpoint in whisper_checkpoints}
vram_results = {checkpoint: [] for checkpoint in whisper_checkpoints}


for checkpoint in whisper_checkpoints:
    print(10 * "=", checkpoint, 10 * "=")
    checkpoint_id = f"openai/whisper-{checkpoint}"
    config = WhisperConfig.from_pretrained(checkpoint_id)

    if checkpoint == "large-v2":
        layer_increments = [2, 4, 6, 8, 16, 32]
    elif checkpoint == "medium.en":
        layer_increments = [2, 4, 6, 8, 16, 24]
    else:
        total_decoder_layers = config.decoder_layers
        layer_increments = np.arange(2, total_decoder_layers + 2, 2)

    layer_increments = layer_increments[::-1]

    for decoder_layers in layer_increments:
        print("Layers: ", decoder_layers)
        config.decoder_layers = int(decoder_layers)
        model = whisper_cls(config)
        model.to("cuda")
        model.half()

        start = time.time()
        for batch in tqdm(dataloader):
            predicted_ids = model.generate(batch["input_features"].to("cuda").half(), max_new_tokens=GENERATED_TOKENS, min_new_tokens=GENERATED_TOKENS)
        runtime = time.time() - start

        decoder_layer_results[checkpoint].append(int(decoder_layers))
        runtime_results[checkpoint].append(runtime)
        param_results[checkpoint].append(model.num_parameters() / 10 ** 6)
        vram_results[checkpoint].append(get_gpu_memory()[0])

        del model
        torch.cuda.empty_cache()

# Save the results
compression_results = {}
for checkpoint in param_results:
    original_params = param_results[checkpoint][0]
    compression_results[checkpoint] = [100 * (original_params - compressed_params) / original_params for compressed_params in param_results[checkpoint]]

# Save the results
headers = ["Checkpoint", "Dec layers", "Params / M", "Compression / %", "VRAM / GB", "Runtime / s"]
with open("results.csv", "w", encoding="UTF8") as f:
    writer = csv.writer(f)
    # write the headers
    writer.writerow(headers)
    # write the data
    for key in decoder_layer_results:
        for i in range(len(decoder_layer_results[key])):
            prefix = key.replace(".en", "").replace("-v2", "") if i == 0 else ""
            data = [prefix, decoder_layer_results[key][i], round(param_results[key][i], 1), round(compression_results[key][i], 1), round(vram_results[key][i] / 1000, 2), round(runtime_results[key][i], 1)]
            writer.writerow(data)
        writer.writerow([])
