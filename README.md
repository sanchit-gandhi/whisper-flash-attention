# Whisper Flash Attention

This is a repository for benchmarking the [Whisper Model](https://arxiv.org/abs/2212.04356) with memory efficient multi-head
attention (MHA) from the [xFormers repository](https://github.com/facebookresearch/xformers) by Facebook research.

The modelling code is split into two parts:
* [flash_attention.py](whisper_flash_attention/flash_attention.py): implements memory efficient attention using the xFormers back-end
* [modeling_whisper_flash_attention.py](whisper_flash_attention/modeling_whisper_flash_attention.py): augments the Hugging Face Transformers Whisper model with memory efficient attention

## Installation
Installation requires [PyTorch 1.13.1](https://pytorch.org/get-started/locally/) or higher, the xFormers package and 
Hugging Face Transformers:
```
pip install -U xformers transformers
```

## Usage
The memory efficient attention model can be loaded in much the same way as the original Hugging Face Transformers model:
```python
from transformers import WhisperConfig
from whisper_flash_attention import WhisperFlashAttentionForConditionalGeneration

config = WhisperConfig()
model = WhisperFlashAttentionForConditionalGeneration(config)
```
Currently, this repository only supports randomly initialised weights. Adding support to convert pre-trained weights to 
the flash attention format is a TODO:
```python
from whisper_flash_attention import WhisperFlashAttentionForConditionalGeneration

model = WhisperFlashAttentionForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
```

## Results
The benchmark can be run from the Python script in [benchmark_flash_attention.py](benchmark_flash_attention.py). 
By default, it uses:
* Batch size = 1: single-batch inference with greedy decoding
* Num batches = 100: benchmark over multiple batches for a good estimate of runtime
* Generated tokens = 25: this is the typical output sequence length for speech

