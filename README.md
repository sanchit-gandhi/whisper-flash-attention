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

### Default Setting
* Batch size = 1: single-batch inference with greedy decoding
* Generated tokens = 25: this is the typical output sequence length for speech
* Num batches = 100: benchmark over multiple batches for a good estimate of runtime

|            |            |            |                 | Vanilla   |          | Flash     |             |
|------------|------------|------------|-----------------|-----------|----------|-----------|-------------|
| Checkpoint | Dec layers | Params / M | Compression / % | VRAM / GB | Time / s | VRAM / GB | Runtime / s |
| tiny       | 4          | 38         | 0               | 1.38      | 17.0     | 1.39      | 16.4        |
|            | 2          | 33         | 13              | 1.35      | 9.3      | 1.36      | 11.0        |
|            |            |            |                 |           |          |           |             |
| base       | 6          | 73         | 0               | 1.52      | 19.4     | 1.53      | 21.3        |
|            | 4          | 64         | 12              | 1.49      | 14.5     | 1.50      | 15.4        |
|            | 2          | 56         | 23              | 1.45      | 9.3      | 1.46      | 9.7         |
|            |            |            |                 |           |          |           |             |
| small      | 12         | 242        | 0               | 2.27      | 39.4     | 2.27      | 43.2        |
|            | 6          | 185        | 24              | 2.03      | 23.3     | 2.04      | 24.2        |
|            | 4          | 166        | 31              | 1.95      | 17.5     | 1.96      | 17.5        |
|            | 2          | 147        | 39              | 1.87      | 12.0     | 1.88      | 12.7        |
|            |            |            |                 |           |          |           |             |
| medium     | 24         | 764        | 0               | 4.21      | 76.0     | 4.21      | 86.6        |
|            | 6          | 462        | 40              | 3.05      | 25.4     | 3.06      | 27.3        |
|            | 4          | 428        | 44              | 2.92      | 20.2     | 2.93      | 19.5        |
|            | 2          | 394        | 48              | 2.79      | 14.0     | 2.80      | 14.3        |

### Larger Batch Size & Sequence Length
Using a larger batch size and sequence length should reveal the benefits of Flash Attention:
* Batch size = 32
* Generated tokens = 256
* Num batches = 10

|            |            |            |                 | Vanilla   |          | Flash     |             |
|------------|------------|------------|-----------------|-----------|----------|-----------|-------------|
| Checkpoint | Dec layers | Params / M | Compression / % | VRAM / GB | Time / s | VRAM / GB | Runtime / s |
| tiny       | 4          | 38         | 0               | 3.23      | 16.1     | 2.09      | 20.7        |
|            | 2          | 33         | 13              | 3.27      | 9.6      | 1.95      | 11.2        |
|            |            |            |                 |           |          |           |             |
| base       | 6          | 73         | 0               | 3.98      | 21.4     | 2.83      | 24.8        |
|            | 4          | 64         | 12              | 3.95      | 15.2     | 2.38      | 20.3        |
|            | 2          | 56         | 23              | 3.91      | 10.1     | 2.14      | 11.5        |
|            |            |            |                 |           |          |           |             |
| small      | 12         | 242        | 0               | 5.99      | 39.1     | 6.07      | 58.5        |
|            | 6          | 185        | 24              | 5.73      | 22.6     | 3.86      | 31.8        |
|            | 4          | 166        | 31              | 5.64      | 19.0     | 3.05      | 24.3        |
|            | 2          | 147        | 39              | 5.55      | 12.5     | 2.89      | 15.2        |
|            |            |            |                 |           |          |           |             |
| medium     | 24         | 764        | 0               | 9.19      | 78.0     | 13.63     | 175.5       |
|            | 6          | 462        | 40              | 7.97      | 29.6     | 5.26      | 51.8        |
|            | 4          | 428        | 44              | 7.83      | 25.3     | 4.38      | 38.1        |
|            | 2          | 394        | 48              | 7.70      | 19.2     | 4.15      | 24.3        |
