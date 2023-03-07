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

The following table shows the VRAM and inference time results for Original (Orig) vs Flash attention for 
various checkpoints and decoder layers:

| Checkpoint | Dec layers | VRAM Orig / GB | VRAM Flash / GB | Time Orig / s | Time Flash / s |
|------------|------------|----------------|-----------------|---------------|----------------|
| tiny       | 4          | 1.38           | 1.39            | 17.0          | 16.4           |
|            | 2          | 1.35           | 1.36            | 9.3           | 11.0           |
|            |            |                |                 |               |                |
| base       | 6          | 1.52           | 1.53            | 19.4          | 21.3           |
|            | 4          | 1.49           | 1.50            | 14.5          | 15.4           |
|            | 2          | 1.45           | 1.46            | 9.3           | 9.7            |
|            |            |                |                 |               |                |
| small      | 12         | 2.27           | 2.27            | 39.4          | 43.2           |
|            | 6          | 2.03           | 2.04            | 23.3          | 24.2           |
|            | 4          | 1.95           | 1.96            | 17.5          | 17.5           |
|            | 2          | 1.87           | 1.88            | 12.0          | 12.7           |
|            |            |                |                 |               |                |
| medium     | 24         | 4.21           | 4.21            | 76.0          | 86.6           |
|            | 6          | 3.05           | 3.06            | 25.4          | 27.3           |
|            | 4          | 2.92           | 2.93            | 20.2          | 19.5           |
|            | 2          | 2.79           | 2.80            | 14.0          | 14.3           |

### Larger Batch Size & Sequence Length
Using a larger batch size and sequence length should reveal the benefits of Flash Attention:
* Batch size = 32
* Generated tokens = 256
* Num batches = 10

| Checkpoint | Dec layers | VRAM Orig / GB | VRAM Flash / GB | Time Orig / s | Time Flash / s |
|------------|------------|----------------|-----------------|---------------|----------------|
| tiny       | 4          | 3.23           | 2.09            | 16.1          | 20.7           |
|            | 2          | 3.27           | 1.95            | 9.6           | 11.2           |
|            |            |                |                 |               |                |
| base       | 6          | 3.98           | 2.83            | 21.4          | 24.8           |
|            | 4          | 3.95           | 2.38            | 15.2          | 20.3           |
|            | 2          | 3.91           | 2.14            | 10.1          | 11.5           |
|            |            |                |                 |               |                |
| small      | 12         | 5.99           | 6.07            | 39.1          | 58.5           |
|            | 6          | 5.73           | 3.86            | 22.6          | 31.8           |
|            | 4          | 5.64           | 3.05            | 19.0          | 24.3           |
|            | 2          | 5.55           | 2.89            | 12.5          | 15.2           |
|            |            |                |                 |               |                |
| medium     | 24         | 9.19           | 13.63           | 78.0          | 175.5          |
|            | 6          | 7.97           | 5.26            | 29.6          | 51.8           |
|            | 4          | 7.83           | 4.38            | 25.3          | 38.1           |
|            | 2          | 7.70           | 4.15            | 19.2          | 24.3           |
