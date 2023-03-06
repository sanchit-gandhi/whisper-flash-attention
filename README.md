# Whisper Flash Attention

This is a repository for benchmarking the [Whisper Model](https://arxiv.org/abs/2212.04356) with memory efficient multi-head
attention (MHA) from the [xFormers repository](https://github.com/facebookresearch/xformers) by Facebook research.

The modelling code is split into two parts:
* [flash_attention.py](whisper_flash_attention/flash_attention.py): implements memory efficient attention using the xFormers back-end
* [modeling_whisper_mqa.py](whisper_flash_attention/modeling_whisper_flash_attention.py): augments the Hugging Face Transformers Whisper model with memory efficient attention

The benchmark can be run from the Python script in [benchmark_flash_attention.py](benchmark_flash_attention.py). 
By default, it uses:
* Batch size = 1: single-batch inference with greedy decoding
* Num batches = 100: benchmark over multiple batches for a good estimate of runtime
* Generated tokens = 25: this is the typical output sequence length for speech

## Installation
Installation requires [PyTorch 1.13.1](https://pytorch.org/get-started/locally/) or higher, the xFormers package and 
Hugging Face Transformers:
```
pip install -U xformers transformers
```


## Results

