# KV Cache Analysis in Language Models

Lab 3 for ECSE 397/600: Efficient Deep Learning

## Overview

This project analyzes how Key-Value caching affects performance in transformer-based language models. Using distilgpt2, we measure the computational and memory trade-offs between cached and non-cached autoregressive generation.

## What is KV Caching?

During text generation, transformers normally reprocess all previous tokens at each step (O(L²) complexity). KV caching stores the key and value tensors from past tokens so only new token attention is computed (O(L) complexity). This speeds up generation but uses more memory.

## Tasks

1. **Model Inspection** - Load distilgpt2 and examine its architecture
2. **No Cache Baseline** - Measure latency without caching across different sequence lengths
3. **Cached Generation** - Compare performance with KV cache enabled
4. **Memory Analysis** - Plot theoretical vs measured cache memory usage
5. **Bonus: Quantization** - Test 8-bit cache compression

## Requirements

- PyTorch
- Transformers (Hugging Face)
- CUDA-capable GPU
- psutil

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
```

Run experiments for sequence lengths: 32, 64, 128, 256 tokens

## Metrics

- Inference latency (ms/token)
- GPU memory usage
- KV cache size: `M = 2 × L × D × N_L × 2 bytes`
