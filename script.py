"""
Lab 3: KV Cache Analysis in Small Language Models
ECSE 397/600: Efficient Deep Learning
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import psutil
from datetime import datetime

# ============================================================================
# Output file setup
# ============================================================================
report_file = "lab3_report.txt"


def log(message, file_handle):
    """Print to console and write to file"""
    print(message)
    file_handle.write(message + "\n")


# Open report file
with open(report_file, "w") as report:
    log(f"Lab 3: KV Cache Analysis Report", report)
    log(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report)
    log("=" * 70, report)
    log("", report)

    # ========================================================================
    # TASK 1: Load and Inspect Model
    # ========================================================================
    log("TASK 1: Load and Inspect Model", report)
    log("=" * 70, report)

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    model.eval()

    # Inspect the model configuration
    config = model.config
    log(f"Model name: {model_name}", report)
    log(f"Number of layers (n_layer): {config.n_layer}", report)
    log(f"Hidden size (n_embd): {config.n_embd}", report)
    log(f"Attention heads (n_head): {config.n_head}", report)
    log(f"Vocabulary size: {config.vocab_size}", report)
    log(f"Max position embeddings: {config.n_positions}", report)
    log("", report)

    # Store for later calculations
    n_layers = config.n_layer
    hidden_dim = config.n_embd
    n_heads = config.n_head

    # ========================================================================
    # WARMUP: Run a few forward passes to warm up GPU
    # ========================================================================
    log("Running warmup passes to stabilize GPU performance...", report)

    warmup_prompt = "Hello world"
    warmup_ids = tokenizer(
        warmup_prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        for _ in range(10):
            # Warmup without cache
            _ = model(warmup_ids, use_cache=False)
            # Warmup with cache
            outputs = model(warmup_ids, use_cache=True)
            _ = model(
                warmup_ids[:, -1:], past_key_values=outputs.past_key_values, use_cache=True)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log("Warmup complete.\n", report)

    # ========================================================================
    # TASK 2: Generation Without KV Cache
    # ========================================================================
    log("TASK 2: Generation Without KV Cache", report)
    log("=" * 70, report)

    def generate_no_cache(prompt, L):
        """Generate L tokens without using KV cache"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(L):
                outputs = model(input_ids, use_cache=False)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                input_ids = torch.cat(
                    [input_ids, next_token.unsqueeze(-1)], dim=-1)

        torch.cuda.synchronize()
        end_time = time.time()

        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_latency = total_time / L

        return avg_latency

    prompt = "The quick brown fox"
    sequence_lengths = [32, 64, 128, 256]
    num_runs = 3  # Number of runs to average
    results_no_cache = []

    for L in sequence_lengths:
        log(f"\nGenerating {L} tokens without cache ({num_runs} runs)...", report)

        latencies = []
        mem_used = 0

        for run in range(num_runs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            mem_before = torch.cuda.memory_allocated()
            latency = generate_no_cache(prompt, L)
            mem_after = torch.cuda.max_memory_allocated()

            latencies.append(latency)
            mem_used = (mem_after - mem_before) / \
                (1024**2)  # MB (use last run)

        avg_latency = sum(latencies) / len(latencies)

        results_no_cache.append({
            'seq_len': L,
            'latency': avg_latency,
            'memory': mem_used
        })

        log(f"  Sequence length: {L}", report)
        log(f"  Average latency: {avg_latency:.2f} ms/token", report)
        log(f"  Peak memory used: {mem_used:.2f} MB", report)

    # Summary table for Task 2
    log("\n" + "-" * 70, report)
    log("Summary: Without KV Cache", report)
    log("-" * 70, report)
    log(f"{'Seq Length':<15} {'Latency (ms/token)':<20} {'Memory (MB)':<15}", report)
    log("-" * 50, report)
    for r in results_no_cache:
        log(f"{r['seq_len']:<15} {r['latency']:<20.2f} {r['memory']:<15.2f}", report)
    log("", report)

    # ========================================================================
    # TASK 3: Generation With KV Cache
    # ========================================================================
    log("TASK 3: Generation With KV Cache", report)
    log("=" * 70, report)

    def generate_with_cache(prompt, L):
        """Generate L tokens using KV cache"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        past_key_values = None

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            # First forward pass - process the entire prompt
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(
                outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([input_ids, next_token], dim=-1)

            # Subsequent passes - only process the new token
            for _ in range(L - 1):
                outputs = model(
                    next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(
                    outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        torch.cuda.synchronize()
        end_time = time.time()

        total_time = (end_time - start_time) * 1000
        avg_latency = total_time / L

        return avg_latency, past_key_values

    results_with_cache = []

    for L in sequence_lengths:
        log(f"\nGenerating {L} tokens with cache ({num_runs} runs)...", report)

        latencies = []
        mem_used = 0
        kv_cache_size = 0

        for run in range(num_runs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            mem_before = torch.cuda.memory_allocated()
            latency, kv_cache = generate_with_cache(prompt, L)
            mem_after = torch.cuda.max_memory_allocated()

            latencies.append(latency)
            mem_used = (mem_after - mem_before) / \
                (1024**2)  # MB (use last run)

            # Calculate actual KV cache size (use last run)
            kv_cache_size = sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in kv_cache
            ) / (1024**2)  # MB

        avg_latency = sum(latencies) / len(latencies)

        results_with_cache.append({
            'seq_len': L,
            'latency': avg_latency,
            'memory': mem_used,
            'kv_cache_size': kv_cache_size
        })

        log(f"  Sequence length: {L}", report)
        log(f"  Average latency: {avg_latency:.2f} ms/token", report)
        log(f"  Peak memory used: {mem_used:.2f} MB", report)
        log(f"  KV cache size: {kv_cache_size:.4f} MB", report)

    # Summary table for Task 3
    log("\n" + "-" * 70, report)
    log("Summary: With KV Cache", report)
    log("-" * 70, report)
    log(f"{'Seq Len':<10} {'Latency (ms)':<15} {'Memory (MB)':<15} {'KV Cache (MB)':<15}", report)
    log("-" * 55, report)
    for r in results_with_cache:
        log(f"{r['seq_len']:<10} {r['latency']:<15.2f} {r['memory']:<15.2f} {r['kv_cache_size']:<15.4f}", report)
    log("", report)

    # ========================================================================
    # COMPARISON: With Cache vs Without Cache
    # ========================================================================
    log("COMPARISON: Speedup Analysis", report)
    log("=" * 70, report)
    log(f"{'Seq Len':<10} {'No Cache (ms)':<15} {'With Cache (ms)':<18} {'Speedup':<10}", report)
    log("-" * 55, report)
    for nc, wc in zip(results_no_cache, results_with_cache):
        speedup = nc['latency'] / wc['latency']
        log(f"{nc['seq_len']:<10} {nc['latency']:<15.2f} {wc['latency']:<18.2f} {speedup:<10.2f}x", report)
    log("", report)

    # ========================================================================
    # TASK 4: Memory Analysis - Theoretical vs Measured
    # ========================================================================
    log("TASK 4: Memory Analysis", report)
    log("=" * 70, report)
    log("", report)
    log("KV Cache Memory Formula:", report)
    log("  M_KV = 2 × L × D × N_layers × sizeof(dtype)", report)
    log(f"  Where: D = {hidden_dim}, N_layers = {n_layers}, dtype = float32 (4 bytes)", report)
    log("", report)

    log(f"{'Seq Len':<10} {'Theoretical (MB)':<18} {'Measured (MB)':<18} {'Difference':<15}", report)
    log("-" * 60, report)

    for r in results_with_cache:
        L = r['seq_len']
        # Add prompt length to sequence length for total tokens in cache
        prompt_len = len(tokenizer(prompt).input_ids)
        total_len = prompt_len + L

        # Theoretical: 2 (K and V) × seq_len × hidden_dim × n_layers × 4 bytes (float32)
        theoretical_bytes = 2 * total_len * hidden_dim * n_layers * 4
        theoretical_mb = theoretical_bytes / (1024**2)

        measured_mb = r['kv_cache_size']
        diff_percent = ((measured_mb - theoretical_mb) /
                        theoretical_mb) * 100 if theoretical_mb > 0 else 0

        log(f"{L:<10} {theoretical_mb:<18.4f} {measured_mb:<18.4f} {diff_percent:>+.2f}%", report)

    log("", report)

    # ========================================================================
    # TASK 5 (Bonus): Quantized KV Cache (8-bit)
    # ========================================================================
    log("TASK 5 (Bonus): Quantized KV Cache Analysis", report)
    log("=" * 70, report)

    try:
        # Check if bitsandbytes is available for quantization
        import bitsandbytes as bnb
        has_bnb = True
    except ImportError:
        has_bnb = False
        log("Note: bitsandbytes not installed. Showing theoretical analysis only.", report)

    log("", report)
    log("Quantization reduces KV cache memory by storing in int8 instead of float32.", report)
    log("Theoretical memory savings: 4x (32-bit -> 8-bit)", report)
    log("", report)

    log(f"{'Seq Len':<10} {'FP32 Cache (MB)':<18} {'INT8 Cache (MB)':<18} {'Savings':<15}", report)
    log("-" * 60, report)

    for r in results_with_cache:
        L = r['seq_len']
        fp32_mb = r['kv_cache_size']
        int8_mb = fp32_mb / 4  # Theoretical 4x reduction
        savings = (1 - int8_mb / fp32_mb) * 100

        log(f"{L:<10} {fp32_mb:<18.4f} {int8_mb:<18.4f} {savings:.1f}%", report)

    log("", report)


print(f"\n✓ All tasks completed! Results saved to {report_file}")
