"""
Generate plots from existing report.txt data
Run locally - no GPU required
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# DATA FROM report.txt (extracted from your results)
# ============================================================================

# Model configuration
n_layers = 6
hidden_dim = 768
prompt_len = 4  # "The quick brown fox" = 4 tokens

# Sequence lengths tested
seq_lens = [32, 64, 128, 256]

# Results WITHOUT KV Cache (from report.txt)
latency_no_cache = [4.48, 4.48, 4.48, 4.58]  # ms/token
memory_no_cache = [13.33, 25.70, 51.31, 100.76]  # MB

# Results WITH KV Cache (from report.txt)
latency_with_cache = [4.52, 4.52, 4.52, 4.52]  # ms/token
memory_with_cache = [2.81, 5.06, 9.56, 18.57]  # MB
kv_cache_measured = [1.2305, 2.3555, 4.6055, 9.1055]  # MB

# Calculate theoretical KV cache sizes
kv_cache_theoretical = []
for L in seq_lens:
    total_len = prompt_len + L
    theoretical_bytes = 2 * total_len * hidden_dim * n_layers * 4
    theoretical_mb = theoretical_bytes / (1024**2)
    kv_cache_theoretical.append(theoretical_mb)

# ============================================================================
# PLOT 1: Latency vs Sequence Length
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(seq_lens, latency_no_cache, 'o-', linewidth=2, markersize=10, 
         label='Without KV Cache', color='#e74c3c')
plt.plot(seq_lens, latency_with_cache, 's-', linewidth=2, markersize=10, 
         label='With KV Cache', color='#2ecc71')
plt.xlabel('Sequence Length (tokens)', fontsize=12)
plt.ylabel('Average Latency (ms/token)', fontsize=12)
plt.title('Latency vs Sequence Length\n(distilgpt2)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(seq_lens)
plt.ylim(0, max(max(latency_no_cache), max(latency_with_cache)) * 1.2)
plt.tight_layout()
plt.savefig('plot1_latency_vs_sequence.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plot1_latency_vs_sequence.png")

# ============================================================================
# PLOT 2: Memory Usage Comparison (Bar Chart)
# ============================================================================
plt.figure(figsize=(10, 6))
x = range(len(seq_lens))
width = 0.35
bars1 = plt.bar([i - width/2 for i in x], memory_no_cache, width, 
                label='Without KV Cache', color='#e74c3c', alpha=0.8)
bars2 = plt.bar([i + width/2 for i in x], memory_with_cache, width, 
                label='With KV Cache', color='#2ecc71', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.xlabel('Sequence Length (tokens)', fontsize=12)
plt.ylabel('Peak Memory Usage (MB)', fontsize=12)
plt.title('Memory Usage vs Sequence Length\n(distilgpt2)', fontsize=14, fontweight='bold')
plt.xticks(x, seq_lens)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot2_memory_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plot2_memory_comparison.png")

# ============================================================================
# PLOT 3: Theoretical vs Measured KV Cache Memory
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(seq_lens, kv_cache_theoretical, 'o--', linewidth=2, markersize=10, 
         label='Theoretical', color='#3498db')
plt.plot(seq_lens, kv_cache_measured, 's-', linewidth=2, markersize=10, 
         label='Measured', color='#e67e22')

# Add percentage difference annotations
for i, (theo, meas) in enumerate(zip(kv_cache_theoretical, kv_cache_measured)):
    diff = ((meas - theo) / theo) * 100
    plt.annotate(f'{diff:+.1f}%', (seq_lens[i], meas), 
                 textcoords="offset points", xytext=(10, 5), fontsize=9)

plt.xlabel('Sequence Length (tokens)', fontsize=12)
plt.ylabel('KV Cache Size (MB)', fontsize=12)
plt.title('Theoretical vs Measured KV Cache Memory\n(distilgpt2)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(seq_lens)
plt.tight_layout()
plt.savefig('plot3_kv_cache_theoretical_vs_measured.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plot3_kv_cache_theoretical_vs_measured.png")

# ============================================================================
# PLOT 4: Combined Analysis (2x2 subplot)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Latency comparison
axes[0, 0].plot(seq_lens, latency_no_cache, 'o-', linewidth=2, markersize=8, 
                label='Without Cache', color='#e74c3c')
axes[0, 0].plot(seq_lens, latency_with_cache, 's-', linewidth=2, markersize=8, 
                label='With Cache', color='#2ecc71')
axes[0, 0].set_xlabel('Sequence Length (tokens)')
axes[0, 0].set_ylabel('Latency (ms/token)')
axes[0, 0].set_title('Latency Comparison', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(seq_lens)

# Subplot 2: Memory comparison
axes[0, 1].plot(seq_lens, memory_no_cache, 'o-', linewidth=2, markersize=8, 
                label='Without Cache', color='#e74c3c')
axes[0, 1].plot(seq_lens, memory_with_cache, 's-', linewidth=2, markersize=8, 
                label='With Cache', color='#2ecc71')
axes[0, 1].set_xlabel('Sequence Length (tokens)')
axes[0, 1].set_ylabel('Peak Memory (MB)')
axes[0, 1].set_title('Peak Memory Usage', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(seq_lens)

# Subplot 3: Theoretical vs Measured KV Cache
axes[1, 0].plot(seq_lens, kv_cache_theoretical, 'o--', linewidth=2, markersize=8, 
                label='Theoretical', color='#3498db')
axes[1, 0].plot(seq_lens, kv_cache_measured, 's-', linewidth=2, markersize=8, 
                label='Measured', color='#e67e22')
axes[1, 0].set_xlabel('Sequence Length (tokens)')
axes[1, 0].set_ylabel('KV Cache Size (MB)')
axes[1, 0].set_title('KV Cache: Theoretical vs Measured', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(seq_lens)

# Subplot 4: Memory Savings
memory_savings = [(nc - wc) / nc * 100 for nc, wc in zip(memory_no_cache, memory_with_cache)]
bars = axes[1, 1].bar(seq_lens, memory_savings, color='#9b59b6', alpha=0.8, width=20)
for bar, saving in zip(bars, memory_savings):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{saving:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1, 1].set_xlabel('Sequence Length (tokens)')
axes[1, 1].set_ylabel('Memory Savings (%)')
axes[1, 1].set_title('Memory Savings with KV Cache', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 100)

plt.suptitle('KV Cache Analysis - distilgpt2', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot4_combined_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plot4_combined_analysis.png")

print("\n✓ All plots generated successfully!")
print("\nPlots created:")
print("  1. plot1_latency_vs_sequence.png")
print("  2. plot2_memory_comparison.png")
print("  3. plot3_kv_cache_theoretical_vs_measured.png")
print("  4. plot4_combined_analysis.png")

