#!/usr/bin/env python3
"""plot_holopolymer_analysis.py

Visualize the effectiveness of holopolymer penalty by showing the percentage
of sequences containing consecutive identical amino acids (≥3, ≥4, ≥5, etc.)
across different gamma values.

Compares three conditions:
- γ = 0.0: No holopolymer penalty (baseline)
- γ = 2.0: Medium holopolymer penalty (ablation studies)  
- γ = 4.0: High holopolymer penalty (current results)

Usage in Jupyter cell:
    %run plot_holopolymer_analysis.py
"""
from __future__ import annotations

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import for display in Jupyter
try:
    from IPython.display import display
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False

# Default paths
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths for different gamma values
GAMMA_DATA = {
    "γ = 0.0": {
        "files": [Path.home() / "mog_dfm/ampflow/results/mog/generic/mog_samples.fa"],
        "color": "#5e4c5f"  # dull purple - represents no penalty (problematic)
    },
    "γ = 2.0": {
        "files": [
            Path.home() / "mog_dfm/ampflow/results/mog/ablations/generic_hp_ab_111_1k/mog_samples.fa",
            Path.home() / "mog_dfm/ampflow/results/mog/ablations/generic_hp_ab_110_1k/mog_samples.fa",
            Path.home() / "mog_dfm/ampflow/results/mog/ablations/generic_hp_ab_011_1k/mog_samples.fa"
        ],
        "color": "#1f4e4e"  # dark teal - represents medium penalty
    },
    "γ = 4.0": {
        "files": [Path.home() / "mog_dfm/ampflow/results/mog/generic_hp4_111_2500/mog_samples.fa"],
        "color": "#66b266"  # light green - represents high penalty (best)
    }
}

def parse_fasta(fasta_path: Path) -> list[str]:
    """Parse FASTA file and return list of sequences."""
    sequences = []
    current_seq = ""
    
    with fasta_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line
        
        # Don't forget the last sequence
        if current_seq:
            sequences.append(current_seq)
    
    return sequences

def count_repeats(sequences: list[str], min_length: int = 3) -> int:
    """Count sequences containing ≥min_length consecutive identical amino acids."""
    pattern = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])\1{" + str(min_length - 1) + ",}")
    count = 0
    
    for seq in sequences:
        if pattern.search(seq):
            count += 1
    
    return count

def analyze_holopolymers(gamma_label: str, file_paths: list[Path]) -> dict[int, float]:
    """Analyze holopolymer content for a given gamma condition."""
    all_sequences = []
    
    # Combine sequences from all files for this gamma value
    for fasta_path in file_paths:
        if fasta_path.exists():
            sequences = parse_fasta(fasta_path)
            all_sequences.extend(sequences)
            print(f"Loaded {len(sequences)} sequences from {fasta_path.name}")
        else:
            print(f"Warning: {fasta_path} not found")
    
    if not all_sequences:
        raise FileNotFoundError(f"No sequences found for {gamma_label}")
    
    total_sequences = len(all_sequences)
    print(f"Total sequences for {gamma_label}: {total_sequences}")
    
    # Count sequences with repeats of different lengths
    results = {}
    for repeat_length in range(3, 8):  # Check for ≥3, ≥4, ≥5, ≥6, ≥7 repeats
        count = count_repeats(all_sequences, repeat_length)
        percentage = (count / total_sequences) * 100
        results[repeat_length] = percentage
        print(f"  ≥{repeat_length} repeats: {count}/{total_sequences} ({percentage:.1f}%)")
    
    return results

def plot_holopolymer_comparison(out_dir: Path | str | None = None):
    """Generate grouped bar chart comparing holopolymer content across gamma values."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    print("Analyzing holopolymer content across gamma values...")
    
    # Analyze each gamma condition
    gamma_results = {}
    for gamma_label, data in GAMMA_DATA.items():
        print(f"\nAnalyzing {gamma_label}:")
        results = analyze_holopolymers(gamma_label, data["files"])
        gamma_results[gamma_label] = results
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Apply HydrAMP styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#B8B8B8')
    ax.spines['bottom'].set_color('#B8B8B8')
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_axisbelow(True)
    
    # Prepare data for grouped bar chart
    repeat_lengths = list(range(3, 8))  # [3, 4, 5, 6, 7]
    x_labels = [f"≥{i}" for i in repeat_lengths]
    
    # Set up bar positions
    bar_width = 0.25
    x_pos = np.arange(len(repeat_lengths))
    
    # Plot bars for each gamma value
    gamma_labels = list(GAMMA_DATA.keys())
    for i, gamma_label in enumerate(gamma_labels):
        color = GAMMA_DATA[gamma_label]["color"]
        percentages = [gamma_results[gamma_label][length] for length in repeat_lengths]
        
        bars = ax.bar(x_pos + i * bar_width, percentages, bar_width, 
                     label=gamma_label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, percentage in zip(bars, percentages):
            if percentage > 0.1:  # Only label if percentage is meaningful
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Consecutive Amino Acid Repeats', fontsize=12)
    ax.set_ylabel('Percentage of Sequences (%)', fontsize=12)
    ax.set_title('Holopolymer Penalty Effectiveness', fontsize=14, fontweight='normal', pad=10)
    
    # Set x-axis
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(x_labels)
    
    # Add legend
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save figure
    out_path = Path(out_dir) / "holopolymer_penalty_effectiveness.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nHolopolymer analysis plot saved to {out_path}")
    
    # Display in Jupyter
    if _IN_JUPYTER:
        display(plt.gcf())
    else:
        plt.show()
    plt.close()
    
    # Print summary statistics
    print("\nSummary:")
    for gamma_label in gamma_labels:
        total_with_repeats = gamma_results[gamma_label][3]  # ≥3 repeats
        print(f"{gamma_label}: {total_with_repeats:.1f}% of sequences contain ≥3 consecutive identical AAs")

def main():
    """Main function to run the holopolymer analysis."""
    plot_holopolymer_comparison()

# Auto-run when executed
if __name__ == "__main__":
    main()
elif _IN_JUPYTER:
    # Auto-run when imported or run in Jupyter
    try:
        main()
    except Exception as err:
        print("plot_holopolymer_analysis –", err)
