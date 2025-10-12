#!/usr/bin/env python3
"""amino_acid_frequencies_hydramp_vs_ampdfm.py

Amino acid frequency comparison across HydrAMP vs AMP-DFM models in the style of HydrAMP:
- Faint grid background
- No top and right borders (spines)
- Consistent colors across datasets
- Shows frequency of each of the 20 standard amino acids
- Compares HydrAMP vs AMP-DFM models (Unguided, AMP-Only, Full-Guidance)

Designed to be run directly in Jupyter notebook cells.

Usage in Jupyter cell:
    %run amino_acid_frequencies_hydramp_vs_ampdfm.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns

# Default paths
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths for HydrAMP vs AMP-DFM comparison
DATA_PATHS = {
    "HydrAMP": Path.home() / "mog_dfm/ampflow/results/hydramp_vs_ampdfm/hydramp_len10_25_2500_scored_with_props.csv",
    "AMP-DFM Unguided": Path.home() / "mog_dfm/ampflow/results/hydramp_vs_ampdfm/unguided_len10_25_2500_scores.csv",
    "AMP-DFM AMP-Only": Path.home() / "mog_dfm/ampflow/results/hydramp_vs_ampdfm/potency_only_len10_25_2500_scores.csv",
    "AMP-DFM Full-Guidance": Path.home() / "mog_dfm/ampflow/results/hydramp_vs_ampdfm/full_mog_len10_25_2500_scores.csv"
}

# Colors for each model (consistent with other visualization scripts)
COLORS = {
    "HydrAMP": "#9D4EDD",           # Purple for HydrAMP
    "AMP-DFM Unguided": "#FF6B6B", # Coral red for unguided
    "AMP-DFM AMP-Only": "#4ECDC4", # Teal for AMP-only (1,0,0)
    "AMP-DFM Full-Guidance": "#45B7D1"  # Sky blue for full guidance (1,1,1)
}

# Standard 20 amino acids
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def calculate_amino_acid_frequencies(sequences: list[str]) -> dict[str, float]:
    """Calculate the frequency of each amino acid in a list of sequences.
    
    Parameters
    ----------
    sequences : list[str]
        List of peptide sequences
        
    Returns
    -------
    dict[str, float]
        Dictionary mapping amino acid to frequency (0-1)
    """
    # Count all amino acids across all sequences
    all_aa_counts = Counter()
    total_residues = 0
    
    for sequence in sequences:
        # Convert to uppercase and filter to only standard amino acids
        clean_seq = ''.join([aa.upper() for aa in sequence if aa.upper() in AMINO_ACIDS])
        all_aa_counts.update(clean_seq)
        total_residues += len(clean_seq)
    
    # Calculate frequencies
    frequencies = {}
    for aa in AMINO_ACIDS:
        frequencies[aa] = all_aa_counts.get(aa, 0) / total_residues if total_residues > 0 else 0.0
    
    return frequencies

def load_and_calculate_aa_frequencies():
    """Load sequences from all models and calculate amino acid frequencies."""
    method_frequencies = {}
    
    for method, path in DATA_PATHS.items():
        if path.exists():
            df = pd.read_csv(path)
            if "sequence" not in df.columns:
                print(f"Warning: 'sequence' column not found in {path}")
                continue
            
            sequences = df['sequence'].tolist()
            print(f"Processing {len(sequences)} sequences from {method}...")
            
            frequencies = calculate_amino_acid_frequencies(sequences)
            method_frequencies[method] = frequencies
            
            print(f"Completed amino acid frequency calculation for {method}")
        else:
            print(f"Warning: File not found - {path}")
    
    if not method_frequencies:
        raise FileNotFoundError("No valid data files found")
    
    return method_frequencies

def apply_hydramp_styling(ax):
    """Apply consistent HydrAMP styling to an axis."""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#B8B8B8')  # Darker grey
    ax.spines['bottom'].set_color('#B8B8B8')  # Darker grey
    
    # Add grid with same thickness as borders
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_axisbelow(True)  # Grid behind plot elements
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)

def add_significance_markers(ax, x_positions, method_frequencies, significance_threshold=0.01):
    """Add significance markers (✓ or ✗) above bars for notable differences."""
    # Get method order
    method_order = ["HydrAMP", "AMP-DFM Unguided", "AMP-DFM AMP-Only", "AMP-DFM Full-Guidance"]
    method_order = [m for m in method_order if m in method_frequencies.keys()]
    
    # Calculate which amino acids show notable patterns
    for i, aa in enumerate(AMINO_ACIDS):
        frequencies = [method_frequencies[method][aa] for method in method_order]
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        
        # Mark as significant if there's a substantial difference between methods
        if max_freq - min_freq > significance_threshold:
            # Add checkmark above the bars for this amino acid
            y_max = ax.get_ylim()[1]
            ax.text(x_positions[i], y_max * 0.95, '✓', 
                   ha='center', va='center', fontsize=12, color='green', fontweight='bold')

def plot_amino_acid_frequencies(out_dir: Path | str | None = None):
    """Generate amino acid frequency comparison plot for HydrAMP vs AMP-DFM models."""
    print("Loading sequences and calculating amino acid frequencies...")
    method_frequencies = load_and_calculate_aa_frequencies()
    
    # Fall back to default directory
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR

    print(f"Saving amino acid frequency plot to: {out_dir}")

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define method order to match other plots
    method_order = ["HydrAMP", "AMP-DFM Unguided", "AMP-DFM AMP-Only", "AMP-DFM Full-Guidance"]
    method_order = [m for m in method_order if m in method_frequencies.keys()]
    
    # Set up bar positions
    x = np.arange(len(AMINO_ACIDS))
    width = 0.2  # Width of each bar
    n_methods = len(method_order)
    
    # Calculate bar positions for each method
    bar_positions = []
    for i in range(n_methods):
        offset = (i - (n_methods - 1) / 2) * width
        bar_positions.append(x + offset)
    
    # Create bars for each method
    for i, method in enumerate(method_order):
        frequencies = [method_frequencies[method][aa] for aa in AMINO_ACIDS]
        color = COLORS[method]
        
        bars = ax.bar(bar_positions[i], frequencies, width, 
                     label=method, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Apply HydrAMP styling
    apply_hydramp_styling(ax)
    
    # Skip significance markers for cleaner look
    # add_significance_markers(ax, x, method_frequencies)
    
    # Customize plot
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Amino Acid Frequencies Across HydrAMP vs AMP-DFM Models', 
                fontsize=14, fontweight='normal', pad=15)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(AMINO_ACIDS)
    
    # Set y-axis limits
    ax.set_ylim(0, None)
    
    # Create legend with HydrAMP styling
    legend = ax.legend(loc='upper right', frameon=True, fontsize=10, 
                      borderpad=0.8, columnspacing=1.5)
    legend.get_frame().set_edgecolor('#B8B8B8')  # Same grey as plot borders
    legend.get_frame().set_linewidth(1.2)        # Same thickness as plot borders
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    plt.tight_layout()
    
    # Save figure
    out_path = Path(out_dir) / "amino_acid_frequencies_hydramp_vs_ampdfm.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Amino acid frequency plot saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def print_frequency_summary():
    """Print a summary table of amino acid frequencies for each method."""
    print("\nLoading data for frequency summary...")
    method_frequencies = load_and_calculate_aa_frequencies()
    
    # Create DataFrame for easy viewing
    df_summary = pd.DataFrame(method_frequencies).T
    df_summary = df_summary.round(4)
    
    print("\nAmino Acid Frequency Summary:")
    print("=" * 80)
    print(df_summary.to_string())
    
    # Calculate and print some statistics
    print("\n\nSummary Statistics:")
    print("=" * 50)
    for method in method_frequencies.keys():
        frequencies = list(method_frequencies[method].values())
        print(f"{method}:")
        print(f"  Mean frequency: {np.mean(frequencies):.4f}")
        print(f"  Std frequency:  {np.std(frequencies):.4f}")
        print(f"  Max frequency:  {np.max(frequencies):.4f} ({AMINO_ACIDS[np.argmax(frequencies)]})")
        print(f"  Min frequency:  {np.min(frequencies):.4f} ({AMINO_ACIDS[np.argmin(frequencies)]})")
        print()

# Run when executed
if __name__ == "__main__":
    plot_amino_acid_frequencies()
    print_frequency_summary()
