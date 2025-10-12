#!/usr/bin/env python3
"""plot_model_hits_comparison.py

Visualize AMP potency hits comparison across different models (AMP-DFM vs PepDFM)
showing hits for different bacterial species in our standard plotting style.

Usage in Jupyter cell:
    %run plot_model_hits_comparison.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
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

DEFAULT_SUMMARY_CSV = Path.home() / "mog_dfm/ampflow/results/model_panel/summary_metrics.csv"

# Colors for each bacterial species (same as densities_stylised.py but adapted for species)
SPECIES_COLORS = {
    "Generic": "#1f78b4",       # dark blue (guided)
    "S.aureus": "#EDA355",      # yellowish gold
    "P.aeruginosa": "#A82223",  # cherry red
    "E.coli": "#77C99D"         # turquoise
}

def load_model_comparison_data(csv_path: Path = DEFAULT_SUMMARY_CSV) -> pd.DataFrame:
    """Load the model comparison summary data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Clean up model labels
    df['clean_label'] = df['label'].str.replace('AMP-DFM (Unconditional)', 'AMP-DFM (Unconditional)')
    df['clean_label'] = df['clean_label'].str.replace('AMP-DFM (Conditional)', 'AMP-DFM (Conditional)')
    df['clean_label'] = df['clean_label'].str.replace('Pep-DFM', 'PepDFM')
    
    return df

def plot_model_hits_comparison(csv_path: Path = DEFAULT_SUMMARY_CSV, out_dir: Path | str | None = None):
    """Generate grouped bar chart comparing potency hits across models and species."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    print("Loading model comparison data...")
    df = load_model_comparison_data(csv_path)
    
    # Extract data for plotting
    models = df['clean_label'].tolist()
    generic_hits = df['hits_generic'].values
    sa_hits = df['hits_sa'].values
    pa_hits = df['hits_pa'].values
    ec_hits = df['hits_ec'].values
    
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
    
    # Set up bar positions
    bar_width = 0.18
    x_pos = np.arange(len(models))
    
    # Plot bars for each species
    bars_generic = ax.bar(x_pos - 1.5*bar_width, generic_hits, bar_width, 
                         label="Generic", color=SPECIES_COLORS["Generic"], alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
    
    bars_sa = ax.bar(x_pos - 0.5*bar_width, sa_hits, bar_width, 
                    label="S.aureus", color=SPECIES_COLORS["S.aureus"], alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    
    bars_pa = ax.bar(x_pos + 0.5*bar_width, pa_hits, bar_width, 
                    label="P.aeruginosa", color=SPECIES_COLORS["P.aeruginosa"], alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    
    bars_ec = ax.bar(x_pos + 1.5*bar_width, ec_hits, bar_width, 
                    label="E.coli", color=SPECIES_COLORS["E.coli"], alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            if value > 0:  # Only label if there are hits
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                       f'{value}', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars_generic, generic_hits)
    add_labels(bars_sa, sa_hits)
    add_labels(bars_pa, pa_hits)
    add_labels(bars_ec, ec_hits)
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Potency Hits (probability ≥ 0.8)', fontsize=12)
    ax.set_title('AMP Potency Hits by Model and Bacterial Species', 
                fontsize=14, fontweight='normal', pad=10)
    
    # Set x-axis - horizontal labels with enough space
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11, ha='center')
    
    # Add legend
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Set y-axis to start from 0 with some padding at top
    max_hits = max(max(generic_hits), max(sa_hits), max(pa_hits), max(ec_hits))
    ax.set_ylim(0, max_hits * 1.15)
    
    plt.tight_layout()
    
    # Save figure
    out_path = Path(out_dir) / "model_potency_hits_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nModel hits comparison plot saved to {out_path}")
    
    # Display in Jupyter
    if _IN_JUPYTER:
        display(plt.gcf())
    else:
        plt.show()
    plt.close()
    
    # Print summary statistics
    print("\nSummary:")
    for i, model in enumerate(models):
        print(f"{model}:")
        print(f"  Generic: {generic_hits[i]} hits")
        print(f"  S.aureus: {sa_hits[i]} hits") 
        print(f"  P.aeruginosa: {pa_hits[i]} hits")
        print(f"  E.coli: {ec_hits[i]} hits")
        total = generic_hits[i] + sa_hits[i] + pa_hits[i] + ec_hits[i]
        print(f"  Total: {total} hits")
        print()

def main():
    """Main function to run the model hits comparison."""
    plot_model_hits_comparison()

# Auto-run when executed
if __name__ == "__main__":
    main()
elif _IN_JUPYTER:
    # Auto-run when imported or run in Jupyter
    try:
        main()
    except Exception as err:
        print("plot_model_hits_comparison –", err)
