#!/usr/bin/env python3
"""violin_potency.py

Violin plots showing potency score distributions across different sampling methods
in the style of HydrAMP paper. Shows baseline, generic, ecoli, paeruginosa, and 
saureus sampling conditions.

Usage in Jupyter cell:
    %run violin_potency.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Default paths
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths for each sampling method
DATA_PATHS = {
    "Baseline": Path.home() / "mog_dfm/ampflow/results/mog/baseline/baseline_scores.csv",
    "Generic": Path.home() / "mog_dfm/ampflow/results/mog/generic/mog_samples_scores.csv", 
    "E.coli": Path.home() / "mog_dfm/ampflow/results/mog/ecoli/mog_samples_scores.csv",
    "P.aeruginosa": Path.home() / "mog_dfm/ampflow/results/mog/paeruginosa/mog_samples_scores.csv",
    "S.aureus": Path.home() / "mog_dfm/ampflow/results/mog/saureus/mog_samples_scores.csv"
}

# Colors for each method
COLORS = {
    "Baseline": "#FA8072",      # salmon (from densities_stylised)
    "Generic": "#1f78b4",       # dark blue (from densities_stylised)  
    "E.coli": "#77C99D",        # turquoise (HydrAMP ecoli color)
    "P.aeruginosa": "#A82223",  # cherry red
    "S.aureus": "#EDA355"       # yellowish gold
}

def load_and_combine_data():
    """Load all property data from all sampling methods and combine into single DataFrame."""
    combined_data = []
    
    for method, path in DATA_PATHS.items():
        if path.exists():
            df = pd.read_csv(path)
            required_cols = ["potency", "hemolysis", "cytotox"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {path}")
                continue
            
            # Extract all property scores and add method label
            # Invert hemolysis & cytotox so that higher = safer (like densities_stylised)
            method_data = pd.DataFrame({
                'potency': df['potency'],
                'hemolysis': 1.0 - df['hemolysis'],  # Invert: higher = less hemolytic (safer)
                'cytotox': 1.0 - df['cytotox'],     # Invert: higher = less cytotoxic (safer)
                'method': method
            })
            combined_data.append(method_data)
            print(f"Loaded {len(df)} samples from {method}")
        else:
            print(f"Warning: File not found - {path}")
    
    if not combined_data:
        raise FileNotFoundError("No valid data files found")
    
    return pd.concat(combined_data, ignore_index=True)

def plot_violin_property(property_name: str, out_dir: Path | str | None = None):
    """Create HydrAMP-style violin plot for a given property (potency, hemolysis, or cytotox).
    
    Parameters
    ----------
    property_name : str
        The property to plot: 'potency', 'hemolysis', or 'cytotox'
    out_dir : Path | str | None, optional
        Output directory for PNG files
    """
    # Load data
    df = load_and_combine_data()
    
    # Check if property exists
    if property_name not in df.columns:
        raise ValueError(f"Property '{property_name}' not found in data")
    
    # Create figure with HydrAMP styling
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Apply HydrAMP styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#B8B8B8')  # Darker grey
    ax.spines['bottom'].set_color('#B8B8B8')  # Darker grey
    
    # Add grid with same thickness as borders
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_axisbelow(True)  # Grid behind plot elements
    
    # Create violin plot
    # Define order to match the desired arrangement
    method_order = ["Baseline", "Generic", "E.coli", "P.aeruginosa", "S.aureus"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    # Create color palette in the correct order
    palette = [COLORS[method] for method in method_order]
    
    # Create violin plot
    violin_parts = ax.violinplot(
        [df[df['method'] == method][property_name].values for method in method_order],
        positions=range(len(method_order)),
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Style the violins
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    # Add HydrAMP-style quartile lines and median markers
    for i, method in enumerate(method_order):
        method_data = df[df['method'] == method][property_name].values
        
        # Calculate quartiles
        q25 = np.percentile(method_data, 25)
        q50 = np.percentile(method_data, 50)  # median
        q75 = np.percentile(method_data, 75)
        
        # Add vertical black line for interquartile range (like HydrAMP)
        ax.plot([i, i], [q25, q75], color='black', linewidth=2, zorder=5)
        
        # Add median as white circle with black border (like HydrAMP)
        ax.scatter(i, q50, s=80, color='white', 
                  edgecolors='black', linewidth=2, zorder=10)
    
    # Customize plot based on property
    ax.set_xlim(-0.5, len(method_order) - 0.5)
    ax.set_ylim(0, 1)  # All properties are 0-1 probabilities
    
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=11)
    
    # Set appropriate y-axis label
    if property_name == 'potency':
        ylabel = 'P$_{AMP}$'
        filename = "violin_potency_distributions.png"
    elif property_name == 'hemolysis':
        ylabel = 'P$_{Hemolysis}$'
        filename = "violin_hemolysis_distributions.png"
    elif property_name == 'cytotox':
        ylabel = 'P$_{Cytotoxicity}$'
        filename = "violin_cytotox_distributions.png"
    else:
        ylabel = f'P$_{{{property_name}}}$'
        filename = f"violin_{property_name}_distributions.png"
    
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Keep x-axis labels horizontal (perpendicular to x-axis)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    
    # Save figure
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def plot_all_violin_properties(out_dir: Path | str | None = None):
    """Generate violin plots for all three properties: potency, hemolysis, and cytotoxicity."""
    properties = ['potency', 'hemolysis', 'cytotox']
    for prop in properties:
        print(f"\nGenerating violin plot for {prop}...")
        plot_violin_property(prop, out_dir)

# Run when executed
if __name__ == "__main__":
    plot_all_violin_properties()
