#!/usr/bin/env python3
"""physicochemical_hydramp_vs_ampdfm.py

Compute physicochemical descriptors using modlAMP and create HydrAMP-style 
violin plots comparing HydrAMP vs AMP-DFM models.

Computes the same four descriptors as HydrAMP:
- Isoelectric point (pI)
- Charge (at pH 7.0)
- Hydrophobic ratio
- Aromaticity

For HydrAMP data, uses pre-computed descriptors where available.
For AMP-DFM models, computes descriptors from sequences.

Usage in Jupyter cell:
    %run physicochemical_hydramp_vs_ampdfm.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import modlAMP for descriptor calculations
try:
    from modlamp.analysis import GlobalDescriptor, GlobalAnalysis
except ImportError:
    raise ImportError("modlAMP is required. Install with: pip install modlamp")

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

# Colors for each model (same as densities script for consistency)
COLORS = {
    "HydrAMP": "#9D4EDD",           # Purple for HydrAMP
    "AMP-DFM Unguided": "#FF6B6B", # Coral red for unguided
    "AMP-DFM AMP-Only": "#4ECDC4", # Teal for AMP-only (1,0,0)
    "AMP-DFM Full-Guidance": "#45B7D1"  # Sky blue for full guidance (1,1,1)
}

def compute_descriptors(sequences: list[str]) -> pd.DataFrame:
    """Compute the four physicochemical descriptors using modlAMP defaults.
    
    Parameters
    ----------
    sequences : list[str]
        List of peptide sequences
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: isoelectric_point, charge, hydrophobic_ratio, aromaticity
    """
    # modlAMP expects a flat list of sequences
    
    # Compute charge using GlobalAnalysis
    ga = GlobalAnalysis(sequences)
    ga.calc_charge()  # Net charge at pH 7.0 with EMBOSS pKa
    charges = np.array(ga.charge).flatten()
    
    # Compute other descriptors using GlobalDescriptor
    gd = GlobalDescriptor(sequences)
    
    # Isoelectric point
    gd.isoelectric_point()
    isoelectric_points = np.array(gd.descriptor).flatten()
    
    # Aromaticity (fraction of F, Y, W)
    gd.aromaticity()
    aromaticities = np.array(gd.descriptor).flatten()
    
    # Hydrophobic ratio (fraction of A, I, L, M, F, W, V)
    gd.hydrophobic_ratio()
    hydrophobic_ratios = np.array(gd.descriptor).flatten()
    
    return pd.DataFrame({
        'isoelectric_point': isoelectric_points,
        'charge': charges,
        'hydrophobic_ratio': hydrophobic_ratios,
        'aromaticity': aromaticities
    })

def load_and_compute_comparison_descriptors():
    """Load sequences from all models and compute/extract descriptors."""
    combined_data = []
    
    for method, path in DATA_PATHS.items():
        if path.exists():
            df = pd.read_csv(path)
            if "sequence" not in df.columns:
                print(f"Warning: 'sequence' column not found in {path}")
                continue
            
            sequences = df['sequence'].tolist()
            print(f"Processing {len(sequences)} sequences from {method}...")
            
            # Recompute all descriptors for every method (including HydrAMP) using modlAMP to ensure
            # full consistency across groups. Any pre-computed columns in the CSV are ignored.
            print(f"Computing descriptors for {method}")
            descriptors_df = compute_descriptors(sequences)
            
            descriptors_df['method'] = method
            combined_data.append(descriptors_df)
            print(f"Completed {method}")
        else:
            print(f"Warning: File not found - {path}")
    
    if not combined_data:
        raise FileNotFoundError("No valid data files found")
    
    return pd.concat(combined_data, ignore_index=True)

def plot_descriptor_violin_subplot(ax, df: pd.DataFrame, descriptor_name: str, ylabel: str, show_legend: bool = False):
    """Create HydrAMP-style violin plot for a descriptor on a given subplot axis."""
    
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
    
    # Define order to match density plots
    method_order = ["HydrAMP", "AMP-DFM Unguided", "AMP-DFM AMP-Only", "AMP-DFM Full-Guidance"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    # Create color palette in the correct order
    palette = [COLORS[method] for method in method_order]
    
    # Filter out any NaN values for this descriptor
    filtered_data = []
    filtered_methods = []
    for method in method_order:
        method_data = df[df['method'] == method][descriptor_name].dropna()
        if len(method_data) > 0:
            filtered_data.append(method_data.values)
            filtered_methods.append(method)
    
    if len(filtered_data) == 0:
        print(f"Warning: No valid data for {descriptor_name}")
        return
    
    # Create violin plot
    violin_parts = ax.violinplot(
        filtered_data,
        positions=range(len(filtered_methods)),
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Style the violins
    for i, pc in enumerate(violin_parts['bodies']):
        method = filtered_methods[i]
        color = COLORS[method]
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    # Add HydrAMP-style quartile lines and median markers
    for i, method in enumerate(filtered_methods):
        method_data = df[df['method'] == method][descriptor_name].dropna().values
        
        if len(method_data) == 0:
            continue
            
        # Calculate quartiles
        q25 = np.percentile(method_data, 25)
        q50 = np.percentile(method_data, 50)  # median
        q75 = np.percentile(method_data, 75)
        
        # Add vertical black line for interquartile range
        ax.plot([i, i], [q25, q75], color='black', linewidth=2, zorder=5)
        
        # Add median as white circle with black border
        ax.scatter(i, q50, s=80, color='white', 
                  edgecolors='black', linewidth=2, zorder=10)
    
    # Customize plot
    ax.set_xlim(-0.5, len(filtered_methods) - 0.5)
    ax.set_xticks(range(len(filtered_methods)))
    ax.set_xticklabels(filtered_methods, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Make x-axis labels vertical/perpendicular
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    
    # Add legend labels only to first subplot
    if show_legend:
        for i, method in enumerate(filtered_methods):
            # Create invisible artists for legend
            ax.scatter([], [], color=COLORS[method], s=100, alpha=0.7, label=method)

def plot_all_descriptor_violins_comparison(out_dir: Path | str | None = None):
    """Generate combined violin plots for all four physicochemical descriptors comparing HydrAMP vs AMP-DFM."""
    print("Loading sequences and computing descriptors for HydrAMP vs AMP-DFM comparison...")
    df = load_and_compute_comparison_descriptors()
    
    # Fall back to default directory
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR

    print(f"Saving HydrAMP vs AMP-DFM descriptor violin plots to: {out_dir}")

    # Create figure with 4 subplots (1x4 grid)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    descriptors_labels = [
        ('isoelectric_point', 'Isoelectric Point', 'Predicted Isoelectric Point'),
        ('charge', 'Net Charge', 'Predicted Net Charge'),
        ('hydrophobic_ratio', 'Hydrophobic Ratio', 'Predicted Hydrophobic Ratio'),
        ('aromaticity', 'Aromaticity', 'Predicted Aromaticity')
    ]
    
    # Plot each descriptor in its own subplot
    for i, (descriptor, ylabel, title) in enumerate(descriptors_labels):
        print(f"Generating violin plot for {descriptor}...")
        plot_descriptor_violin_subplot(axes[i], df, descriptor, ylabel, show_legend=(i == 0))
        # Add clean subplot title - larger font, no bold
        axes[i].set_title(title, fontsize=14, fontweight='normal', pad=10)
    
    # Remove individual legends (only if they exist)
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for bottom legend and vertical labels
    
    # Create shared legend at bottom with HydrAMP-style border
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5,-0.20), 
                       frameon=True, fontsize=10, ncol=4, 
                       borderpad=0.8, columnspacing=1.5)
    
    # Style the legend frame to match plot borders
    legend.get_frame().set_edgecolor('#B8B8B8')  # Same grey as plot borders
    legend.get_frame().set_linewidth(1.2)        # Same thickness as plot borders
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    # Save combined figure
    out_path = Path(out_dir) / "violin_hydramp_vs_ampdfm_physicochemical_descriptors.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"HydrAMP vs AMP-DFM descriptor violin plot saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def plot_individual_descriptor_violins_comparison(out_dir: Path | str | None = None):
    """Generate individual violin plots for each physicochemical descriptor."""
    print("Loading sequences and computing descriptors...")
    df = load_and_compute_comparison_descriptors()
    
    # Fall back to default directory
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    descriptors_info = [
        ('isoelectric_point', 'Isoelectric Point', "violin_hydramp_vs_ampdfm_isoelectric_point.png"),
        ('charge', 'Net Charge', "violin_hydramp_vs_ampdfm_charge.png"),
        ('hydrophobic_ratio', 'Hydrophobic Ratio', "violin_hydramp_vs_ampdfm_hydrophobic_ratio.png"),
        ('aromaticity', 'Aromaticity', "violin_hydramp_vs_ampdfm_aromaticity.png")
    ]
    
    for descriptor, ylabel, filename in descriptors_info:
        print(f"\nGenerating individual violin plot for {descriptor}...")
        
        # Create figure with HydrAMP styling
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the descriptor
        plot_descriptor_violin_subplot(ax, df, descriptor, ylabel, show_legend=True)
        
        # Add title
        ax.set_title(f"Predicted {ylabel}", fontsize=14, fontweight='normal', pad=10)
        
        # Add legend
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        out_path = Path(out_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Individual descriptor violin plot saved to {out_path}")
        
        # Always display in Jupyter
        from IPython.display import display
        display(plt.gcf())
        plt.close()
    
    print(f"\nAll individual descriptor plots completed!")

# Run when executed
if __name__ == "__main__":
    plot_all_descriptor_violins_comparison()
