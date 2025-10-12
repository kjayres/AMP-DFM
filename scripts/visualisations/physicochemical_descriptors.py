#!/usr/bin/env python3
"""physicochemical_descriptors.py

Compute physicochemical descriptors using modlAMP and create HydrAMP-style 
violin plots showing distributions across different sampling methods.

Computes the same four descriptors as HydrAMP:
- Isoelectric point (pI)
- Charge (at pH 7.0)
- Hydrophobic ratio
- Aromaticity

Usage in Jupyter cell:
    %run physicochemical_descriptors.py
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

# Data paths for each sampling method (updated for _hp4_111_2500 results)
DATA_PATHS = {
    "Unguided": Path.home() / "mog_dfm/ampflow/results/mog/baseline/baseline_scores.csv",
    "Generic": Path.home() / "mog_dfm/ampflow/results/mog/generic_hp4_111_2500/mog_samples_scores.csv", 
    "E.coli": Path.home() / "mog_dfm/ampflow/results/mog/ecoli_hp4_111_2500/mog_samples_scores.csv",
    "P.aeruginosa": Path.home() / "mog_dfm/ampflow/results/mog/paeruginosa_hp4_111_2500/mog_samples_scores.csv",
    "S.aureus": Path.home() / "mog_dfm/ampflow/results/mog/saureus_hp4_111_2500/mog_samples_scores.csv"
}

# Colors for each method (same as density plots)
COLORS = {
    "Unguided": "#FA8072",      # salmon
    "Generic": "#1f78b4",       # dark blue
    "E.coli": "#77C99D",        # turquoise
    "P.aeruginosa": "#A82223",  # cherry red
    "S.aureus": "#EDA355"       # yellowish gold
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

def load_and_compute_descriptors():
    """Load sequences from all sampling methods and compute descriptors."""
    combined_data = []
    
    for method, path in DATA_PATHS.items():
        if path.exists():
            df = pd.read_csv(path)
            if "sequence" not in df.columns:
                print(f"Warning: 'sequence' column not found in {path}")
                continue
            
            sequences = df['sequence'].tolist()
            print(f"Computing descriptors for {len(sequences)} sequences from {method}...")
            
            # Compute descriptors
            descriptors_df = compute_descriptors(sequences)
            descriptors_df['method'] = method
            
            combined_data.append(descriptors_df)
            print(f"Completed {method}")
        else:
            print(f"Warning: File not found - {path}")
    
    if not combined_data:
        raise FileNotFoundError("No valid data files found")
    
    return pd.concat(combined_data, ignore_index=True)

def plot_descriptor_violin(descriptor_name: str, df: pd.DataFrame, out_dir: Path | str | None = None):
    """Create HydrAMP-style violin plot for a physicochemical descriptor.
    
    Parameters
    ----------
    descriptor_name : str
        Name of the descriptor column to plot
    df : pd.DataFrame
        DataFrame containing descriptor data and method labels
    out_dir : Path | str | None, optional
        Output directory for PNG files
    """
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
    
    # Define order to match other plots
    method_order = ["Unguided", "Generic", "E.coli", "P.aeruginosa", "S.aureus"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    # Create color palette in the correct order
    palette = [COLORS[method] for method in method_order]
    
    # Create violin plot
    violin_parts = ax.violinplot(
        [df[df['method'] == method][descriptor_name].values for method in method_order],
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
        method_data = df[df['method'] == method][descriptor_name].values
        
        # Calculate quartiles
        q25 = np.percentile(method_data, 25)
        q50 = np.percentile(method_data, 50)  # median
        q75 = np.percentile(method_data, 75)
        
        # Add vertical black line for interquartile range
        ax.plot([i, i], [q25, q75], color='black', linewidth=2, zorder=5)
        
        # Add median as white circle with black border
        ax.scatter(i, q50, s=80, color='white', 
                  edgecolors='black', linewidth=2, zorder=10)
    
    # Customize plot based on descriptor
    ax.set_xlim(-0.5, len(method_order) - 0.5)
    
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=11)
    
    # Set appropriate labels and filename
    if descriptor_name == 'isoelectric_point':
        ylabel = 'Isoelectric Point'
        filename = "violin_isoelectric_point.png"
    elif descriptor_name == 'charge':
        ylabel = 'Net Charge'
        filename = "violin_charge.png"
    elif descriptor_name == 'hydrophobic_ratio':
        ylabel = 'Hydrophobic Ratio'
        filename = "violin_hydrophobic_ratio.png"
    elif descriptor_name == 'aromaticity':
        ylabel = 'Aromaticity'
        filename = "violin_aromaticity.png"
    else:
        ylabel = descriptor_name.replace('_', ' ').title()
        filename = f"violin_{descriptor_name}.png"
    
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Make x-axis labels vertical/perpendicular
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    
    plt.tight_layout()
    
    # Save figure
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Descriptor violin plot saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

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
    
    # Define order to match other plots
    method_order = ["Unguided", "Generic", "E.coli", "P.aeruginosa", "S.aureus"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    # Create color palette in the correct order
    palette = [COLORS[method] for method in method_order]
    
    # Create violin plot
    violin_parts = ax.violinplot(
        [df[df['method'] == method][descriptor_name].values for method in method_order],
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
        method_data = df[df['method'] == method][descriptor_name].values
        
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
    ax.set_xlim(-0.5, len(method_order) - 0.5)
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Make x-axis labels vertical/perpendicular
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    
    # Add legend labels only to first subplot
    if show_legend:
        for i, method in enumerate(method_order):
            # Create invisible artists for legend
            ax.scatter([], [], color=palette[i], s=100, alpha=0.7, label=method)

def plot_all_descriptor_violins_combined(out_dir: Path | str | None = None):
    """Generate combined violin plots for all four physicochemical descriptors with shared legend."""
    print("Loading sequences and computing descriptors...")
    df = load_and_compute_descriptors()
    
    # Fall back to default directory
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR

    print(f"Saving combined descriptor violin plots to: {out_dir}")

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
    plt.subplots_adjust(bottom=0.35)  # Make room for bottom legend and vertical labels
    
    # Create shared legend at bottom with HydrAMP-style border
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                       frameon=True, fontsize=10, ncol=5, 
                       borderpad=0.8, columnspacing=1.5)
    
    # Style the legend frame to match plot borders
    legend.get_frame().set_edgecolor('#B8B8B8')  # Same grey as plot borders
    legend.get_frame().set_linewidth(1.2)        # Same thickness as plot borders
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    # Save combined figure
    out_path = Path(out_dir) / "violin_physicochemical_descriptors_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Combined descriptor violin plot saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def plot_all_descriptor_violins(out_dir: Path | str | None = None):
    """Generate violin plots for all four physicochemical descriptors."""
    print("Loading sequences and computing descriptors...")
    df = load_and_compute_descriptors()
    
    descriptors = ['isoelectric_point', 'charge', 'hydrophobic_ratio', 'aromaticity']
    
    for descriptor in descriptors:
        print(f"\nGenerating violin plot for {descriptor}...")
        plot_descriptor_violin(descriptor, df, out_dir)
    
    print(f"\nAll descriptor plots completed!")

# Run when executed
if __name__ == "__main__":
    plot_all_descriptor_violins_combined()
