#!/usr/bin/env python3
"""densities_hydramp_vs_ampdfm.py

Stylized density plots comparing HydrAMP vs AMP-DFM models in the style of HydrAMP:
- Faint grid background
- No top and right borders (spines)
- Consistent colors across datasets
- Shows both HydrAMP self-reported and AMP-DFM potency scores
- Shows haemolysis and cytotoxicity safety distributions

Designed to be run directly in Jupyter notebook cells.

Usage in Jupyter cell:
    %run densities_hydramp_vs_ampdfm.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
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

# Colors for each model (distinct from previous plots)
COLORS = {
    "HydrAMP (Self-reported)": "#6A1B9A",     # Darker purple for HydrAMP self-reported
    "HydrAMP (AMP-DFM Scored)": "#9D4EDD",    # Lighter purple for HydrAMP scored by AMP-DFM
    "AMP-DFM Unguided": "#FF6B6B",           # Coral red for unguided
    "AMP-DFM AMP-Only": "#4ECDC4",           # Teal for AMP-only (1,0,0)
    "AMP-DFM Full-Guidance": "#45B7D1"       # Sky blue for full guidance (1,1,1)
}

def load_and_combine_comparison_data():
    """Load all comparison data and combine into single DataFrame."""
    combined_data = []
    
    for method, path in DATA_PATHS.items():
        if path.exists():
            df = pd.read_csv(path)
            
            if method == "HydrAMP":
                # HydrAMP has both 'amp' (self-reported) and 'potency' (AMP-DFM judge)
                required_cols = ["amp", "potency", "hemolysis", "cytotox"]
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Missing required columns in {path}")
                    continue
                
                # For potency: create separate entries for self-reported vs AMP-DFM scored
                # For safety: only one entry since all safety scores come from AMP-DFM
                
                # HydrAMP self-reported potency (for potency plot only)
                hydramp_self_data = pd.DataFrame({
                    'potency': df['amp'],                    # HydrAMP's self-reported score
                    'hemolysis': None,                       # No self-reported safety scores
                    'cytotox': None,                        # No self-reported safety scores
                    'method': "HydrAMP (Self-reported)"
                })
                
                # HydrAMP sequences scored by AMP-DFM (for all plots)
                hydramp_scored_data = pd.DataFrame({
                    'potency': df['potency'],                # AMP-DFM judge score
                    'hemolysis': 1.0 - df['hemolysis'],      # Invert: 1 = toxic, 0 = safe
                    'cytotox': 1.0 - df['cytotox'],         # Invert: 1 = toxic, 0 = safe
                    'method': "HydrAMP (AMP-DFM Scored)"
                })
                
                combined_data.extend([hydramp_self_data, hydramp_scored_data])
                print(f"Loaded {len(df)} samples from {method} (self-reported potency + AMP-DFM scored all)")
            else:
                # AMP-DFM models only have AMP-DFM scores
                required_cols = ["potency", "hemolysis", "cytotox"]
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Missing required columns in {path}")
                    continue
                    
                method_data = pd.DataFrame({
                    'potency': df['potency'],               # AMP-DFM potency score
                    'hemolysis': 1.0 - df['hemolysis'],     # Invert: 1 = toxic, 0 = safe
                    'cytotox': 1.0 - df['cytotox'],        # Invert: 1 = toxic, 0 = safe
                    'method': method
                })
                
                combined_data.append(method_data)
                print(f"Loaded {len(df)} samples from {method}")
        else:
            print(f"Warning: File not found - {path}")
    
    if not combined_data:
        raise FileNotFoundError("No valid data files found")
    
    return pd.concat(combined_data, ignore_index=True)

def _plot_kde_subplot(ax, df: pd.DataFrame, metric: str, label: str, show_legend: bool = False):
    """Plot a stylized KDE overlay on a given subplot axis."""
    
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
    
    # Define method order
    method_order = ["HydrAMP (Self-reported)", "HydrAMP (AMP-DFM Scored)", "AMP-DFM Unguided", "AMP-DFM AMP-Only", "AMP-DFM Full-Guidance"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    handled = []
    mean_positions = []  # Track mean positions for annotation spacing
    
    for method in method_order:
        if method not in df.method.unique():
            continue  # skip missing category
        
        # Filter out None values for metrics that don't apply to all methods
        subset_data = df[df.method == method][metric]
        subset_data = subset_data.dropna()
        
        if len(subset_data) == 0:
            continue
            
        color = COLORS[method]
        
        sns.kdeplot(subset_data, fill=True, alpha=0.4, color=color, 
                   label=method if show_legend else "", ax=ax)
        mean_val = subset_data.mean()
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)
        mean_positions.append((mean_val, method, color))
        handled.append(method)

    # Annotate means with smart positioning to avoid overlap
    if mean_positions:
        ymax = ax.get_ylim()[1]
        y_levels = [0.85, 0.75, 0.65, 0.55]  # Different heights for annotations
        
        # Sort by mean value to assign heights
        mean_positions.sort(key=lambda x: x[0])
        
        for i, (mean_val, method, color) in enumerate(mean_positions):
            ypos = ymax * y_levels[i % len(y_levels)]
            text = f"{method}\nMean: {mean_val:.3f}"
            ax.text(mean_val, ypos, text,
                    color="white", fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.8))

    if not handled:
        raise ValueError("No data available for plotting – check input CSVs.")

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    # Set proper axis limits
    ax.set_ylim(bottom=0)
    
    # Set x-limits based on the metric
    if "HydrAMP" in label:
        ax.set_xlim(-0.1, 1.1)
    elif "AMP-DFM" in label:
        ax.set_xlim(-0.25, 1.3)  # Extended for AMP-DFM potency like original
    else:  # Safety metrics
        ax.set_xlim(-0.1, 1.1)
    
    # Show tick labels from 0 to 1.0
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Adjust tick parameters for cleaner look - remove tick marks
    ax.tick_params(axis='both', which='major', labelsize=9, length=0)

def plot_all_methods_density_overlays(out_dir: Path | str | None = None):
    """Generate stylized KDE density overlays for all models with shared legend.
    
    Creates a 1x3 subplot layout with potency, haemolysis toxicity, and cytotoxicity.
    """
    # Load data from all methods
    print("Loading data from all models...")
    combined = load_and_combine_comparison_data()

    # Fall back to default directory when caller did not specify one
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR

    print(f"Saving HydrAMP vs AMP-DFM comparison plots to: {out_dir}")

    # Create figure with 3 subplots and extra space for bottom legend
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    metrics_labels = [
        ("potency", "Predicted Antimicrobial Activity"),
        ("hemolysis", "Predicted Haemolysis"),
        ("cytotox", "Predicted Cytotoxicity")
    ]
    
    # Plot each metric in its own subplot
    for i, (metric, label) in enumerate(metrics_labels):
        print(f"Generating density plot for {metric}...")
        _plot_kde_subplot(axes[i], combined, metric, label, show_legend=(i == 0))
        # Add clean subplot title - larger font, no bold
        axes[i].set_title(label, fontsize=14, fontweight='normal', pad=10)
    
    # Remove individual legends (only if they exist)
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for bottom legend
    
    # Create shared legend at bottom with HydrAMP-style border
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                       frameon=True, fontsize=10, ncol=5, 
                       borderpad=0.8, columnspacing=1.5)
    
    # Style the legend frame to match plot borders
    legend.get_frame().set_edgecolor('#B8B8B8')  # Same grey as plot borders
    legend.get_frame().set_linewidth(1.2)        # Same thickness as plot borders
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    # Save combined figure
    out_path = Path(out_dir) / "kde_hydramp_vs_ampdfm_all_properties_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

# Run when executed
if __name__ == "__main__":
    plot_all_methods_density_overlays()
