#!/usr/bin/env python3
"""densities_stylised.py

Stylized density plots for PepDFM objective properties in the style of HydrAMP:
- Faint grid background
- No top and right borders (spines)
- Same colors and labels as the original density_plots.py

Designed to be run directly in Jupyter notebook cells. Saves plots as 
kde_stylised_*.png to avoid overwriting original plots.

Usage in Jupyter cell:
    %run densities_stylised.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Default paths
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASELINE_CSV = Path.home() / "mog_dfm/ampflow/results/mog/baseline/baseline_scores.csv"
DEFAULT_GUIDED_CSV = Path.home() / "mog_dfm/ampflow/results/mog/generic_hp4_111_2500/mog_samples_scores.csv"

# Data paths for each sampling method (updated for _hp4_111_2500 results)
DATA_PATHS = {
    "Unguided": Path.home() / "mog_dfm/ampflow/results/mog/baseline/baseline_scores.csv",
    "Generic": Path.home() / "mog_dfm/ampflow/results/mog/generic_hp4_111_2500/mog_samples_scores.csv", 
    "E.coli": Path.home() / "mog_dfm/ampflow/results/mog/ecoli_hp4_111_2500/mog_samples_scores.csv",
    "P.aeruginosa": Path.home() / "mog_dfm/ampflow/results/mog/paeruginosa_hp4_111_2500/mog_samples_scores.csv",
    "S.aureus": Path.home() / "mog_dfm/ampflow/results/mog/saureus_hp4_111_2500/mog_samples_scores.csv"
}

# Colors for each method (same as violin_plots.py)
COLORS = {
    "Unguided": "#FA8072",      # salmon (unconditional/unguided)
    "Generic": "#1f78b4",       # dark blue (guided)
    "E.coli": "#77C99D",        # turquoise
    "P.aeruginosa": "#A82223",  # cherry red
    "S.aureus": "#EDA355"       # yellowish gold
}

def _plot_kde_stylised(df: pd.DataFrame, metric: str, label: str, out_path: Path | None = None):
    """Plot a stylized KDE overlay for *metric* across all sampling methods.
    
    The input *df* must have the columns [metric, "method"] where *method* is
    one of the sampling methods.
    """
    # Create figure with HydrAMP-style formatting
    fig, ax = plt.subplots(figsize=(8, 5))  # Even wider and taller for better readability
    
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
    
    # Define order to match violin plots
    method_order = ["Unguided", "Generic", "E.coli", "P.aeruginosa", "S.aureus"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    handled = []
    mean_positions = []  # Track mean positions for annotation spacing
    
    for method in method_order:
        if method not in df.method.unique():
            continue  # skip missing category
        subset = df[df.method == method][metric]
        color = COLORS[method]
        
        sns.kdeplot(subset, fill=True, alpha=0.4, color=color, 
                   label=method, ax=ax)
        mean_val = subset.mean()
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)
        mean_positions.append((mean_val, method, color))
        handled.append(method)

    # Annotate means with smart positioning to avoid overlap
    if mean_positions:
        ymax = ax.get_ylim()[1]
        y_levels = [0.85, 0.75, 0.65, 0.55, 0.45]  # Different heights for annotations
        
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
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    
    # Set proper axis limits - allow KDE to extend naturally but fix y-axis
    ax.set_ylim(bottom=0)
    
    # Allow natural KDE extension but ensure x-axis ticks don't go below 0
    x_min, x_max = ax.get_xlim()
    if x_min < 0:
        # Keep the natural KDE extension but adjust tick locations
        x_ticks = ax.get_xticks()
        x_ticks = x_ticks[x_ticks >= 0]  # Only show ticks >= 0
        ax.set_xticks(x_ticks)
    
    # Adjust tick parameters for cleaner look - remove tick marks
    ax.tick_params(axis='both', which='major', labelsize=9, length=0)
    
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Stylised figure saved to {out_path}")

    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def load_and_combine_all_data():
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
            # Invert hemolysis & cytotox so that higher = safer (like violin_plots)
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
    
    # Define order to match violin plots
    method_order = ["Unguided", "Generic", "E.coli", "P.aeruginosa", "S.aureus"]
    method_order = [m for m in method_order if m in df['method'].unique()]
    
    handled = []
    mean_positions = []  # Track mean positions for annotation spacing
    
    for method in method_order:
        if method not in df.method.unique():
            continue  # skip missing category
        subset = df[df.method == method][metric]
        color = COLORS[method]
        
        sns.kdeplot(subset, fill=True, alpha=0.4, color=color, 
                   label=method if show_legend else "", ax=ax)
        mean_val = subset.mean()
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)
        mean_positions.append((mean_val, method, color))
        handled.append(method)

    # Annotate means with smart positioning to avoid overlap
    if mean_positions:
        ymax = ax.get_ylim()[1]
        y_levels = [0.85, 0.75, 0.65, 0.55, 0.45]  # Different heights for annotations
        
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
    
    # Set proper axis limits - customize based on metric
    ax.set_ylim(bottom=0)
    
    # Set different x-limits based on the metric
    if "Antimicrobial Activity" in label:
        ax.set_xlim(-0.25, 1.3)  # Extended left range for antimicrobial activity
    elif "Haemolysis" in label:
        ax.set_xlim(-0.1, 1.1)
    else:  # Cytotoxicity
        ax.set_xlim(-0.1, 1.0)
    
    # Show tick labels from 0 to 1.0 (including 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Adjust tick parameters for cleaner look - remove tick marks
    ax.tick_params(axis='both', which='major', labelsize=9, length=0)

def plot_all_methods_density_overlays(out_dir: Path | str | None = None):
    """Generate stylized KDE density overlays for all 5 sampling methods with shared legend.
    
    Creates a 1x3 subplot layout with shared legend like HydrAMP.
    """
    # Load data from all methods
    print("Loading data from all sampling methods...")
    combined = load_and_combine_all_data()

    # Fall back to default directory when caller did not specify one
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR

    print(f"Saving stylised density plots to: {out_dir}")

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
    out_path = Path(out_dir) / "kde_stylised_all_properties_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Combined stylised figure saved to {out_path}")
    
    # Always display in Jupyter
    from IPython.display import display
    display(plt.gcf())
    plt.close()

def plot_stylised_density_overlays(guided_csv: Path | str | None = None, 
                                  baseline_csv: Path | str | None = None,
                                  out_dir: Path | str | None = None):
    """Generate stylized KDE density overlays for potency, hemolysis, cytotoxicity.

    Parameters
    ----------
    guided_csv : Path | str | None, optional
        CSV containing guided samples with the required columns.
        If None, uses default path.
    baseline_csv : Path | str | None, optional
        CSV with unconditional samples & scores; if None uses default path.
    out_dir : Path | str | None, optional
        Output directory for PNG copies. If None, uses default plots directory.
    """
    # Use defaults if not specified
    if guided_csv is None:
        guided_csv = DEFAULT_GUIDED_CSV
    if baseline_csv is None:
        baseline_csv = DEFAULT_BASELINE_CSV
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR

    # Load and process guided data
    guided_df = pd.read_csv(guided_csv)
    if not {"potency", "hemolysis", "cytotox"}.issubset(guided_df.columns):
        raise RuntimeError("guided CSV must contain potency / hemolysis / cytotox columns")
    guided_df = guided_df[["potency", "hemolysis", "cytotox"]].copy()
    # Invert hemolysis & cytotox so that higher = more toxic
    guided_df["hemolysis"] = 1.0 - guided_df["hemolysis"]
    guided_df["cytotox"]   = 1.0 - guided_df["cytotox"]
    guided_df["src"] = "guided"

    # Load and process baseline data if available
    if Path(baseline_csv).exists():
        uncond_df = pd.read_csv(baseline_csv)
        if not {"potency", "hemolysis", "cytotox"}.issubset(uncond_df.columns):
            raise RuntimeError("baseline CSV must contain potency / hemolysis / cytotox columns")
        uncond_df = uncond_df[["potency", "hemolysis", "cytotox"]].copy()
        # Invert toxicity-related probabilities so higher means more toxic
        uncond_df["hemolysis"] = 1.0 - uncond_df["hemolysis"]
        uncond_df["cytotox"]   = 1.0 - uncond_df["cytotox"]
        uncond_df["src"] = "uncond"
        combined = pd.concat([guided_df, uncond_df], ignore_index=True)
        print(f"Using baseline CSV: {baseline_csv}")
    else:
        combined = guided_df
        print(f"Baseline CSV not found at {baseline_csv}, plotting guided data only")

    print(f"Using guided CSV: {guided_csv}")
    print(f"Saving stylised plots to: {out_dir}")

    # Generate stylized plots
    for metric, label in [
        ("potency", "Predicted Antimicrobial Activity"),
        ("hemolysis", "Predicted Haemolysis"),
        ("cytotox", "Predicted Cytotoxicity")]:
        out_path = Path(out_dir) / f"kde_stylised_{metric}.png"
        _plot_kde_stylised(combined, metric, label, out_path)

# Run the stylized plots when executed
if __name__ == "__main__":
    plot_all_methods_density_overlays()
