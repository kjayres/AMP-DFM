#!/usr/bin/env python3
"""plot_ablation_analysis.py

Visualize the ablation study results showing the effect of importance vector 
guidance on generated peptide properties. Shows hit rates ≥0.8 for:
- Antimicrobial activity: Hit rate ≥ 0.8
- Haemolysis safety: Safety rate ≥ 0.8 (non-haemolytic)  
- Cytotoxicity safety: Safety rate ≥ 0.8 (non-cytotoxic)

Displays results in narrative order to show progression from no guidance
to full multi-objective optimization.

Usage in Jupyter cell:
    %run plot_ablation_analysis.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import for display in Jupyter
try:
    from IPython.display import display
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False

# Default plots directory
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Ablation data in narrative order - all as hit rates >0.8
ABLATION_DATA = {
    "000\n(No guidance)": {
        "amp_hit": 0.9,        # AMP hit rate ≥ 0.8
        "haem_safety": 47.5,   # Haemolysis safety rate ≥ 0.8 (100 - 52.5)
        "cyto_safety": 5.8     # Cytotoxicity safety rate ≥ 0.8 (100 - 94.2)
    },
    "100\n(AMP only)": {
        "amp_hit": 88.6,
        "haem_safety": 20.7,   # 100 - 79.3
        "cyto_safety": 1.5     # 100 - 98.5
    },
    "110\n(AMP + Haemo)": {
        "amp_hit": 71.2,
        "haem_safety": 98.4,   # 100 - 1.6
        "cyto_safety": 4.6     # 100 - 95.4
    },
    "101\n(AMP + Cytotox)": {
        "amp_hit": 86.2,
        "haem_safety": 24.4,   # 100 - 75.6
        "cyto_safety": 92.6    # 100 - 7.4
    },
    "010\n(Haemo only)": {
        "amp_hit": 4.3,
        "haem_safety": 99.8,   # 100 - 0.2
        "cyto_safety": 10.7    # 100 - 89.3
    },
    "001\n(Cytotox only)": {
        "amp_hit": 2.2,
        "haem_safety": 23.1,   # 100 - 76.9
        "cyto_safety": 98.0    # 100 - 2.0
    },
    "011\n(Haemo + Cytotox)": {
        "amp_hit": 3.7,
        "haem_safety": 98.7,   # 100 - 1.3
        "cyto_safety": 74.1    # 100 - 25.9
    },
    "111\n(All properties)": {
        "amp_hit": 66.1,
        "haem_safety": 98.8,   # 100 - 1.2
        "cyto_safety": 59.9    # 100 - 40.1
    }
}

# Colors matching the qualitative palette provided
COLORS = {
    "amp_hit": "#082a54",      # Dark Blue for AMP hits
    "haem_safety": "#59a89c",  # Teal for haemolysis safety  
    "cyto_safety": "#a559aa"   # Purple for cytotoxicity safety
}

# Metric labels and properties
METRICS = {
    "amp_hit": {
        "label": "Antimicrobial Activity Hit Rate (≥0.8)",
        "color": COLORS["amp_hit"],
        "positive": True  # Higher is better
    },
    "haem_safety": {
        "label": "Haemolysis Safety Rate (≥0.8)",
        "color": COLORS["haem_safety"], 
        "positive": True  # Higher is better (safety rate)
    },
    "cyto_safety": {
        "label": "Cytotoxicity Safety Rate (≥0.8)",
        "color": COLORS["cyto_safety"],
        "positive": True  # Higher is better (safety rate)
    }
}

def plot_ablation_comparison(out_dir: Path | str | None = None):
    """Generate grouped bar chart comparing ablation study results."""
    
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    print("Creating ablation study visualization...")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Apply HydrAMP styling (same as holopolymer analysis)
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
    conditions = list(ABLATION_DATA.keys())
    metrics = list(METRICS.keys())
    
    # Set up bar positions
    bar_width = 0.25
    x_pos = np.arange(len(conditions))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [ABLATION_DATA[condition][metric] for condition in conditions]
        color = METRICS[metric]["color"]
        label = METRICS[metric]["label"]
        
        bars = ax.bar(x_pos + i * bar_width, values, bar_width,
                     label=label, color=color, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            if value > 2.0:  # Only label if value is meaningful (>2%)
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Importance Vector Configuration', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Ablation Study: Hit Rates for Multi-Objective Guidance (≥0.8)', fontsize=14, fontweight='normal', pad=10)
    
    # Set x-axis
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(conditions, fontsize=9)
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Set y-axis to start from 0 and extend to 105% for label space
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for bottom legend
    
    # Create shared legend at bottom with HydrAMP-style border
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                       frameon=True, fontsize=10, ncol=3, 
                       borderpad=0.8, columnspacing=1.5)
    
    # Style the legend frame to match plot borders
    legend.get_frame().set_edgecolor('#B8B8B8')  # Same grey as plot borders
    legend.get_frame().set_linewidth(1.2)        # Same thickness as plot borders
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    
    # Save figure
    out_path = Path(out_dir) / "ablation_study_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Ablation study plot saved to {out_path}")
    
    # Display in Jupyter
    if _IN_JUPYTER:
        display(plt.gcf())
    else:
        plt.show()
    plt.close()
    
    # Print summary statistics
    print("\nSummary:")
    print("Condition\t\tAMP Hit%\tHaem Safe%\tCyto Safe%")
    print("-" * 60)
    for condition in conditions:
        data = ABLATION_DATA[condition]
        condition_clean = condition.split('\n')[0]  # Remove line break for summary
        print(f"{condition_clean:15}\t{data['amp_hit']:8.1f}\t{data['haem_safety']:9.1f}\t{data['cyto_safety']:9.1f}")

def main():
    """Main function to run the ablation analysis."""
    plot_ablation_comparison()

# Auto-run when executed
if __name__ == "__main__":
    main()
elif _IN_JUPYTER:
    # Auto-run when imported or run in Jupyter
    try:
        main()
    except Exception as err:
        print("plot_ablation_analysis –", err)
