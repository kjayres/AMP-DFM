#!/usr/bin/env python3
"""plot_threshold_sweep.py

Visualize hit rates across different threshold combinations for all properties.
Creates plots showing how success rates change with varying thresholds
for potency, haemolysis, and cytotoxicity across different methods.

Designed to be run directly in Jupyter notebook cells.

Usage in Jupyter cell:
    %run plot_threshold_sweep.py
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

# Default plots directory
DEFAULT_PLOTS_DIR = Path.home() / "mog_dfm/ampflow/plots"
DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_methods_data():
    """Load data from all methods for threshold sweeping."""
    data = {}
    
    # MOG targets - updated names to match other scripts
    mog_base = Path.home() / "mog_dfm/ampflow/results/mog"
    mog_targets = {
        'Unguided': mog_base / 'baseline/baseline_scores.csv',
        'E.coli': mog_base / 'ecoli_hp4_111_2500/mog_samples_scores.csv',
        'Generic': mog_base / 'generic_hp4_111_2500/mog_samples_scores.csv',
        'P.aeruginosa': mog_base / 'paeruginosa_hp4_111_2500/mog_samples_scores.csv',
        'S.aureus': mog_base / 'saureus_hp4_111_2500/mog_samples_scores.csv'
    }
    
    # Note: Removed HydrAMP vs AMP-DFM variants for focused MOG comparison
    
    # Load MOG data
    for method, path in mog_targets.items():
        if path.exists():
            df = pd.read_csv(path)
            if all(col in df.columns for col in ['potency', 'hemolysis', 'cytotox']):
                data[method] = df[['potency', 'hemolysis', 'cytotox']].copy()
                print(f"Loaded {len(data[method])} samples from {method}")
    
    # Note: HydrAMP/AMP-DFM data loading removed for focused MOG comparison
    
    return data

def calculate_hit_rates_sweep(data, thresholds):
    """Calculate hit rates for multiple threshold combinations."""
    results = {}
    
    for method, df in data.items():
        method_results = []
        
        for pot_thresh in thresholds:
            for hem_thresh in thresholds:
                for cyto_thresh in thresholds:
                    # Calculate hit rate for this threshold combination
                    hits = (
                        (df['potency'] > pot_thresh) & 
                        (df['hemolysis'] > hem_thresh) & 
                        (df['cytotox'] > cyto_thresh)
                    ).sum()
                    
                    hit_rate = hits / len(df)
                    
                    method_results.append({
                        'potency_threshold': pot_thresh,
                        'hemolysis_threshold': hem_thresh,
                        'cytotox_threshold': cyto_thresh,
                        'hit_rate': hit_rate,
                        'hit_count': hits,
                        'total': len(df)
                    })
        
        results[method] = pd.DataFrame(method_results)
    
    return results



def plot_threshold_line_plots(results, out_dir=None):
    """Plot line plots showing hit rates vs threshold for uniform thresholds across all properties."""
    if out_dir is None:
        out_dir = DEFAULT_PLOTS_DIR
    
    # Filter for cases where all three thresholds are equal
    uniform_results = {}
    for method, df in results.items():
        uniform = df[
            (df['potency_threshold'] == df['hemolysis_threshold']) &
            (df['hemolysis_threshold'] == df['cytotox_threshold'])
        ].copy()
        uniform['threshold'] = uniform['potency_threshold']
        uniform_results[method] = uniform
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Apply HydrAMP styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#B8B8B8')
    ax.spines['bottom'].set_color('#B8B8B8')
    ax.grid(True, alpha=0.3, linewidth=1.2)
    ax.set_axisbelow(True)
    
    # Define colors to match other scripts (densities_stylised.py and physicochemical_descriptors.py)
    colors = {
        'Unguided': '#FA8072',      # salmon
        'E.coli': '#77C99D',        # turquoise
        'Generic': '#1f78b4',       # dark blue
        'P.aeruginosa': '#A82223',  # cherry red
        'S.aureus': '#EDA355'       # yellowish gold
    }
    
    # Calculate average hit rates across the MOG methods (excluding Unguided)
    mog_methods = ['S.aureus', 'Generic', 'E.coli', 'P.aeruginosa']
    available_mog_methods = [method for method in mog_methods if method in uniform_results]
    
    if len(available_mog_methods) > 1:
        # Calculate average across available MOG methods
        threshold_values = uniform_results[available_mog_methods[0]]['threshold'].values
        avg_hit_rates = np.zeros(len(threshold_values))
        
        for method in available_mog_methods:
            method_hit_rates = uniform_results[method]['hit_rate'].values
            avg_hit_rates += method_hit_rates
        
        avg_hit_rates /= len(available_mog_methods)
        
        # Plot average line first (thicker, faint navy)
        ax.plot(threshold_values, avg_hit_rates * 100, 
               label='Guided Average', color='#2F4F4F', linestyle='-', linewidth=3.5,
               alpha=0.6, marker='s', markersize=5, zorder=1)
    
    # Plot lines for each method
    for method, df in uniform_results.items():
        color = colors.get(method, '#777777')
        linestyle = '-'  # All methods use solid lines
        linewidth = 2.5 if method == 'E.coli' else 2
        
        ax.plot(df['threshold'], df['hit_rate'] * 100, 
               label=method, color=color, linestyle=linestyle, linewidth=linewidth,
               marker='o', markersize=4, zorder=2)
    
    # Add dashed red line at 0.8 threshold
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='0.8 Threshold')
    
    # Add percentage annotations and markers at 0.8 threshold
    threshold_0_8_data = []
    
    # First collect data for guided average at 0.8
    if len(available_mog_methods) > 1:
        avg_hit_rate_0_8 = 0
        for method in available_mog_methods:
            method_df = uniform_results[method]
            hit_rate_0_8 = method_df[method_df['threshold'] == 0.8]['hit_rate'].iloc[0] if len(method_df[method_df['threshold'] == 0.8]) > 0 else 0
            avg_hit_rate_0_8 += hit_rate_0_8
        avg_hit_rate_0_8 /= len(available_mog_methods)
        
        # Add marker for guided average
        ax.scatter(0.8, avg_hit_rate_0_8 * 100, s=80, color='#2F4F4F', 
                  edgecolors='white', linewidth=2, zorder=3)
        
        # Add annotation for guided average
        ax.annotate(f'{avg_hit_rate_0_8 * 100:.1f}%', 
                   xy=(0.8, avg_hit_rate_0_8 * 100), 
                   xytext=(0.87, avg_hit_rate_0_8 * 100),
                   fontsize=10, ha='left', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc='#2F4F4F', ec="none", alpha=0.9),
                   color='white', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                 color='#2F4F4F', alpha=0.6, lw=1.5))
    
    # Add markers and annotations for individual methods at 0.8 threshold
    # Collect data for smart positioning
    method_data = []
    for method, df in uniform_results.items():
        color = colors.get(method, '#777777')
        
        # Find hit rate at 0.8 threshold
        threshold_0_8_row = df[df['threshold'] == 0.8]
        if len(threshold_0_8_row) > 0:
            hit_rate_0_8 = threshold_0_8_row['hit_rate'].iloc[0]
            method_data.append((method, color, hit_rate_0_8))
    
    # Sort by hit rate for better positioning
    method_data.sort(key=lambda x: x[2], reverse=True)
    
    # Define x-offsets to spread out annotations horizontally
    x_offsets = [0.82, 0.84, 0.86, 0.88, 0.90]  # Spread to the right of the 0.8 line
    
    for i, (method, color, hit_rate_0_8) in enumerate(method_data):
        # Add marker point
        ax.scatter(0.8, hit_rate_0_8 * 100, s=60, color=color, 
                  edgecolors='white', linewidth=1.5, zorder=3)
        
        # Special positioning for E.coli - move it higher with downward arrow
        if method == 'E.coli':
            x_offset = 0.84
            y_offset = hit_rate_0_8 * 100 + 15  # Position above the point
            arrow_style = '->'
            connection_style = 'arc3,rad=0.2'
            va_align = 'bottom'
        else:
            # Use different x-offset for other methods to avoid overlap
            x_offset = x_offsets[i % len(x_offsets)]
            y_offset = hit_rate_0_8 * 100
            arrow_style = '->'
            connection_style = 'arc3,rad=0'
            va_align = 'center'
        
        # Add percentage annotation with larger font and spread positioning
        ax.annotate(f'{hit_rate_0_8 * 100:.1f}%', 
                   xy=(0.8, hit_rate_0_8 * 100), 
                   xytext=(x_offset, y_offset),
                   fontsize=10, ha='left', va=va_align,
                   bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
                   color='white', fontweight='bold',
                   arrowprops=dict(arrowstyle=arrow_style, connectionstyle=connection_style, 
                                 color=color, alpha=0.6, lw=1.5))
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title('Predicted Hit Rates vs Uniform Thresholds Across All Properties', 
                fontsize=14, fontweight='normal', pad=15)
    
    # Add legend
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
    
    # Set reasonable axis limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 100)
    
    # Fix overlapping origin labels - show single "0" at corner
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['', '20', '40', '60', '80', '100'])  # Empty string for y-axis 0
    
    plt.tight_layout()
    
    # Save figure
    out_path = Path(out_dir) / "threshold_sweep_lines.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Threshold sweep line plot saved to {out_path}")
    
    # Display in Jupyter
    if _IN_JUPYTER:
        display(plt.gcf())
    else:
        plt.show()
    plt.close()



def main():
    """Main function to generate threshold sweep line plot visualization."""
    print("Loading data from all methods...")
    data = load_all_methods_data()
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Define threshold range
    thresholds = np.arange(0.1, 1.0, 0.1)
    print(f"Calculating hit rates for {len(thresholds)} threshold values...")
    
    # Calculate hit rates across threshold combinations
    results = calculate_hit_rates_sweep(data, thresholds)
    
    print("Generating threshold sweep line plot...")
    
    # Generate line plots for uniform thresholds
    plot_threshold_line_plots(results)
    
    print("Threshold sweep visualization completed!")

# Run when executed
if __name__ == "__main__":
    main()
elif _IN_JUPYTER:
    # Auto-run when imported or run in Jupyter
    try:
        main()
    except Exception as err:
        print("plot_threshold_sweep –", err)
