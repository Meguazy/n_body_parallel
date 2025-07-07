#!/usr/bin/env python3

"""
N-Body Simulation Performance Analysis Script
Location: scripts/analyze_results.py
Run from project root directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import sys

def check_environment():
    """Check if we're in the right directory and files exist"""
    
    if not os.path.exists('results'):
        print("Error: 'results' directory not found!")
        print("This script must be run from the project root directory.")
        return False
    
    if not os.path.exists('results/performance_results.csv'):
        print("Error: results/performance_results.csv not found!")
        print("Please run experiments first:")
        print("  ./scripts/run_experiments.sh")
        return False
    
    return True

def load_and_analyze_results():
    """Load the results CSV and perform analysis"""
    
    try:
        df = pd.read_csv('results/performance_results.csv')
        print("Data loaded successfully!")
        print(f"Total records: {len(df)}")
        print(f"Configurations: {df['exec_id'].unique()}")
        print(f"Processor counts: {sorted(df['processors_number'].unique())}")
        print()
    except FileNotFoundError:
        print("Error: results/performance_results.csv not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    return df

def create_summary_statistics(df):
    """Create summary statistics for each configuration"""
    
    print("=== SUMMARY STATISTICS ===")
    
    for exec_id in sorted(df['exec_id'].unique()):
        config_data = df[df['exec_id'] == exec_id]
        print(f"\nConfiguration: {exec_id}")
        print(f"Bodies: {config_data['num_bodies'].iloc[0]}, Steps: {config_data['num_steps'].iloc[0]}")
        print(f"Max Speedup: {config_data['speed_up'].max():.3f} (at {config_data.loc[config_data['speed_up'].idxmax(), 'processors_number']} processors)")
        print(f"Max Efficiency: {config_data['efficiency'].max():.3f} (at {config_data.loc[config_data['efficiency'].idxmax(), 'processors_number']} processors)")
        
        # Check if 16 processors data exists
        max_proc_data = config_data[config_data['processors_number'] == config_data['processors_number'].max()]
        if not max_proc_data.empty:
            max_proc = config_data['processors_number'].max()
            max_proc_eff = max_proc_data['efficiency'].iloc[0]
            print(f"Efficiency at {max_proc} processors: {max_proc_eff:.3f}")

def create_visualizations(df):
    """Create comprehensive visualizations"""
    
    # Create plots directory
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    # Set style for better looking plots
    plt.style.use('default')  # Use default instead of seaborn-v0_8 for compatibility
    
    # Create a custom color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Speedup vs Processors for all configurations
    plt.figure(figsize=(12, 8))
    for i, exec_id in enumerate(sorted(df['exec_id'].unique())):
        config_data = df[df['exec_id'] == exec_id].sort_values('processors_number')
        plt.plot(config_data['processors_number'], config_data['speed_up'], 
                marker='o', linewidth=2, markersize=6, label=f'{exec_id}', 
                color=colors[i % len(colors)])
    
    # Add ideal speedup line
    processors = sorted(df['processors_number'].unique())
    plt.plot(processors, processors, 'k--', alpha=0.7, linewidth=2, label='Ideal Linear Speedup')
    
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Number of Processors\nN-Body Simulation Performance', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Efficiency vs Processors for all configurations
    plt.figure(figsize=(12, 8))
    for i, exec_id in enumerate(sorted(df['exec_id'].unique())):
        config_data = df[df['exec_id'] == exec_id].sort_values('processors_number')
        plt.plot(config_data['processors_number'], config_data['efficiency'], 
                marker='s', linewidth=2, markersize=6, label=f'{exec_id}',
                color=colors[i % len(colors)])
    
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Efficiency')
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Parallel Efficiency', fontsize=12)
    plt.title('Parallel Efficiency vs Number of Processors\nN-Body Simulation Performance', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(1.2, df['efficiency'].max() * 1.1))
    plt.tight_layout()
    plt.savefig('results/plots/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Execution Time vs Processors
    plt.figure(figsize=(12, 8))
    for i, exec_id in enumerate(sorted(df['exec_id'].unique())):
        config_data = df[df['exec_id'] == exec_id].sort_values('processors_number')
        plt.plot(config_data['processors_number'], config_data['elapsed_time'], 
                marker='^', linewidth=2, markersize=6, label=f'{exec_id}',
                color=colors[i % len(colors)])
    
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Elapsed Time (seconds)', fontsize=12)
    plt.title('Execution Time vs Number of Processors\nN-Body Simulation Performance', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/plots/execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of Efficiency
    try:
        pivot_efficiency = df.pivot(index='exec_id', columns='processors_number', values='efficiency')
        plt.figure(figsize=(14, 8))
        
        # Use matplotlib directly if seaborn is not available
        try:
            import seaborn as sns
            sns.heatmap(pivot_efficiency, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0.8, vmin=0, vmax=1.2, cbar_kws={'label': 'Parallel Efficiency'})
        except ImportError:
            # Fallback to matplotlib imshow
            im = plt.imshow(pivot_efficiency.values, cmap='RdYlGn', vmin=0, vmax=1.2, aspect='auto')
            plt.colorbar(im, label='Parallel Efficiency')
            plt.xticks(range(len(pivot_efficiency.columns)), pivot_efficiency.columns)
            plt.yticks(range(len(pivot_efficiency.index)), pivot_efficiency.index)
            plt.xlabel('Number of Processors')
            plt.ylabel('Configuration (Bodies-Steps)')
            
            # Add text annotations
            for i in range(len(pivot_efficiency.index)):
                for j in range(len(pivot_efficiency.columns)):
                    plt.text(j, i, f'{pivot_efficiency.iloc[i, j]:.3f}', 
                            ha='center', va='center', fontsize=10)
        
        plt.title('Parallel Efficiency Heatmap\nN-Body Simulation across Configurations', fontsize=14)
        plt.xlabel('Number of Processors', fontsize=12)
        plt.ylabel('Configuration (Bodies-Steps)', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/plots/efficiency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
    
    # 5. Individual configuration analysis
    n_configs = len(df['exec_id'].unique())
    rows = (n_configs + 2) // 3  # Calculate rows needed
    cols = min(3, n_configs)     # Max 3 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
    else:
        axes = axes.flatten()    
        
    for i, exec_id in enumerate(sorted(df['exec_id'].unique())):
        if i >= len(axes):
            break
            
        config_data = df[df['exec_id'] == exec_id].sort_values('processors_number')
        
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(config_data['processors_number'], config_data['speed_up'], 
                        'b-o', linewidth=2, markersize=6, label='Speedup')
        line2 = ax2.plot(config_data['processors_number'], config_data['efficiency'], 
                        'r-s', linewidth=2, markersize=6, label='Efficiency')
        
        ax1.plot(config_data['processors_number'], config_data['processors_number'], 
                'k--', alpha=0.5, label='Ideal Speedup')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Processors')
        ax1.set_ylabel('Speedup', color='b')
        ax2.set_ylabel('Efficiency', color='r')
        ax1.set_title(f'Config: {exec_id}')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    
    # Remove unused subplots
    for i in range(len(df['exec_id'].unique()), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('results/plots/individual_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_report(df):
    """Create a detailed performance report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("N-BODY SIMULATION PARALLEL PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("EXPERIMENTAL SETUP:")
    report_lines.append(f"• Processor range: {df['processors_number'].min()} to {df['processors_number'].max()}")
    report_lines.append(f"• Problem configurations: {len(df['exec_id'].unique())}")
    report_lines.append(f"• Total runs: {len(df)}")
    report_lines.append("")
    
    report_lines.append("CONFIGURATION DETAILS:")
    for exec_id in sorted(df['exec_id'].unique()):
        config_data = df[df['exec_id'] == exec_id]
        bodies = config_data['num_bodies'].iloc[0]
        steps = config_data['num_steps'].iloc[0]
        operations = bodies * bodies * steps
        report_lines.append(f"• {exec_id}: {bodies:,} bodies, {steps} steps (~{operations/1e6:.1f}M operations)")
    report_lines.append("")
    
    report_lines.append("PERFORMANCE HIGHLIGHTS:")
    
    # Best overall speedup
    best_speedup_idx = df['speed_up'].idxmax()
    best_speedup_row = df.loc[best_speedup_idx]
    report_lines.append(f"• Best speedup: {best_speedup_row['speed_up']:.3f}x "
                       f"(Config: {best_speedup_row['exec_id']}, {best_speedup_row['processors_number']} processors)")
    
    # Best efficiency
    best_efficiency_idx = df['efficiency'].idxmax()
    best_efficiency_row = df.loc[best_efficiency_idx]
    report_lines.append(f"• Best efficiency: {best_efficiency_row['efficiency']:.3f} "
                       f"(Config: {best_efficiency_row['exec_id']}, {best_efficiency_row['processors_number']} processors)")
    
    # Efficiency at maximum processors
    max_proc_data = df[df['processors_number'] == df['processors_number'].max()]
    if not max_proc_data.empty:
        avg_efficiency_max = max_proc_data['efficiency'].mean()
        report_lines.append(f"• Average efficiency at {df['processors_number'].max()} processors: {avg_efficiency_max:.3f}")
    
    report_lines.append("")
    report_lines.append("SCALABILITY ANALYSIS:")
    
    for exec_id in sorted(df['exec_id'].unique()):
        config_data = df[df['exec_id'] == exec_id].sort_values('processors_number')
        max_speedup = config_data['speed_up'].max()
        max_speedup_procs = config_data.loc[config_data['speed_up'].idxmax(), 'processors_number']
        final_efficiency = config_data['efficiency'].iloc[-1]
        
        report_lines.append(f"• {exec_id}: Peak speedup {max_speedup:.2f}x at {max_speedup_procs} procs, "
                           f"final efficiency {final_efficiency:.3f}")
    
    report_lines.append("")
    report_lines.append("FILES GENERATED:")
    report_lines.append("• results/performance_results.csv - Raw experimental data")
    report_lines.append("• results/plots/speedup_comparison.png - Speedup comparison")
    report_lines.append("• results/plots/efficiency_comparison.png - Efficiency comparison") 
    report_lines.append("• results/plots/execution_time_comparison.png - Execution time comparison")
    report_lines.append("• results/plots/efficiency_heatmap.png - Efficiency heatmap")
    report_lines.append("• results/plots/individual_configurations.png - Individual analysis")
    report_lines.append("• results/performance_report.txt - This report")
    
    # Save report
    with open('results/performance_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print summary
    print("\n".join(report_lines))

def main():
    """Main analysis function"""
    
    print("N-Body Simulation Performance Analysis")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Load data
    df = load_and_analyze_results()
    if df is None:
        sys.exit(1)
    
    # Perform analysis
    create_summary_statistics(df)
    print("\nGenerating visualizations...")
    create_visualizations(df)
    print("\nCreating performance report...")
    create_performance_report(df)
    
    print(f"\nAnalysis complete! Check the 'results/' directory for:")
    print("• performance_results.csv - Raw data")
    print("• performance_report.txt - Detailed report") 
    print("• plots/ directory - All visualizations")

if __name__ == "__main__":
    main()