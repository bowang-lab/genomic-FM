#!/usr/bin/env python3
"""
Analysis script for genomic control vector sweep results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional

def load_sweep_results(sweep_dir: Path) -> List[Dict]:
    """Load all result files from a sweep directory."""
    results = []

    for json_file in sweep_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results

def analyze_sweep_results(results: List[Dict]) -> Dict:
    """Analyze sweep results and compute summary statistics."""

    if not results:
        return {}

    # Extract key metrics
    metrics_data = []
    alpha_data = []

    for result in results:
        config = result.get('config', {})

        # Extract performance metrics
        metrics_row = {
            'run_id': result.get('run_id', ''),
            'val_accuracy': config.get('val_accuracy', 0),
            'test_accuracy': config.get('test_accuracy', 0),
            'val_accuracy_improvement': config.get('val_accuracy_improvement', 0),
            'test_accuracy_improvement': config.get('test_accuracy_improvement', 0),
            'val_f1': config.get('val_f1', 0),
            'test_f1': config.get('test_f1', 0),
            'val_matthews_correlation': config.get('val_matthews_correlation', 0),
            'test_matthews_correlation': config.get('test_matthews_correlation', 0),
            'alpha_mean': config.get('alpha_mean', 0),
            'alpha_std': config.get('alpha_std', 0),
            'alpha_min': config.get('alpha_min', 0),
            'alpha_max': config.get('alpha_max', 0),
        }
        metrics_data.append(metrics_row)

        # Extract alpha values
        alpha_row = {'run_id': result.get('run_id', '')}
        alpha_values = result.get('alpha_values', [])
        for i, alpha in enumerate(alpha_values):
            alpha_row[f'alpha_{i}'] = alpha
        alpha_data.append(alpha_row)

    # Create DataFrames
    metrics_df = pd.DataFrame(metrics_data)
    alpha_df = pd.DataFrame(alpha_data)

    # Compute summary statistics
    summary = {
        'total_runs': len(results),
        'best_val_improvement': metrics_df['val_accuracy_improvement'].max(),
        'best_test_improvement': metrics_df['test_accuracy_improvement'].max(),
        'mean_val_improvement': metrics_df['val_accuracy_improvement'].mean(),
        'mean_test_improvement': metrics_df['test_accuracy_improvement'].mean(),
        'std_val_improvement': metrics_df['val_accuracy_improvement'].std(),
        'std_test_improvement': metrics_df['test_accuracy_improvement'].std(),
    }

    # Find best runs
    best_val_idx = metrics_df['val_accuracy_improvement'].idxmax()
    best_test_idx = metrics_df['test_accuracy_improvement'].idxmax()

    summary['best_val_run'] = {
        'run_id': metrics_df.loc[best_val_idx, 'run_id'],
        'val_improvement': metrics_df.loc[best_val_idx, 'val_accuracy_improvement'],
        'test_improvement': metrics_df.loc[best_val_idx, 'test_accuracy_improvement'],
        'alpha_values': alpha_df.loc[best_val_idx].drop('run_id').values.tolist()
    }

    summary['best_test_run'] = {
        'run_id': metrics_df.loc[best_test_idx, 'run_id'],
        'val_improvement': metrics_df.loc[best_test_idx, 'val_accuracy_improvement'],
        'test_improvement': metrics_df.loc[best_test_idx, 'test_accuracy_improvement'],
        'alpha_values': alpha_df.loc[best_test_idx].drop('run_id').values.tolist()
    }

    return {
        'summary': summary,
        'metrics_df': metrics_df,
        'alpha_df': alpha_df
    }

def plot_sweep_results(analysis: Dict, output_dir: Path):
    """Create visualization plots for sweep results."""

    metrics_df = analysis['metrics_df']
    alpha_df = analysis['alpha_df']

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. Performance improvement distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(metrics_df['val_accuracy_improvement'], bins=20, alpha=0.7, label='Validation')
    axes[0].hist(metrics_df['test_accuracy_improvement'], bins=20, alpha=0.7, label='Test')
    axes[0].set_xlabel('Accuracy Improvement')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Accuracy Improvements')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Scatter plot: val vs test improvement
    axes[1].scatter(metrics_df['val_accuracy_improvement'],
                   metrics_df['test_accuracy_improvement'],
                   alpha=0.6)
    axes[1].set_xlabel('Validation Accuracy Improvement')
    axes[1].set_ylabel('Test Accuracy Improvement')
    axes[1].set_title('Validation vs Test Performance')
    axes[1].grid(True, alpha=0.3)

    # Add diagonal line
    min_val = min(metrics_df['val_accuracy_improvement'].min(),
                  metrics_df['test_accuracy_improvement'].min())
    max_val = max(metrics_df['val_accuracy_improvement'].max(),
                  metrics_df['test_accuracy_improvement'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Alpha values analysis
    alpha_cols = [col for col in alpha_df.columns if col.startswith('alpha_')]
    if alpha_cols:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Alpha values heatmap for top runs
        top_runs_idx = metrics_df.nlargest(10, 'val_accuracy_improvement').index
        top_alpha_values = alpha_df.loc[top_runs_idx, alpha_cols].values

        im = axes[0].imshow(top_alpha_values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Top Runs (by Val Improvement)')
        axes[0].set_title('Alpha Values for Top 10 Runs')
        plt.colorbar(im, ax=axes[0])

        # Layer-wise alpha distribution
        alpha_means = alpha_df[alpha_cols].mean()
        alpha_stds = alpha_df[alpha_cols].std()
        layer_indices = range(len(alpha_means))

        axes[1].errorbar(layer_indices, alpha_means, yerr=alpha_stds,
                        fmt='o-', capsize=3, capthick=1)
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('Alpha Value')
        axes[1].set_title('Mean Alpha Values by Layer (with std)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'alpha_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Correlation matrix
    numeric_cols = ['val_accuracy_improvement', 'test_accuracy_improvement',
                    'val_f1', 'test_f1', 'val_matthews_correlation',
                    'test_matthews_correlation', 'alpha_mean', 'alpha_std']

    corr_matrix = metrics_df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f')
    plt.title('Correlation Matrix of Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(analysis: Dict, output_path: Path):
    """Generate a text report of the sweep results."""

    summary = analysis['summary']

    report = f"""
# Genomic Control Vector Sweep Analysis Report

## Summary Statistics
- Total runs: {summary['total_runs']}
- Best validation improvement: {summary['best_val_improvement']:.4f}
- Best test improvement: {summary['best_test_improvement']:.4f}
- Mean validation improvement: {summary['mean_val_improvement']:.4f} ± {summary['std_val_improvement']:.4f}
- Mean test improvement: {summary['mean_test_improvement']:.4f} ± {summary['std_test_improvement']:.4f}

## Best Validation Run
- Run ID: {summary['best_val_run']['run_id']}
- Validation improvement: {summary['best_val_run']['val_improvement']:.4f}
- Test improvement: {summary['best_val_run']['test_improvement']:.4f}
- Alpha values: {[f'{x:.3f}' for x in summary['best_val_run']['alpha_values']]}

## Best Test Run
- Run ID: {summary['best_test_run']['run_id']}
- Validation improvement: {summary['best_test_run']['val_improvement']:.4f}
- Test improvement: {summary['best_test_run']['test_improvement']:.4f}
- Alpha values: {[f'{x:.3f}' for x in summary['best_test_run']['alpha_values']]}

## Insights
"""

    # Add insights based on data
    if summary['best_val_improvement'] > 0.01:
        report += "- Control vectors show meaningful improvement in accuracy\n"
    else:
        report += "- Control vectors show minimal improvement in accuracy\n"

    if abs(summary['mean_val_improvement'] - summary['mean_test_improvement']) < 0.005:
        report += "- Good generalization from validation to test set\n"
    else:
        report += "- Potential overfitting or dataset bias detected\n"

    with open(output_path, 'w') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description="Analyze genomic control vector sweep results")
    parser.add_argument("sweep_dir", type=str, help="Directory containing sweep results")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                        help="Output directory for analysis results")

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)

    if not sweep_dir.exists():
        print(f"Error: Sweep directory {sweep_dir} does not exist")
        return 1

    # Load and analyze results
    print(f"Loading results from {sweep_dir}")
    results = load_sweep_results(sweep_dir)

    if not results:
        print("No results found in sweep directory")
        return 1

    print(f"Loaded {len(results)} runs")

    # Analyze results
    print("Analyzing results...")
    analysis = analyze_sweep_results(results)

    # Create visualizations
    print("Creating visualizations...")
    plot_sweep_results(analysis, output_dir)

    # Generate report
    print("Generating report...")
    generate_report(analysis, output_dir / "report.txt")

    # Save summary as JSON
    with open(output_dir / "summary.json", 'w') as f:
        # Remove DataFrames for JSON serialization
        summary_for_json = analysis['summary']
        json.dump(summary_for_json, f, indent=2)

    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Best validation improvement: {analysis['summary']['best_val_improvement']:.4f}")
    print(f"Best test improvement: {analysis['summary']['best_test_improvement']:.4f}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
