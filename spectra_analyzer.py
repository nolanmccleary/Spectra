import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
import argparse
import sys


# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


# Set figure parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


@dataclass
class AttackConfig:
    key: str
    label: str
    color: str


@dataclass
class MetricConfig:
    key: str
    label: str
    unit: str = ""
    log_scale: bool = False
    bins: int = 30


class AttackAnalyzer:
    
    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = None
        self.df = None
        
        # Default attack configurations
        self.attack_configs = [
            AttackConfig('ahash_attack', 'AHash', '#e74c3c'),
            AttackConfig('dhash_attack', 'DHash', '#2ecc71'),
            AttackConfig('phash_attack', 'PHash', '#3498db'),
            AttackConfig('pdq_attack', 'PDQ', '#f39c12'),
        ]
        
        # Pre-validation metrics
        self.pre_metrics = [
            MetricConfig('ideal_beta', 'Ideal β', ''),
            MetricConfig('ideal_scale_factor', 'Scale Factor', ''),
            MetricConfig('num_steps', 'Steps to Convergence', 'steps'),
            MetricConfig('lpips', 'LPIPS Distance', ''),
            MetricConfig('l2', 'L2 Distance', ''),
            MetricConfig('hamming_distance', 'Hamming Distance', 'bits'),
        ]
        
        # Post-validation metrics
        self.post_metrics = [
            MetricConfig('lpips', 'Post-Attack LPIPS', ''),
            MetricConfig('l2', 'Post-Attack L2', ''),
            MetricConfig('ahash_hamming', 'AHash Hamming', 'bits'),
            MetricConfig('dhash_hamming', 'DHash Hamming', 'bits'),
            MetricConfig('phash_hamming', 'PHash Hamming', 'bits'),
            MetricConfig('pdq_hamming', 'PDQ Hamming', 'bits'),
        ]
    
    def load_data(self) -> bool:
        """Load and parse JSON data"""
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            self._create_dataframe()
            return True
        except FileNotFoundError:
            print(f"Error: File {self.json_file} not found")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.json_file}")
            return False
    
    def _create_dataframe(self):
        """Convert JSON data to structured DataFrame"""
        rows = []
        
        for attack_name, attack_data in self.data.items():
            per_image_results = attack_data.get("per_image_results", {})
            
            for image_name, result in per_image_results.items():
                pre_val = result.get("pre_validation", {})
                post_val = result.get("post_validation", {})
                
                if pre_val.get("success") == True:
                    row = {
                        'attack_type': attack_name,
                        'image_name': image_name,
                        'success': True
                    }
                    
                    # Add pre-validation metrics
                    for metric in self.pre_metrics:
                        value = pre_val.get(metric.key, np.nan)
                        if value != "N/A" and value is not None:
                            try:
                                row[f'pre_{metric.key}'] = float(value)
                            except (ValueError, TypeError):
                                row[f'pre_{metric.key}'] = np.nan
                        else:
                            row[f'pre_{metric.key}'] = np.nan
                    
                    # Add post-validation metrics
                    for metric in self.post_metrics:
                        value = post_val.get(metric.key, np.nan)
                        if value != "N/A" and value is not None:
                            try:
                                row[f'post_{metric.key}'] = float(value)
                            except (ValueError, TypeError):
                                row[f'post_{metric.key}'] = np.nan
                        else:
                            row[f'post_{metric.key}'] = np.nan
                    
                    rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"Loaded {len(self.df)} successful attack results")
    
    def get_present_attacks(self) -> List[AttackConfig]:
        """Get list of attacks present in the data"""
        if self.df is None:
            return []
        
        present_attack_keys = set(self.df['attack_type'].unique())
        return [config for config in self.attack_configs if config.key in present_attack_keys]
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics for all metrics"""
        if self.df is None:
            return pd.DataFrame()
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col.startswith(('pre_', 'post_'))]
        
        # Calculate statistics by attack type
        stats = []
        for attack_type in self.df['attack_type'].unique():
            attack_data = self.df[self.df['attack_type'] == attack_type]
            
            for col in numeric_cols:
                if not attack_data[col].isna().all():
                    stats.append({
                        'attack_type': attack_type,
                        'metric': col,
                        'count': attack_data[col].count(),
                        'mean': attack_data[col].mean(),
                        'std': attack_data[col].std(),
                        'min': attack_data[col].min(),
                        'q25': attack_data[col].quantile(0.25),
                        'median': attack_data[col].median(),
                        'q75': attack_data[col].quantile(0.75),
                        'max': attack_data[col].max(),
                    })
        
        return pd.DataFrame(stats)

class AttackVisualizer(AttackAnalyzer):
    """Extended analyzer with visualization capabilities"""
    
    def plot_metric_distributions(self, metrics: List[MetricConfig], prefix: str = 'pre_', 
                                 figsize: Tuple[int, int] = None, save_path: str = None):
        """Plot distribution histograms for specified metrics"""
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        present_attacks = self.get_present_attacks()
        if not present_attacks:
            print("No attacks found in data.")
            return
        
        n_metrics = len(metrics)
        n_attacks = len(present_attacks)
        
        if figsize is None:
            figsize = (4 * n_attacks, 4 * n_metrics)
        
        fig, axes = plt.subplots(n_metrics, n_attacks, figsize=figsize, squeeze=False)
        
        for row, metric in enumerate(metrics):
            col_name = f"{prefix}{metric.key}"
            
            if col_name not in self.df.columns:
                continue
            
            for col, attack_config in enumerate(present_attacks):
                ax = axes[row, col]
                
                # Get data for this attack and metric
                attack_data = self.df[self.df['attack_type'] == attack_config.key]
                data = attack_data[col_name].dropna()
                
                if len(data) > 0:
                    # Create histogram
                    ax.hist(data, bins=metric.bins, color=attack_config.color, 
                           alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Add statistics annotation
                    stats_text = (f"μ={data.mean():.3f}\n"
                                f"σ={data.std():.3f}\n"
                                f"med={data.median():.3f}\n"
                                f"n={len(data)}")
                    
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                           fontsize=8, ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Set labels and title
                if row == 0:
                    ax.set_title(f"{attack_config.label}", fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f"{metric.label}\n{metric.unit}".strip())
                if row == n_metrics - 1:
                    ax.set_xlabel(f"{metric.label} {metric.unit}".strip())
                
                # Set log scale if specified
                if metric.log_scale:
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plt.suptitle(f"{prefix.replace('_', ' ').title()}Validation Metrics Distribution", 
                     fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, output_dir: str = "analysis_output"):
        """Generate a complete analysis report with all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating comprehensive attack analysis report...")
        
        # 1. Hyperparameter analysis
        print("1. Analyzing hyperparameters...")
        hyperparameter_metrics = [
            MetricConfig('ideal_beta', 'Ideal β', ''),
            MetricConfig('ideal_scale_factor', 'Scale Factor', ''),
        ]
        self.plot_metric_distributions(
            hyperparameter_metrics, 
            prefix='pre_', 
            figsize=(12, 8),
            save_path=output_path / "hyperparameters.png"
        )
        
        # 2. Convergence analysis
        print("2. Analyzing convergence metrics...")
        convergence_metrics = [
            MetricConfig('num_steps', 'Steps to Convergence', 'steps'),
            MetricConfig('lpips', 'LPIPS Distance', ''),
            MetricConfig('l2', 'L2 Distance', ''),
        ]
        self.plot_metric_distributions(
            convergence_metrics, 
            prefix='pre_', 
            figsize=(12, 12),
            save_path=output_path / "convergence.png"
        )
        
        # 3. Post-attack quality
        print("3. Analyzing post-attack quality...")
        quality_metrics = [
            MetricConfig('lpips', 'Post-Attack LPIPS', ''),
            MetricConfig('l2', 'Post-Attack L2', ''),
        ]
        self.plot_metric_distributions(
            quality_metrics, 
            prefix='post_', 
            figsize=(8, 8),
            save_path=output_path / "quality.png"
        )
        
        # 4. Hash robustness
        print("4. Analyzing hash function robustness...")
        hash_metrics = [
            MetricConfig('ahash_hamming', 'AHash Hamming', 'bits'),
            MetricConfig('dhash_hamming', 'DHash Hamming', 'bits'),
            MetricConfig('phash_hamming', 'PHash Hamming', 'bits'),
            MetricConfig('pdq_hamming', 'PDQ Hamming', 'bits'),
        ]
        self.plot_metric_distributions(
            hash_metrics, 
            prefix='post_', 
            figsize=(16, 16),
            save_path=output_path / "hash_robustness.png"
        )
        
        # 5. Generate summary statistics
        print("5. Generating summary statistics...")
        stats_df = self.get_summary_stats()
        stats_df.to_csv(output_path / "summary_statistics.csv", index=False)
        
        # 6. Save processed data
        print("6. Saving processed data...")
        if self.df is not None:
            self.df.to_csv(output_path / "processed_data.csv", index=False)
        
        print(f"\nReport generated successfully in {output_path}")
        print(f"Files created:")
        print(f"  - hyperparameters.png")
        print(f"  - convergence.png") 
        print(f"  - quality.png")
        print(f"  - hash_robustness.png")
        print(f"  - summary_statistics.csv")
        print(f"  - processed_data.csv")






def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Analyze attack results")
    parser.add_argument("json_file", type=str, help="JSON file containing attack results")
    parser.add_argument("--output", "-o", type=str, default="analysis_output",
                       help="Output directory for analysis results (default: analysis_output)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Show interactive plots (default: save to files)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AttackVisualizer(args.json_file)
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    print(f"\nData loaded successfully!")
    print(f"Available attacks: {[config.label for config in analyzer.get_present_attacks()]}")
    print(f"Total successful attacks: {len(analyzer.df)}")
    
    if args.interactive:
        print("\nRunning interactive analysis...")
        # Run a subset of analysis interactively
        hyperparameter_metrics = [
            MetricConfig('ideal_beta', 'Ideal β', ''),
            MetricConfig('ideal_scale_factor', 'Scale Factor', ''),
        ]
        analyzer.plot_metric_distributions(hyperparameter_metrics, prefix='pre_')
    else:
        # Generate full report
        analyzer.generate_full_report(args.output)




if __name__ == "__main__":
    main()