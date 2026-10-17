"""Script to plot metric trends across runs."""
import matplotlib.pyplot as plt
from eval.results_store import ResultsStore
from typing import List


class ResultsPlotter:
    """Plots evaluation metrics trends over time."""
    
    def __init__(self):
        """Initialize the plotter."""
        self.store = ResultsStore()
    
    def plot_all_metrics(self, save_path: str = "eval/results/metrics_trend.png"):
        """
        Plot all metrics over time.
        
        Args:
            save_path: Path to save the plot
        """
        runs = self.store.get_all_runs()
        
        if len(runs) < 2:
            print("Need at least 2 evaluation runs to plot trends")
            return
        
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("RAG Evaluation Metrics Over Time", fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Get trend data
            trend = self.store.get_metric_trend(metric)
            
            if not trend:
                ax.text(0.5, 0.5, f"No data for {metric}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            timestamps = [t[0] for t in trend]
            values = [t[1] for t in trend]
            
            # Plot
            ax.plot(range(len(values)), values, marker='o', linewidth=2)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.set_xlabel('Run Number')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add commit hashes as labels
            for j, (ts, val) in enumerate(trend):
                run = runs[j]
                commit_short = run["commit_hash"][:7]
                ax.annotate(commit_short, (j, val), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    
    def plot_comparison(self, run_index1: int, run_index2: int, 
                       save_path: str = "eval/results/comparison.png"):
        """
        Plot comparison between two runs.
        
        Args:
            run_index1: Index of first run
            run_index2: Index of second run
            save_path: Path to save the plot
        """
        comparison = self.store.compare_runs(run_index1, run_index2)
        
        metrics = list(comparison["differences"].keys())
        values1 = [comparison["run1"]["metrics"][m] for m in metrics]
        values2 = [comparison["run2"]["metrics"][m] for m in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar([i - width/2 for i in x], values1, width, 
                      label=f'Run {run_index1} ({comparison["run1"]["commit_hash"][:7]})')
        bars2 = ax.bar([i + width/2 for i in x], values2, width,
                      label=f'Run {run_index2} ({comparison["run2"]["commit_hash"][:7]})')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.close()
    
    def print_summary(self):
        """Print a summary of all evaluation runs."""
        runs = self.store.get_all_runs()
        
        if not runs:
            print("No evaluation runs found")
            return
        
        print("\n" + "="*60)
        print("EVALUATION RUNS SUMMARY")
        print("="*60)
        
        for i, run in enumerate(runs):
            print(f"\nRun {i + 1}:")
            print(f"  Timestamp: {run['timestamp']}")
            print(f"  Commit: {run['commit_hash']}")
            print(f"  Questions: {run['num_questions']}")
            print(f"  Metrics:")
            for metric, value in run['metrics'].items():
                print(f"    {metric}: {value:.4f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    plotter = ResultsPlotter()
    plotter.print_summary()
    
    if len(plotter.store.get_all_runs()) >= 2:
        plotter.plot_all_metrics()
    else:
        print("Run more evaluations to generate trend plots")
