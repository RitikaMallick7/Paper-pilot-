"""Storage for evaluation results over time."""
from typing import List, Dict
import json
import os
from datetime import datetime


class ResultsStore:
    """Stores and retrieves evaluation results over time."""
    
    def __init__(self, results_dir: str = "eval/results"):
        """
        Initialize the results store.
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, "evaluation_results.jsonl")
        os.makedirs(results_dir, exist_ok=True)
    
    def save_run(
        self,
        metrics: Dict,
        commit_hash: str,
        num_questions: int
    ) -> Dict:
        """
        Save an evaluation run result.
        
        Args:
            metrics: Dictionary of Ragas metrics
            commit_hash: Git commit hash for this run
            num_questions: Number of questions in evaluation
            
        Returns:
            The saved result record
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "commit_hash": commit_hash,
            "num_questions": num_questions,
            "metrics": metrics
        }
        
        # Append to JSONL file
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        return result
    
    def get_all_runs(self) -> List[Dict]:
        """
        Get all evaluation runs.
        
        Returns:
            List of all result records
        """
        if not os.path.exists(self.results_file):
            return []
        
        runs = []
        with open(self.results_file, 'r') as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        
        return runs
    
    def get_latest_run(self) -> Dict:
        """
        Get the most recent evaluation run.
        
        Returns:
            The latest result record
        """
        runs = self.get_all_runs()
        if not runs:
            return {}
        return runs[-1]
    
    def get_runs_by_commit(self, commit_hash: str) -> List[Dict]:
        """
        Get all runs for a specific commit.
        
        Args:
            commit_hash: Git commit hash
            
        Returns:
            List of result records for that commit
        """
        runs = self.get_all_runs()
        return [r for r in runs if r.get("commit_hash") == commit_hash]
    
    def get_metric_trend(self, metric_name: str) -> List[tuple]:
        """
        Get trend data for a specific metric over time.
        
        Args:
            metric_name: Name of the metric (e.g., "faithfulness")
            
        Returns:
            List of (timestamp, value) tuples
        """
        runs = self.get_all_runs()
        trend = []
        
        for run in runs:
            timestamp = run.get("timestamp")
            value = run.get("metrics", {}).get(metric_name)
            if value is not None:
                trend.append((timestamp, value))
        
        return trend
    
    def compare_runs(self, run_index1: int, run_index2: int) -> Dict:
        """
        Compare two evaluation runs.
        
        Args:
            run_index1: Index of first run (0 = oldest)
            run_index2: Index of second run
            
        Returns:
            Dictionary with comparison data
        """
        runs = self.get_all_runs()
        
        if run_index1 >= len(runs) or run_index2 >= len(runs):
            raise ValueError("Invalid run index")
        
        run1 = runs[run_index1]
        run2 = runs[run_index2]
        
        comparison = {
            "run1": {
                "timestamp": run1["timestamp"],
                "commit_hash": run1["commit_hash"],
                "metrics": run1["metrics"]
            },
            "run2": {
                "timestamp": run2["timestamp"],
                "commit_hash": run2["commit_hash"],
                "metrics": run2["metrics"]
            },
            "differences": {}
        }
        
        # Calculate differences
        for metric in run1["metrics"]:
            diff = run2["metrics"][metric] - run1["metrics"][metric]
            comparison["differences"][metric] = diff
        
        return comparison
