"""Ragas evaluation script."""
from typing import List, Dict
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from agent.graph import PaperPilotAgent
from eval.gold_qa import GoldQASet, GoldQA
from eval.results_store import ResultsStore
import sys
import os
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RagasEvaluator:
    """Evaluates the RAG system using Ragas metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.agent = PaperPilotAgent()
        self.gold_qa_set = GoldQASet()
        self.results_store = ResultsStore()
    
    def run_evaluation(self) -> Dict:
        """
        Run full evaluation against gold Q&A set.
        
        Returns:
            Dictionary with Ragas metrics
        """
        gold_pairs = self.gold_qa_set.get_all()
        
        # Prepare dataset for Ragas
        evaluation_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        print(f"Evaluating against {len(gold_pairs)} gold Q&A pairs...")
        
        for qa in gold_pairs:
            print(f"Processing: {qa.question[:50]}...")
            
            # Run the agent
            result = self.agent.query(qa.question)
            
            evaluation_data["question"].append(qa.question)
            evaluation_data["answer"].append(result["answer"])
            evaluation_data["contexts"].append([c["text"] for c in result["citations"]])
            evaluation_data["ground_truth"].append(qa.answer)
        
        # Create HuggingFace dataset
        dataset = Dataset.from_dict(evaluation_data)
        
        # Run Ragas evaluation
        print("Running Ragas evaluation...")
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        # Convert to dictionary
        metrics = {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"]
        }
        
        # Get git commit hash
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            commit_hash = "unknown"
        
        # Store results
        self.results_store.save_run(
            metrics=metrics,
            commit_hash=commit_hash,
            num_questions=len(gold_pairs)
        )
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print evaluation results in a readable format."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Faithfulness: {metrics['faithfulness']:.4f}")
        print(f"Answer Relevancy: {metrics['answer_relevancy']:.4f}")
        print(f"Context Precision: {metrics['context_precision']:.4f}")
        print(f"Context Recall: {metrics['context_recall']:.4f}")
        print("="*50)
        
        # Interpretation
        print("\nInterpretation:")
        print("- Faithfulness: How well the answer is grounded in context")
        print("- Answer Relevancy: How relevant the answer is to the question")
        print("- Context Precision: How relevant the retrieved context is")
        print("- Context Recall: How much of the ground truth is in context")
        print("\nAll metrics range from 0 to 1, higher is better.")


if __name__ == "__main__":
    evaluator = RagasEvaluator()
    metrics = evaluator.run_evaluation()
    evaluator.print_results(metrics)
