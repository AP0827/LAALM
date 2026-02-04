"""
Master Script: Run All Ablation Studies and Theoretical Analyses

This script runs all Gap #2 and Gap #3 experiments in sequence.

Usage:
    python run_all_experiments.py --test-set samples/ --output results/
    
    # Or run specific experiments:
    python run_all_experiments.py --test-set samples/ --experiments ablation_attention theory_orthogonality
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time


EXPERIMENTS = {
    # Gap #2: Ablation Studies
    'ablation_attention': {
        'script': 'ablation_attention.py',
        'description': 'Attention mechanism vs baselines',
        'estimated_time': '30-60 min',
        'gap': 2
    },
    'ablation_hyperparameters': {
        'script': 'ablation_hyperparameters.py',
        'description': 'Hyperparameter sensitivity analysis',
        'estimated_time': '60-90 min',
        'gap': 2
    },
    'ablation_llm': {
        'script': 'ablation_llm.py',
        'description': 'LLM provider comparison (Groq vs Ollama vs None)',
        'estimated_time': '30-45 min',
        'gap': 2
    },
    
    # Gap #3: Theoretical Analysis
    'theory_orthogonality': {
        'script': 'theory_error_orthogonality.py',
        'description': 'Error orthogonality analysis',
        'estimated_time': '20-30 min',
        'gap': 3
    },
    'theory_distributional': {
        'script': 'theory_distributional_recovery.py',
        'description': 'N-Best distributional recovery',
        'estimated_time': '20-30 min',
        'gap': 3
    }
}


def run_experiment(experiment_name: str, test_set: str, output_dir: str):
    """Run a single experiment."""
    exp_config = EXPERIMENTS[experiment_name]
    script_path = os.path.join(os.path.dirname(__file__), exp_config['script'])
    
    print(f"\n{'='*80}")
    print(f"RUNNING: {experiment_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Estimated time: {exp_config['estimated_time']}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Run script
    cmd = [
        sys.executable,
        script_path,
        '--test-set', test_set,
        '--output', output_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        
        print(f"\n✅ {experiment_name} completed in {elapsed/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {experiment_name} failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all ablation studies and theoretical analyses")
    parser.add_argument('--test-set', type=str, required=True, help="Path to test set directory")
    parser.add_argument('--output', type=str, default='results/', help="Output directory")
    parser.add_argument('--experiments', nargs='+', default=['all'],
                       choices=['all'] + list(EXPERIMENTS.keys()),
                       help="Which experiments to run")
    parser.add_argument('--gap', type=int, choices=[2, 3],
                       help="Run only experiments for specific gap")
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = args.experiments
    
    # Filter by gap if specified
    if args.gap:
        experiments_to_run = [
            exp for exp in experiments_to_run 
            if EXPERIMENTS[exp]['gap'] == args.gap
        ]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Test set: {args.test_set}")
    print(f"Output directory: {args.output}")
    print(f"Experiments to run: {len(experiments_to_run)}")
    print(f"\nExperiments:")
    for exp in experiments_to_run:
        config = EXPERIMENTS[exp]
        print(f"  - {exp} (Gap #{config['gap']}): {config['description']}")
    print(f"{'='*80}\n")
    
    # Confirm
    response = input("Proceed with experiments? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    total_start = time.time()
    results = {}
    
    for exp_name in experiments_to_run:
        success = run_experiment(exp_name, args.test_set, args.output)
        results[exp_name] = 'success' if success else 'failed'
    
    total_elapsed = time.time() - total_start
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nResults:")
    
    for exp_name, status in results.items():
        symbol = '✅' if status == 'success' else '❌'
        print(f"  {symbol} {exp_name}: {status}")
    
    success_count = sum(1 for s in results.values() if s == 'success')
    print(f"\nSuccess rate: {success_count}/{len(results)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
