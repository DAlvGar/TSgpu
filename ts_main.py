#!/usr/bin/env python

import importlib
import json
import sys
import traceback
from datetime import timedelta
from timeit import default_timer as timer


import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
from multiprocessing import Pool

from thompson_sampling import ThompsonSampler, GPUThompsonSampler
from ts_logger import get_logger
from evaluators import GPUFPEvaluator, FPEvaluator


def read_input(json_filename: str) -> dict:
    """
    Read input parameters from a json file
    :param json_filename: input json file
    :return: a dictionary with the input parameters
    """
    input_data = None
    with open(json_filename, 'r') as ifs:
        input_data = json.load(ifs)
        module = importlib.import_module("evaluators")
        evaluator_class_name = input_data["evaluator_class_name"]
        class_ = getattr(module, evaluator_class_name)
        evaluator_arg = input_data["evaluator_arg"]
        evaluator = class_(evaluator_arg)
        input_data['evaluator_class'] = evaluator
    return input_data


def parse_input_dict(input_data: dict) -> None:
    """
    Parse the input dictionary and add the necessary information
    :param input_data:
    """
    module = importlib.import_module("evaluators")
    evaluator_class_name = input_data["evaluator_class_name"]
    class_ = getattr(module, evaluator_class_name)
    evaluator_arg = input_data["evaluator_arg"]
    evaluator = class_(evaluator_arg)
    input_data['evaluator_class'] = evaluator


def run_ts(input_dict: dict, hide_progress: bool = False) -> None:
    """
    Perform a Thompson sampling run
    :param hide_progress: hide the progress bar
    :param input_dict: dictionary with input parameters
    """
    evaluator = input_dict["evaluator_class"]
    reaction_smarts = input_dict["reaction_smarts"]
    num_ts_iterations = input_dict["num_ts_iterations"]
    reagent_file_list = input_dict["reagent_file_list"]
    num_warmup_trials = input_dict["num_warmup_trials"]
    result_filename = input_dict.get("results_filename")
    ts_mode = input_dict["ts_mode"]
    log_filename = input_dict.get("log_filename")
    logger = get_logger(__name__, filename=log_filename)
    ts = GPUThompsonSampler(mode=ts_mode)
    ts.set_hide_progress(hide_progress)
    ts.set_evaluator(evaluator)
    ts.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
    ts.set_reaction(reaction_smarts)
    # run the warm-up phase to generate an initial set of scores for each reagent
    warmup_results = ts.warm_up(num_warmup_trials=num_warmup_trials)
    # run the search with TS
    results = ts.search_batch_gpu(num_cycles=num_ts_iterations, batch_size=100)
    total_evaluations = evaluator.counter
    percent_searched = total_evaluations / ts.get_num_prods() * 100
    logger.info(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")
    # write the results to disk
    out_df = pd.DataFrame(results, columns=["score", "SMILES", "Name"])
    if result_filename is not None:
        out_df.to_csv(result_filename, index=False)
        logger.info(f"Saved results to: {result_filename}")
    if not hide_progress:
        if ts_mode == "maximize":
            print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))
        else:
            print(out_df.sort_values("score", ascending=True).drop_duplicates(subset="SMILES").head(10))
    return out_df


def run_10_cycles():
    """ A testing function for the paper
    :return: None
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in range(0, 10):
        input_dict['results_filename'] = f"ts_result_{i:03d}.csv"
        run_ts(input_dict, hide_progress=False)


def compare_iterations():
    """ A testing function for the paper
    :return:
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in (2, 5, 10, 50, 100):
        num_ts_iterations = i * 1000
        input_dict["num_ts_iterations"] = num_ts_iterations
        input_dict["results_filename"] = f"iteration_test_{i}K.csv"
        run_ts(input_dict)

def test_gpu_implementation():
    """Test GPU implementation with different scenarios and compare with CPU version
    :return: None
    """
    import os
    from datetime import datetime
    import pandas as pd
    import numpy as np
    from timeit import default_timer as timer
    
    # Setup logging
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("gpu_test_results", exist_ok=True)
    log_file = f"gpu_test_results/gpu_test_{test_timestamp}.log"
    logger = get_logger(__name__, filename=log_file)
    
    # Test configurations
    test_configs = [
        {
            'name': 'small_fp_similarity',
            'json_file': 'examples/quinazoline_fp_100.json',
            'description': 'Small fingerprint similarity test with 100 reagents'
        },
        {
            'name': 'medium_fp_similarity',
            'json_file': 'examples/quinazoline_fp_sim.json',
            'description': 'Medium fingerprint similarity test'
        },
        {
            'name': 'small_amide_formation',
            'json_file': 'examples/amide_fp_sim.json',
            'description': 'Small amide formation test'
        }
    ]
    
    # Batch sizes to test
    batch_sizes = [32, 64, 128, 256]
    
    # Results storage
    results = []
    
    for config in test_configs:
        logger.info(f"\\nTesting configuration: {config['name']}")
        logger.info(config['description'])
        
        # Read input configuration
        input_dict = read_input(config['json_file'])
        
        # Test different batch sizes
        for batch_size in batch_sizes:
            logger.info(f"\\nTesting with batch_size={batch_size}")
            total_time_gpu = 0
            total_time_cpu = 0
            
            # GPU Test
            try:
                # GPU Implementation
                start_gpu = timer()
                
                ts_gpu = GPUThompsonSampler(mode="maximize")
                ts_gpu.set_hide_progress(True)
                ts_gpu.set_evaluator(input_dict['evaluator_class'])
                ts_gpu.read_reagents(reagent_file_list=input_dict['reagent_file_list'])
                ts_gpu.set_reaction(input_dict['reaction_smarts'])
                
                # Warm-up phase
                warmup_start = timer()
                warmup_results_gpu = ts_gpu.warm_up_gpu(
                    num_warmup_trials=input_dict['num_warmup_trials'],
                    batch_size=batch_size
                )
                warmup_time = timer() - warmup_start
                
                # Search phase
                search_start = timer()
                results_gpu = ts_gpu.search_batch_gpu(
                    num_cycles=input_dict['num_ts_iterations'],
                    batch_size=batch_size
                )
                search_time = timer() - search_start
                
                total_time_gpu = timer() - start_gpu
                
                # Calculate metrics
                gpu_scores = [r[0] for r in results_gpu]
                gpu_max_score = max(gpu_scores)
                gpu_mean_score = np.mean(gpu_scores)
                
                logger.info(f"GPU Implementation Results:")
                logger.info(f"Total time: {total_time_gpu:.2f}s")
                logger.info(f"Warm-up time: {warmup_time:.2f}s")
                logger.info(f"Search time: {search_time:.2f}s")
                logger.info(f"Max score: {gpu_max_score:.4f}")
                logger.info(f"Mean score: {gpu_mean_score:.4f}")
                logger.info(f"Number of valid molecules: {len(results_gpu)}")
                
                # Store results
                results.append({
                    'config_name': config['name'],
                    'batch_size': batch_size,
                    'implementation': 'GPU',
                    'total_time': total_time_gpu,
                    'warmup_time': warmup_time,
                    'search_time': search_time,
                    'max_score': gpu_max_score,
                    'mean_score': gpu_mean_score,
                    'valid_molecules': len(results_gpu),
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"GPU implementation failed: {str(e)}")
                traceback.print_exc()
                results.append({
                    'config_name': config['name'],
                    'batch_size': batch_size,
                    'implementation': 'GPU',
                    'error': str(e)
                })
            
            # CPU Test for comparison
            try:
                start_cpu = timer()
                
                ts_cpu = ThompsonSampler(mode="maximize")
                ts_cpu.set_hide_progress(True)
                ts_cpu.set_evaluator(input_dict['evaluator_class'])
                ts_cpu.read_reagents(reagent_file_list=input_dict['reagent_file_list'])
                ts_cpu.set_reaction(input_dict['reaction_smarts'])
                
                # Warm-up phase
                warmup_start = timer()
                warmup_results_cpu = ts_cpu.warm_up(num_warmup_trials=input_dict['num_warmup_trials'])
                warmup_time = timer() - warmup_start
                
                # Search phase
                search_start = timer()
                results_cpu = ts_cpu.search(num_cycles=input_dict['num_ts_iterations'])
                search_time = timer() - search_start
                
                total_time_cpu = timer() - start_cpu
                
                # Calculate metrics
                cpu_scores = [r[0] for r in results_cpu]
                cpu_max_score = max(cpu_scores)
                cpu_mean_score = np.mean(cpu_scores)
                
                logger.info(f"\\nCPU Implementation Results:")
                logger.info(f"Total time: {total_time_cpu:.2f}s")
                logger.info(f"Warm-up time: {warmup_time:.2f}s")
                logger.info(f"Search time: {search_time:.2f}s")
                logger.info(f"Max score: {cpu_max_score:.4f}")
                logger.info(f"Mean score: {cpu_mean_score:.4f}")
                logger.info(f"Number of valid molecules: {len(results_cpu)}")
                
                # Calculate speedup
                speedup = total_time_cpu / total_time_gpu
                logger.info(f"\\nGPU Speedup: {speedup:.2f}x")
                
                # Store results
                results.append({
                    'config_name': config['name'],
                    'batch_size': batch_size,
                    'implementation': 'CPU',
                    'total_time': total_time_cpu,
                    'warmup_time': warmup_time,
                    'search_time': search_time,
                    'max_score': cpu_max_score,
                    'mean_score': cpu_mean_score,
                    'valid_molecules': len(results_cpu),
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"CPU implementation failed: {str(e)}")
                traceback.print_exc()
                results.append({
                    'config_name': config['name'],
                    'batch_size': batch_size,
                    'implementation': 'CPU',
                    'error': str(e)
                })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_file = f"gpu_test_results/performance_comparison_{test_timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"\\nResults saved to {results_file}")
    
    # Generate summary statistics
    summary = results_df[results_df['error'].isna()].groupby(
        ['config_name', 'batch_size', 'implementation']
    ).agg({
        'total_time': ['mean', 'std'],
        'max_score': ['mean', 'std'],
        'valid_molecules': 'mean'
    }).round(4)
    
    summary_file = f"gpu_test_results/summary_{test_timestamp}.csv"
    summary.to_csv(summary_file)
    logger.info(f"Summary statistics saved to {summary_file}")
    
    return results_df, summary

def plot_performance_comparison(results_df):
    """Generate performance comparison plots
    :param results_df: results dataframe
    :return: None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn')
    
    # Create directory for plots
    os.makedirs("gpu_test_results/plots", exist_ok=True)
    
    # Filter out failed runs
    valid_results = results_df[results_df['error'].isna()]
    
    # Plot 1: Execution time comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=valid_results, x='config_name', y='total_time', 
                hue='implementation', ci='sd')
    plt.title('Total Execution Time Comparison')
    plt.xlabel('Test Configuration')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('gpu_test_results/plots/execution_time_comparison.png')
    
    # Plot 2: Batch size impact
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=valid_results[valid_results['implementation'] == 'GPU'],
                 x='batch_size', y='total_time', hue='config_name', marker='o')
    plt.title('Impact of Batch Size on GPU Performance')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('gpu_test_results/plots/batch_size_impact.png')
    
    # Plot 3: Score comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=valid_results, x='config_name', y='max_score',
                hue='implementation')
    plt.title('Maximum Score Comparison')
    plt.xlabel('Test Configuration')
    plt.ylabel('Max Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('gpu_test_results/plots/score_comparison.png')

def main():
    if len(sys.argv) > 1:
        # Normal execution with JSON file
        start = timer()
        json_filename = sys.argv[1]
        input_dict = read_input(json_filename)
        run_ts(input_dict)
        end = timer()
        print("Elapsed time", timedelta(seconds=end - start))
    else:
        # Run GPU implementation tests
        results_df, summary = test_gpu_implementation()
        plot_performance_comparison(results_df)


if __name__ == "__main__":
    main()
