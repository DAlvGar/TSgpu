import json
import logging
from typing import Dict, Any
from thompson_sampling import ThompsonSampler, GPUThompsonSampler
from multibandit_sampler import ReactionThompsonSampler, GPUReactionThompsonSampler
from reaction_utils import read_reaction_file, validate_reaction_file

def setup_logging(log_filename: str = None) -> logging.Logger:
    """Sets up logging configuration."""
    logger = logging.getLogger('thompson_sampling')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_filename:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_fields = [
        'reaction_file',
        'reagent_file_list',
        'evaluator_class_name',
        'evaluator_arg',
        'num_ts_iterations',
        'num_warmup_trials',
        'ts_mode'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    return config

def main():
    # Load configuration
    config = load_config('config.json')
    
    # Setup logging
    logger = setup_logging(config.get('log_filename'))
    logger.info("Starting Thompson Sampling optimization")
    
    # Validate reaction file
    if not validate_reaction_file(config['reaction_file']):
        raise ValueError("Invalid reaction file format")
    
    # Read reaction data
    reaction_smarts, reaction_names = read_reaction_file(config['reaction_file'])
    logger.info(f"Loaded {len(reaction_smarts)} reactions from file")
    
    # Initialize sampler based on GPU preference
    use_gpu = config.get('use_gpu', False)
    if use_gpu:
        sampler = GPUReactionThompsonSampler(
            reaction_smarts_list=reaction_smarts,
            reaction_names=reaction_names,
            mode=config['ts_mode'],
            log_filename=config.get('log_filename')
        )
        logger.info("Using GPU-accelerated sampler")
    else:
        sampler = ReactionThompsonSampler(
            reaction_smarts_list=reaction_smarts,
            mode=config['ts_mode'],
            log_filename=config.get('log_filename')
        )
        logger.info("Using CPU sampler")
    
    # Set up evaluator
    evaluator_class = globals()[config['evaluator_class_name']]
    evaluator = evaluator_class(**config['evaluator_arg'])
    sampler.set_evaluator(evaluator)
    
    # Load reagent lists
    for reagent_file in config['reagent_file_list']:
        sampler.load_reagent_list(reagent_file)
    
    # Run warm-up phase
    logger.info("Starting warm-up phase")
    warmup_results = sampler.warm_up(num_warmup_trials=config['num_warmup_trials'])
    logger.info(f"Completed warm-up phase with {len(warmup_results)} trials")
    
    # Run main search
    logger.info("Starting main search phase")
    results = sampler.search(num_cycles=config['num_ts_iterations'])
    logger.info(f"Completed search phase with {len(results)} iterations")
    
    # Save results
    output_file = config.get('output_file', 'results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'warmup_results': warmup_results,
            'search_results': results
        }, f, indent=2)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 