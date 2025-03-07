import unittest
import os
import json
import pandas as pd
from multibandit_sampler import GPUReactionThompsonSampler
from thompson_sampling import DisallowTracker

class TestQuinazolineSampler(unittest.TestCase):
    def setUp(self):
        # Load example configuration
        with open('examples/quinazoline_fp_sim.json', 'r') as f:
            self.config = json.load(f)
        
        # Create test reaction file
        self.test_reaction_file = "test_quinazoline_reactions.csv"
        self.test_reactions = pd.DataFrame({
            'smarts': [
                '[C:1]C(=O)O.[C:2]N>>[C:1]C(=O)N[C:2]',  # Amide formation
                '[C:1]C(=O)Cl.[C:2]N>>[C:1]C(=O)N[C:2]',  # Amide from acid chloride
                '[C:1]C(=O)O.[C:2]NH2>>[C:1]C(=O)N[C:2]',  # Primary amide
                '[C:1]C(=O)O.[C:2]N(C)C>>[C:1]C(=O)N([C:2])C'  # N-alkyl amide
            ],
            'name': [
                'Direct Amide Formation',
                'Acid Chloride Amidation',
                'Primary Amide Formation',
                'N-Alkyl Amide Formation'
            ]
        })
        self.test_reactions.to_csv(self.test_reaction_file, header=False, index=False, sep='\t')

        # Create test reagent files
        self.test_reagent_files = []
        for i, reagent_file in enumerate(self.config['reagent_file_list']):
            test_file = f"test_reagents_{i}.csv"
            # Read first 10 lines of each reagent file for testing
            df = pd.read_csv(reagent_file, nrows=10)
            df.to_csv(test_file, index=False)
            self.test_reagent_files.append(test_file)

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_reaction_file):
            os.remove(self.test_reaction_file)
        for file in self.test_reagent_files:
            if os.path.exists(file):
                os.remove(file)

    def test_sampler_initialization(self):
        """Test that the sampler can be initialized with the quinazoline example data."""
        sampler = GPUReactionThompsonSampler(
            reaction_smarts_list=self.test_reactions['smarts'].tolist(),
            reaction_names=self.test_reactions['name'].tolist(),
            mode=self.config['ts_mode']
        )
        
        # Set up evaluator
        evaluator_class = globals()[self.config['evaluator_class_name']]
        evaluator = evaluator_class(**self.config['evaluator_arg'])
        sampler.set_evaluator(evaluator)
        
        # Load reagent lists
        for reagent_file in self.test_reagent_files:
            sampler.load_reagent_list(reagent_file)
        
        self.assertEqual(len(sampler.reaction_arms), 4)
        self.assertEqual(len(sampler.reagent_lists), len(self.test_reagent_files))

    def test_warm_up_phase(self):
        """Test the warm-up phase with real chemical data."""
        sampler = GPUReactionThompsonSampler(
            reaction_smarts_list=self.test_reactions['smarts'].tolist(),
            reaction_names=self.test_reactions['name'].tolist(),
            mode=self.config['ts_mode']
        )
        
        # Set up evaluator
        evaluator_class = globals()[self.config['evaluator_class_name']]
        evaluator = evaluator_class(**self.config['evaluator_arg'])
        sampler.set_evaluator(evaluator)
        
        # Load reagent lists
        for reagent_file in self.test_reagent_files:
            sampler.load_reagent_list(reagent_file)
        
        # Run warm-up
        warmup_results = sampler.warm_up(num_warmup_trials=2)
        
        self.assertEqual(len(warmup_results), 8)  # 4 reactions * 2 trials
        for result in warmup_results:
            self.assertEqual(len(result), 4)  # (reaction_name, reaction_smarts, selected_reagents, score)
            self.assertIsInstance(result[3], float)  # score should be a float

    def test_search_phase(self):
        """Test the main search phase with real chemical data."""
        sampler = GPUReactionThompsonSampler(
            reaction_smarts_list=self.test_reactions['smarts'].tolist(),
            reaction_names=self.test_reactions['name'].tolist(),
            mode=self.config['ts_mode']
        )
        
        # Set up evaluator
        evaluator_class = globals()[self.config['evaluator_class_name']]
        evaluator = evaluator_class(**self.config['evaluator_arg'])
        sampler.set_evaluator(evaluator)
        
        # Load reagent lists
        for reagent_file in self.test_reagent_files:
            sampler.load_reagent_list(reagent_file)
        
        # Run search
        results = sampler.search(num_cycles=2)
        
        self.assertEqual(len(results), 2)  # 2 cycles
        for result in results:
            self.assertEqual(len(result), 5)  # (reaction_name, reaction_smarts, selected_reagents, product_smiles, score)
            self.assertIsInstance(result[4], float)  # score should be a float
            self.assertIsInstance(result[3], str)  # product_smiles should be a string

    def test_disallow_tracker(self):
        """Test that the disallow tracker works with the real chemical data."""
        sampler = GPUReactionThompsonSampler(
            reaction_smarts_list=self.test_reactions['smarts'].tolist(),
            reaction_names=self.test_reactions['name'].tolist(),
            mode=self.config['ts_mode']
        )
        
        # Load reagent lists
        for reagent_file in self.test_reagent_files:
            sampler.load_reagent_list(reagent_file)
        
        # Test initial state
        self.assertEqual(len(sampler._disallow_tracker._initial_reagent_counts), len(self.test_reagent_files))
        
        # Test mask generation
        selected_reagents = [DisallowTracker.Empty] * len(self.test_reagent_files)
        mask = sampler._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
        self.assertIsInstance(mask, set)

if __name__ == '__main__':
    unittest.main() 