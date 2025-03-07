import unittest
import os
import json
import pandas as pd
from multibandit_sampler import GPUReactionThompsonSampler
from thompson_sampling import DisallowTracker

class TestAmideSampler(unittest.TestCase):
    def setUp(self):
        # Load example configuration
        with open('examples/amide_fp_sim.json', 'r') as f:
            self.config = json.load(f)
        
        # Create test reaction file
        self.test_reaction_file = "test_amide_reactions.csv"
        self.test_reactions = pd.DataFrame({
            'smarts': [
                '[C:1]C(=O)O.[C:2]N>>[C:1]C(=O)N[C:2]',  # Direct amide formation
                '[C:1]C(=O)Cl.[C:2]N>>[C:1]C(=O)N[C:2]',  # Acid chloride amidation
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

        # Create test reagent files from the data folder
        self.test_reagent_files = []
        reagent_files = [
            'data/carboxylic_acids_100.smi',
            'data/primary_amines_100.smi'
        ]
        
        for i, reagent_file in enumerate(reagent_files):
            test_file = f"test_reagents_{i}.csv"
            # Read the reagent file and convert to CSV format
            with open(reagent_file, 'r') as f:
                lines = f.readlines()[:10]  # Use first 10 lines for testing
                df = pd.DataFrame([line.strip().split() for line in lines], columns=['smiles', 'name'])
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
        """Test that the sampler can be initialized with the amide example data."""
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
        self.assertEqual(len(sampler.reagent_lists), 2)  # carboxylic acids and primary amines

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

    def test_reaction_selection(self):
        """Test that reaction selection works with the amide reactions."""
        sampler = GPUReactionThompsonSampler(
            reaction_smarts_list=self.test_reactions['smarts'].tolist(),
            reaction_names=self.test_reactions['name'].tolist(),
            mode=self.config['ts_mode']
        )
        
        # Load reagent lists
        for reagent_file in self.test_reagent_files:
            sampler.load_reagent_list(reagent_file)
        
        # Test that all reactions are properly initialized
        for arm in sampler.reaction_arms:
            self.assertIn(arm.reaction_smarts, self.test_reactions['smarts'].tolist())
            self.assertIn(arm.reaction_name, self.test_reactions['name'].tolist())

if __name__ == '__main__':
    unittest.main() 