import unittest
import numpy as np
import cupy as cp
from multibandit_sampler import (
    ReactionArm, ReactionThompsonSampler,
    GPUReactionArm, GPUReactionThompsonSampler
)
from thompson_sampling import DisallowTracker

class TestReactionArm(unittest.TestCase):
    def setUp(self):
        self.arm = ReactionArm("test_smarts", prior_mean=0.5, prior_std=0.2)
        self.gpu_arm = GPUReactionArm("test_smarts", "test_name", prior_mean=0.5, prior_std=0.2)

    def test_initialization(self):
        self.assertEqual(self.arm.reaction_smarts, "test_smarts")
        self.assertEqual(self.arm.current_mean, 0.5)
        self.assertEqual(self.arm.current_std, 0.2)
        self.assertEqual(self.arm.num_evaluations, 0)

    def test_update(self):
        self.arm.update(0.8)
        self.assertEqual(self.arm.num_evaluations, 1)
        self.assertGreater(self.arm.current_mean, 0.5)
        self.assertGreater(self.arm.current_std, 0.2)

    def test_gpu_initialization(self):
        self.assertEqual(self.gpu_arm.reaction_smarts, "test_smarts")
        self.assertEqual(self.gpu_arm.reaction_name, "test_name")
        self.assertTrue(isinstance(self.gpu_arm.current_mean, cp.ndarray))
        self.assertTrue(isinstance(self.gpu_arm.current_std, cp.ndarray))

    def test_gpu_update(self):
        self.gpu_arm.update(0.8)
        self.assertEqual(self.gpu_arm.num_evaluations, 1)
        self.assertTrue(isinstance(self.gpu_arm.current_mean, cp.ndarray))
        self.assertTrue(isinstance(self.gpu_arm.current_std, cp.ndarray))

class TestReactionThompsonSampler(unittest.TestCase):
    def setUp(self):
        self.reaction_smarts = ["rxn1", "rxn2", "rxn3"]
        self.reaction_names = ["Reaction1", "Reaction2", "Reaction3"]
        self.sampler = ReactionThompsonSampler(self.reaction_smarts, mode="maximize")
        self.gpu_sampler = GPUReactionThompsonSampler(
            self.reaction_smarts, self.reaction_names, mode="maximize"
        )

    def test_initialization(self):
        self.assertEqual(len(self.sampler.reaction_arms), 3)
        self.assertEqual(len(self.gpu_sampler.reaction_arms), 3)
        for arm in self.sampler.reaction_arms:
            self.assertIsInstance(arm, ReactionArm)
        for arm in self.gpu_sampler.reaction_arms:
            self.assertIsInstance(arm, GPUReactionArm)

    def test_set_reaction_arms(self):
        new_smarts = ["new1", "new2"]
        new_names = ["New1", "New2"]
        self.gpu_sampler.set_reaction_arms(new_smarts, new_names)
        self.assertEqual(len(self.gpu_sampler.reaction_arms), 2)
        with self.assertRaises(ValueError):
            self.gpu_sampler.set_reaction_arms(new_smarts, ["New1"])

    def test_random_reagent_selection(self):
        selection = self.sampler.random_reagent_selection()
        self.assertIsInstance(selection, list)
        gpu_selection = self.gpu_sampler.random_reagent_selection()
        self.assertIsInstance(gpu_selection, list)

    def test_warm_up(self):
        results = self.sampler.warm_up(num_warmup_trials=1)
        self.assertIsInstance(results, list)
        gpu_results = self.gpu_sampler.warm_up(num_warmup_trials=1)
        self.assertIsInstance(gpu_results, list)

    def test_search(self):
        results = self.sampler.search(num_cycles=1)
        self.assertIsInstance(results, list)
        gpu_results = self.gpu_sampler.search(num_cycles=1)
        self.assertIsInstance(gpu_results, list)

if __name__ == '__main__':
    unittest.main() 