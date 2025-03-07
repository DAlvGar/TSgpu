import random
from typing import List, Optional, Tuple

import functools
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
import cupy as cp
from multiprocessing import Pool

from disallow_tracker import DisallowTracker, GPUDisallowTracker
from reagent import Reagent
from ts_logger import get_logger
from ts_utils import read_reagents
from evaluators import DBEvaluator



class ThompsonSampler:
    def __init__(self, mode="maximize", log_filename: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self._disallow_tracker = None
        self.hide_progress = False
        self._mode = mode
        if self._mode == "maximize":
            self.pick_function = np.nanargmax
            self._top_func = max
        elif self._mode == "minimize":
            self.pick_function = np.nanargmin
            self._top_func = min
        elif self._mode == "maximize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = max
        elif self._mode == "minimize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = min
        else:
            raise ValueError(f"{mode} is not a supported argument")
        self._warmup_std = None

    def _boltzmann_reweighted_pick(self, scores: np.ndarray):
        """Rather than choosing the top sampled score, use a reweighted probability.

        Zhao, H., Nittinger, E. & Tyrchan, C. Enhanced Thompson Sampling by Roulette
        Wheel Selection for Screening Ultra-Large Combinatorial Libraries.
        bioRxiv 2024.05.16.594622 (2024) doi:10.1101/2024.05.16.594622
        suggested several modifications to the Thompson Sampling procedure.
        This method implements one of those, namely a Boltzmann style probability distribution
        from the sampled values. The reagent is chosen based on that distribution rather than
        simply the max sample.
        """
        if self._mode == "minimize_boltzmann":
            scores = -scores
        exp_terms = np.exp(scores / self._warmup_std)
        probs = exp_terms / np.nansum(exp_terms)
        probs[np.isnan(probs)] = 0.0
        return np.random.choice(probs.shape[0], p=probs)

    def set_hide_progress(self, hide_progress: bool) -> None:
        """
        Hide the progress bars
        :param hide_progress: set to True to hide the progress baars
        """
        self.hide_progress = hide_progress

    def read_reagents(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents(reagent_file_list, num_to_select)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])

    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes an RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def evaluate(self, choice_list: List[int]) -> Tuple[str, str, float]:
        """Evaluate a set of reagents
        :param choice_list: list of reagent ids
        :return: smiles for the reaction product, score for the reaction product
        """
        selected_reagents = []
        for idx, choice in enumerate(choice_list):
            component_reagent_list = self.reagent_lists[idx]
            selected_reagents.append(component_reagent_list[choice])
        prod = self.reaction.RunReactants([reagent.mol for reagent in selected_reagents])
        product_name = "_".join([reagent.reagent_name for reagent in selected_reagents])
        res = np.nan
        product_smiles = "FAIL"
        if prod:
            prod_mol = prod[0][0]  # RunReactants returns Tuple[Tuple[Mol]]
            Chem.SanitizeMol(prod_mol)
            product_smiles = Chem.MolToSmiles(prod_mol)
            if isinstance(self.evaluator, DBEvaluator):
                res = self.evaluator.evaluate(product_name)
                res = float(res)
            else:
                res = self.evaluator.evaluate(prod_mol)
            if np.isfinite(res):
                [reagent.add_score(res) for reagent in selected_reagents]
        return product_smiles, product_name, res

    def warm_up(self, num_warmup_trials=3):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners
        :param num_warmup_trials: number of times to sample each reagent
        """
        # get the list of reagent indices
        idx_list = list(range(0, len(self.reagent_lists)))
        # get the number of reagents for each component in the reaction
        reagent_count_list = [len(x) for x in self.reagent_lists]
        warmup_results = []
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            # The number of reagents for this component
            current_max = reagent_count_list[i]
            # For each reagent...
            for j in tqdm(range(0, current_max), desc=f"Warmup {i + 1} of {len(idx_list)}", disable=self.hide_progress):
                # For each warmup trial...
                for k in range(0, num_warmup_trials):
                    current_list = [DisallowTracker.Empty] * len(idx_list)
                    current_list[i] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                    if j not in disallow_mask:
                        ## ok we can select this reagent
                        current_list[i] = j
                        # Randomly select reagents for each additional component of the reaction
                        for p in partner_list:
                            # tell the disallow tracker which site we are filling
                            current_list[p] = DisallowTracker.To_Fill
                            # get the new disallow mask
                            disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                            selection_scores = np.random.uniform(size=reagent_count_list[p])
                            # null out the disallowed ones
                            selection_scores[list(disallow_mask)] = np.NaN
                            # and select a random one
                            current_list[p] = np.nanargmax(selection_scores).item(0)
                        self._disallow_tracker.update(current_list)
                        product_smiles, product_name, score = self.evaluate(current_list)
                        if np.isfinite(score):
                            warmup_results.append([score, product_smiles, product_name])

        warmup_scores = [ws[0] for ws in warmup_results]
        self.logger.info(
            f"warmup score stats: "
            f"cnt={len(warmup_scores)}, "
            f"mean={np.mean(warmup_scores):0.4f}, "
            f"std={np.std(warmup_scores):0.4f}, "
            f"min={np.min(warmup_scores):0.4f}, "
            f"max={np.max(warmup_scores):0.4f}")
        # initialize each reagent
        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std
        for i in range(0, len(self.reagent_lists)):
            for j in range(0, len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                except ValueError:
                    self.logger.info(f"Skipping reagent {reagent.reagent_name} because there were no successful evaluations during warmup")
                    self._disallow_tracker.retire_one_synthon(i, j)
        self.logger.info(f"Top score found during warmup: {max(warmup_scores):.3f}")
        return warmup_results

    def search(self, num_cycles=25):
        """Run the search
        :param: num_cycles: number of search iterations
        :return: a list of SMILES and scores
        """
        out_list = []
        rng = np.random.default_rng()
        for i in tqdm(range(0, num_cycles), desc="Cycle", disable=self.hide_progress):
            selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
            for cycle_id in random.sample(range(0, len(self.reagent_lists)), len(self.reagent_lists)):
                reagent_list = self.reagent_lists[cycle_id]
                selected_reagents[cycle_id] = DisallowTracker.To_Fill
                disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                stds = np.array([r.current_std for r in reagent_list])
                mu = np.array([r.current_mean for r in reagent_list])
                choice_row = rng.normal(size=len(reagent_list)) * stds + mu
                if disallow_mask:
                    choice_row[np.array(list(disallow_mask))] = np.NaN
                selected_reagents[cycle_id] = self.pick_function(choice_row)
            self._disallow_tracker.update(selected_reagents)
            # Select a reagent for each component, according to the choice function
            smiles, name, score = self.evaluate(selected_reagents)
            if np.isfinite(score):
                out_list.append([score, smiles, name])
            if i % 100 == 0:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info(f"Iteration: {i} max score: {top_score:2f} smiles: {top_smiles} {top_name}")
        return out_list

    def search_batch(self, num_cycles=25, batch_size=100):
        """Run the search with batch evaluation
        :param num_cycles: number of search iterations
        :param batch_size: number of molecules to evaluate in parallel
        :return: a list of SMILES and scores
        """
        out_list = []
        rng = np.random.default_rng()
        
        for i in tqdm(range(0, num_cycles), desc="Cycle", disable=self.hide_progress):
            # Generate multiple candidates in parallel
            batch_candidates = []
            batch_names = []
            batch_mols = []
            
            for _ in range(batch_size):
                selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
                for cycle_id in random.sample(range(0, len(self.reagent_lists)), len(self.reagent_lists)):
                    reagent_list = self.reagent_lists[cycle_id]
                    selected_reagents[cycle_id] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                    
                    # Thompson sampling for this component
                    stds = np.array([r.current_std for r in reagent_list])
                    mu = np.array([r.current_mean for r in reagent_list])
                    choice_row = rng.normal(size=len(reagent_list)) * stds + mu
                    if disallow_mask:
                        choice_row[np.array(list(disallow_mask))] = np.NaN
                    selected_reagents[cycle_id] = self.pick_function(choice_row)
                
                self._disallow_tracker.update(selected_reagents)
                batch_candidates.append(selected_reagents)
                
                # Generate product molecule
                selected_reagents_objs = [self.reagent_lists[idx][choice] 
                                        for idx, choice in enumerate(selected_reagents)]
                prod = self.reaction.RunReactants([r.mol for r in selected_reagents_objs])
                
                if prod:
                    prod_mol = prod[0][0]
                    try:
                        Chem.SanitizeMol(prod_mol)
                        batch_mols.append(prod_mol)
                        batch_names.append("_".join([r.reagent_name for r in selected_reagents_objs]))
                    except:
                        batch_mols.append(None)
                        batch_names.append(None)
                else:
                    batch_mols.append(None)
                    batch_names.append(None)
            
            # Batch evaluate valid molecules
            valid_mols = [m for m in batch_mols if m is not None]
            valid_indices = [i for i, m in enumerate(batch_mols) if m is not None]
            
            if valid_mols:
                if isinstance(self.evaluator, (GPUFPEvaluator, FPEvaluator)):
                    # Use batch evaluation for fingerprint-based evaluators
                    scores = self.evaluator.evaluate_batch(valid_mols)
                else:
                    # Fallback to sequential evaluation for other evaluators
                    scores = [self.evaluator.evaluate(mol) for mol in valid_mols]
                
                # Update reagent scores and collect results
                for idx, score in zip(valid_indices, scores):
                    if np.isfinite(score):
                        candidates = batch_candidates[idx]
                        # Update scores for selected reagents
                        for component_idx, reagent_idx in enumerate(candidates):
                            self.reagent_lists[component_idx][reagent_idx].add_score(score)
                        
                        out_list.append([score, Chem.MolToSmiles(batch_mols[idx]), batch_names[idx]])
            
            if i % 10 == 0:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info(f"Iteration: {i} max score: {top_score:2f} smiles: {top_smiles} {top_name}")
        
        return out_list

    def _run_reactions_parallel(self, batch_candidates, num_workers=4):
        """Run reactions in parallel using multiprocessing"""
        with Pool(num_workers) as pool:
            batch_results = pool.map(self._process_single_reaction, batch_candidates)
        return batch_results

    def _generate_fps_batch(self, mols):
        """Generate fingerprints in batch with caching"""
        fps = []
        for mol in mols:
            smi = Chem.MolToSmiles(mol, canonical=True)
            if smi in self._fp_cache:
                fps.append(self._fp_cache[smi])
            else:
                fp = self.fpgen.GetFingerprint(mol)
                self._fp_cache[smi] = fp
                fps.append(fp)
        return fps

class GPUThompsonSampler(ThompsonSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_rng_normal = cp.random.normal
        self._fp_cache = {}  # Cache for fingerprints
        self._disallow_tracker = None  # Will be initialized as GPUDisallowTracker
        
    def read_reagents(self, reagent_file_list, num_to_select=None):
        super().read_reagents(reagent_file_list, num_to_select)
        # Replace CPU tracker with GPU tracker
        reagent_counts = [len(x) for x in self.reagent_lists]
        self._disallow_tracker = GPUDisallowTracker(reagent_counts)    
    
    def _sample_choices_gpu(self, means, stds, mask=None, batch_size=100):
        """Sample multiple choices in parallel on GPU"""
        # Move data to GPU
        means_gpu = cp.asarray(means)
        stds_gpu = cp.asarray(stds)
        
        # Generate samples
        samples = self.gpu_rng_normal(size=(batch_size, len(means))) * stds_gpu + means_gpu
        
        if mask is not None:
            mask_gpu = cp.asarray(mask)
            samples[:, mask_gpu] = cp.nan
        
        # Get choices
        if self._mode == "maximize":
            choices = cp.nanargmax(samples, axis=1)
        else:
            choices = cp.nanargmin(samples, axis=1)
        
        return cp.asnumpy(choices)

    def _process_batch_reactions(self, batch_candidates, num_workers=4):
        """Process reactions in parallel for a batch of candidates
        
        Args:
            batch_candidates: List of selected reagent indices for each candidate
            num_workers: Number of CPU workers for parallel processing
            
        Returns:
            Tuple of (batch_mols, batch_names)
        """
        batch_mols = []
        batch_names = []
        
        # Process reactions in parallel using multiprocessing
        with Pool(num_workers) as pool:
            results = pool.starmap(self._process_single_candidate, 
                                 [(candidate,) for candidate in batch_candidates])
        
        for mol, name in results:
            batch_mols.append(mol)
            batch_names.append(name)
            
        return batch_mols, batch_names
    
    def _process_single_candidate(self, selected_reagents):
        """Process a single reaction candidate
        
        Args:
            selected_reagents: List of reagent indices
            
        Returns:
            Tuple of (molecule, name) or (None, None) if reaction fails
        """
        try:
            selected_reagents_objs = [self.reagent_lists[idx][choice] 
                                    for idx, choice in enumerate(selected_reagents)]
            
            prod = self.reaction.RunReactants([r.mol for r in selected_reagents_objs])
            
            if prod:
                prod_mol = prod[0][0]
                Chem.SanitizeMol(prod_mol)
                name = "_".join([r.reagent_name for r in selected_reagents_objs])
                return prod_mol, name
        except:
            pass
        
        return None, None
    
    def search_batch_gpu(self, num_cycles=25, batch_size=100, num_workers=4):
        """GPU-accelerated batch search
        
        Args:
            num_cycles: Number of search iterations
            batch_size: Number of molecules to evaluate in parallel
            num_workers: Number of CPU workers for parallel reaction processing
            
        Returns:
            List of [score, SMILES, name] for successful candidates
        """
        out_list = []
        
        for i in tqdm(range(0, num_cycles), desc="Cycle", disable=self.hide_progress):
            # Pre-compute means and stds for each component
            component_stats = []
            for reagent_list in self.reagent_lists:
                means = np.array([r.current_mean for r in reagent_list])
                stds = np.array([r.current_std for r in reagent_list])
                component_stats.append((means, stds))
            
            # Sample all components in parallel for the batch
            batch_candidates = []
            for _ in range(batch_size):
                selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
                
                # Sample each component using GPU
                for component_idx in range(len(self.reagent_lists)):
                    means, stds = component_stats[component_idx]
                    selected_reagents[component_idx] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                    
                    choices = self._sample_choices_gpu(means, stds, disallow_mask, batch_size=1)[0]
                    selected_reagents[component_idx] = choices
                
                self._disallow_tracker.update(selected_reagents)
                batch_candidates.append(selected_reagents)
            
            # Process reactions in parallel
            batch_mols, batch_names = self._process_batch_reactions(batch_candidates, num_workers)
            
            # Filter valid molecules
            valid_mols = [m for m in batch_mols if m is not None]
            valid_indices = [i for i, m in enumerate(batch_mols) if m is not None]
            
            if valid_mols:
                if isinstance(self.evaluator, (GPUFPEvaluator, FPEvaluator)):
                    # Use GPU batch evaluation
                    scores = self.evaluator.evaluate_batch(valid_mols)
                else:
                    # Fallback to sequential evaluation
                    scores = [self.evaluator.evaluate(mol) for mol in valid_mols]
                
                # Update reagent scores and collect results
                for idx, score in zip(valid_indices, scores):
                    if np.isfinite(score):
                        candidates = batch_candidates[idx]
                        # Update scores for selected reagents
                        for component_idx, reagent_idx in enumerate(candidates):
                            self.reagent_lists[component_idx][reagent_idx].add_score(score)
                        
                        mol = batch_mols[idx]
                        out_list.append([score, Chem.MolToSmiles(mol), batch_names[idx]])
            
            # Log progress
            if i % 10 == 0:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info(f"Iteration: {i} max score: {top_score:2f} smiles: {top_smiles} {top_name}")
        
        return out_list

    def warm_up_gpu(self, num_warmup_trials=3, batch_size=100, num_workers=4):
        """GPU-accelerated warm-up phase
        
        Args:
            num_warmup_trials: Number of trials per reagent
            batch_size: Batch size for parallel processing
            num_workers: Number of CPU workers for parallel reaction processing
            
        Returns:
            List of warmup results
        """
        warmup_results = []
        idx_list = list(range(len(self.reagent_lists)))
        reagent_count_list = [len(x) for x in self.reagent_lists]
        
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            current_max = reagent_count_list[i]
            
            # Process reagents in batches
            for j in tqdm(range(0, current_max, batch_size), 
                         desc=f"Warmup {i + 1} of {len(idx_list)}", 
                         disable=self.hide_progress):
                
                batch_end = min(j + batch_size, current_max)
                batch_candidates = []
                
                # Generate candidates for the batch
                for reagent_idx in range(j, batch_end):
                    for _ in range(num_warmup_trials):
                        current_list = [DisallowTracker.Empty] * len(idx_list)
                        current_list[i] = reagent_idx
                        
                        # Randomly select partners using GPU
                        for p in partner_list:
                            current_list[p] = DisallowTracker.To_Fill
                            disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                            choices = self._sample_choices_gpu(
                                np.ones(reagent_count_list[p]),  # Equal probability
                                np.zeros(reagent_count_list[p]), # No uncertainty
                                disallow_mask,
                                batch_size=1
                            )[0]
                            current_list[p] = choices
                        
                        self._disallow_tracker.update(current_list)
                        batch_candidates.append(current_list)
                
                # Process reactions and evaluate
                batch_mols, batch_names = self._process_batch_reactions(batch_candidates, num_workers)
                
                valid_mols = [m for m in batch_mols if m is not None]
                valid_indices = [i for i, m in enumerate(batch_mols) if m is not None]
                
                if valid_mols:
                    if isinstance(self.evaluator, (GPUFPEvaluator, FPEvaluator)):
                        scores = self.evaluator.evaluate_batch(valid_mols)
                    else:
                        scores = [self.evaluator.evaluate(mol) for mol in valid_mols]
                    
                    for idx, score in zip(valid_indices, scores):
                        if np.isfinite(score):
                            mol = batch_mols[idx]
                            warmup_results.append([score, Chem.MolToSmiles(mol), batch_names[idx]])
        
        # Initialize reagents with warmup results
        warmup_scores = [ws[0] for ws in warmup_results]
        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std
        
        for i in range(len(self.reagent_lists)):
            for j in range(len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                except ValueError:
                    self.logger.info(f"Skipping reagent {reagent.reagent_name} due to no successful evaluations")
                    self._disallow_tracker.retire_one_synthon(i, j)
        
        self.logger.info(f"Top score found during warmup: {max(warmup_scores):.3f}")
        return warmup_results
