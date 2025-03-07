import random
import numpy as np
import cupy as cp
from typing import List, Tuple, Any, Optional

# Import the existing ThompsonSampler and DisallowTracker from your codebase.
# Adjust the import if your file structure is different.
from thompson_sampling import ThompsonSampler, DisallowTracker

################################################################
# ReactionArm: Represents one reaction template with its own bandit stats.
################################################################
class ReactionArm:
    def __init__(self, reaction_smarts: str, prior_mean: float = 0.5, prior_std: float = 0.2):
        """
        Initializes the reaction arm.
        :param reaction_smarts: Reaction SMARTS string.
        :param prior_mean: Initial mean (prior for the bandit)
        :param prior_std: Initial uncertainty (std)
        """
        self.reaction_smarts = reaction_smarts
        self.current_mean = prior_mean
        self.current_std = prior_std
        self.num_evaluations = 0

    def update(self, score: float):
        """
        Incrementally update the reaction arm statistics with the new score.
        Uses a simple incremental update formula.
        """
        self.num_evaluations += 1
        alpha = 1.0 / self.num_evaluations
        # Update mean using exponential moving average:
        self.current_mean = (1 - alpha) * self.current_mean + alpha * score
        # Update std in a rudimentary way; for production consider using Welford's algorithm.
        diff = abs(score - self.current_mean)
        self.current_std = (1 - alpha) * self.current_std + alpha * diff


################################################################
# ReactionThompsonSampler: Extends the regular ThompsonSampler by
# incorporating a list of reaction arms. For each cycle it selects a
# reaction using Thompson sampling over the reaction arms, then
# selects reagents as before.
################################################################
class ReactionThompsonSampler(ThompsonSampler):
    def __init__(self, reaction_smarts_list: List[str], mode: str = "maximize", log_filename: Optional[str] = None):
        """
        Initializes the ReactionThompsonSampler.
        :param reaction_smarts_list: List of reaction SMARTS strings.
        :param mode: "maximize" or "minimize"
        :param log_filename: Optional log file.
        """
        super().__init__(mode, log_filename)
        self.reaction_arms: List[ReactionArm] = []
        self.set_reaction_arms(reaction_smarts_list)

    def set_reaction_arms(self, reaction_smarts_list: List[str]):
        """
        Takes a list of reaction SMARTS and creates ReactionArm objects.
        """
        self.reaction_arms = []
        for rxn in reaction_smarts_list:
            # Set default priors; you can make these parameters too.
            self.reaction_arms.append(ReactionArm(rxn, prior_mean=0.5, prior_std=0.2))

    def random_reagent_selection(self) -> List[int]:
        """
        Helper method: Randomly selects a reagent index for each reagent list.
        (Used in warm-up)
        """
        selection = []
        for reagent_list in self.reagent_lists:
            selection.append(random.choice(range(len(reagent_list))))
        return selection

    def warm_up(self, num_warmup_trials: int = 3) -> List[Tuple[str, List[int], float]]:
        """
        Warm-up phase for both reaction and reagents. Iterates over each reaction arm and performs
        a few random reagent combinations, evaluates them, and updates the reaction bandit.
        :return: List of tuples: (reaction_smarts, selected_reagents, score)
        """
        warmup_results = []
        for arm in self.reaction_arms:
            # Set the reaction for the current arm.
            self.set_reaction(arm.reaction_smarts)
            for _ in range(num_warmup_trials):
                # Use random reagent selection
                selected_reagents = self.random_reagent_selection()
                # Evaluate candidate; using your existing evaluate method.
                # The evaluate() method returns (product_smiles, product_name, score)
                smiles, name, score = self.evaluate(selected_reagents)
                arm.update(score)
                warmup_results.append((arm.reaction_smarts, selected_reagents, score))
        return warmup_results

    def search(self, num_cycles: int = 25) -> List[Tuple[str, List[int], str, float]]:
        """
        Runs the multi-bandit search over reactions and reagents.
        In each cycle:
          1. Sample a reaction using Thompson sampling over reaction arms.
          2. Set the reaction and then select reagents using the existing TS logic.
          3. Execute the reaction and evaluate the product.
          4. Update both reagent bandits and the reaction arm.
        :return: List of tuples: (reaction_smarts, selected_reagents, product_smiles, score)
        """
        results = []
        rng = np.random.default_rng()
        for i in range(num_cycles):
            # 1. Reaction selection: sample a value per reaction arm.
            reaction_samples = []
            for arm in self.reaction_arms:
                sample = rng.normal(scale=arm.current_std) + arm.current_mean
                reaction_samples.append(sample)
            reaction_idx = np.nanargmax(reaction_samples)
            selected_reaction = self.reaction_arms[reaction_idx].reaction_smarts
            self.set_reaction(selected_reaction)

            # 2. Reagent selection: use the existing TS logic to choose reagents.
            selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
            for cycle_id in random.sample(range(len(self.reagent_lists)), len(self.reagent_lists)):
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

            # 3. Execute reaction and evaluate product.
            smiles, name, score = self.evaluate(selected_reagents)

            # 4. Update reagent statistics (as in your existing code).
            for component_idx, reagent_idx in enumerate(selected_reagents):
                self.reagent_lists[component_idx][reagent_idx].add_score(score)
            # 5. Update the chosen reaction arm.
            self.reaction_arms[reaction_idx].update(score)

            results.append((selected_reaction, selected_reagents, smiles, score))
            self.logger.info(f"Iteration {i}: Reaction: {selected_reaction}, Reagents: {selected_reagents}, Score: {score}")
        return results

################################################################
# GPUReactionArm: GPU-accelerated version of ReactionArm
################################################################
class GPUReactionArm:
    def __init__(self, reaction_smarts: str, reaction_name: str, prior_mean: float = 0.5, prior_std: float = 0.2):
        """
        Initializes the GPU reaction arm.
        :param reaction_smarts: Reaction SMARTS string
        :param reaction_name: Name/identifier of the reaction
        :param prior_mean: Initial mean (prior for the bandit)
        :param prior_std: Initial uncertainty (std)
        """
        self.reaction_smarts = reaction_smarts
        self.reaction_name = reaction_name
        self.current_mean = cp.array(prior_mean)
        self.current_std = cp.array(prior_std)
        self.num_evaluations = 0

    def update(self, score: float):
        """
        GPU-accelerated update of reaction arm statistics.
        """
        self.num_evaluations += 1
        alpha = 1.0 / self.num_evaluations
        # Update mean using exponential moving average
        self.current_mean = (1 - alpha) * self.current_mean + alpha * cp.array(score)
        # Update std
        diff = cp.abs(cp.array(score) - self.current_mean)
        self.current_std = (1 - alpha) * self.current_std + alpha * diff


################################################################
# GPUReactionThompsonSampler: GPU-accelerated version of ReactionThompsonSampler
################################################################
class GPUReactionThompsonSampler(GPUThompsonSampler):
    def __init__(self, reaction_smarts_list: List[str], reaction_names: List[str], mode: str = "maximize", log_filename: Optional[str] = None):
        """
        Initializes the GPU ReactionThompsonSampler.
        :param reaction_smarts_list: List of reaction SMARTS strings
        :param reaction_names: List of reaction names/identifiers
        :param mode: "maximize" or "minimize"
        :param log_filename: Optional log file
        """
        super().__init__(mode, log_filename)
        self.reaction_arms: List[GPUReactionArm] = []
        self.set_reaction_arms(reaction_smarts_list, reaction_names)

    def set_reaction_arms(self, reaction_smarts_list: List[str], reaction_names: List[str]):
        """
        Creates GPUReactionArm objects from SMARTS and names.
        """
        if len(reaction_smarts_list) != len(reaction_names):
            raise ValueError("reaction_smarts_list and reaction_names must have the same length")
        self.reaction_arms = []
        for smarts, name in zip(reaction_smarts_list, reaction_names):
            self.reaction_arms.append(GPUReactionArm(smarts, name))

    def random_reagent_selection(self) -> List[int]:
        """
        GPU-accelerated random reagent selection.
        """
        selection = []
        for reagent_list in self.reagent_lists:
            selection.append(int(cp.random.randint(0, len(reagent_list))))
        return selection

    def warm_up(self, num_warmup_trials: int = 3) -> List[Tuple[str, str, List[int], float]]:
        """
        GPU-accelerated warm-up phase.
        :return: List of tuples: (reaction_name, reaction_smarts, selected_reagents, score)
        """
        warmup_results = []
        for arm in self.reaction_arms:
            self.set_reaction(arm.reaction_smarts)
            for _ in range(num_warmup_trials):
                selected_reagents = self.random_reagent_selection()
                smiles, name, score = self.evaluate(selected_reagents)
                arm.update(score)
                warmup_results.append((arm.reaction_name, arm.reaction_smarts, selected_reagents, score))
        return warmup_results

    def search(self, num_cycles: int = 25) -> List[Tuple[str, str, List[int], str, float]]:
        """
        GPU-accelerated multi-bandit search.
        :return: List of tuples: (reaction_name, reaction_smarts, selected_reagents, product_smiles, score)
        """
        results = []
        for i in range(num_cycles):
            # 1. Reaction selection using GPU
            reaction_samples = cp.array([cp.random.normal(arm.current_mean, arm.current_std) 
                                      for arm in self.reaction_arms])
            reaction_idx = int(cp.argmax(reaction_samples))
            selected_reaction = self.reaction_arms[reaction_idx]
            self.set_reaction(selected_reaction.reaction_smarts)

            # 2. Reagent selection using GPU
            selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
            for cycle_id in random.sample(range(len(self.reagent_lists)), len(self.reagent_lists)):
                reagent_list = self.reagent_lists[cycle_id]
                selected_reagents[cycle_id] = DisallowTracker.To_Fill
                disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                
                # GPU-accelerated sampling
                stds = cp.array([r.current_std for r in reagent_list])
                mu = cp.array([r.current_mean for r in reagent_list])
                choice_row = cp.random.normal(size=len(reagent_list)) * stds + mu
                if disallow_mask:
                    choice_row[cp.array(list(disallow_mask))] = cp.nan
                selected_reagents[cycle_id] = self.pick_function(choice_row.get())
            
            self._disallow_tracker.update(selected_reagents)

            # 3. Execute and evaluate
            smiles, name, score = self.evaluate(selected_reagents)

            # 4. Update statistics
            for component_idx, reagent_idx in enumerate(selected_reagents):
                self.reagent_lists[component_idx][reagent_idx].add_score(score)
            selected_reaction.update(score)

            results.append((selected_reaction.reaction_name, selected_reaction.reaction_smarts, 
                          selected_reagents, smiles, score))
            self.logger.info(f"Iteration {i}: Reaction: {selected_reaction.reaction_name}, "
                           f"Reagents: {selected_reagents}, Score: {score}")
        return results

# For convenience, you could add a GPU version below by subclassing your GPUThompsonSampler similarly.