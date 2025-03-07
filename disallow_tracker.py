""" Class for random sampling without replacement from a combinatorial library"""
import itertools
import random
from collections import defaultdict
from typing import DefaultDict, Set, Tuple, List, Union

import numpy as np
import cupy as cp   


class DisallowTracker:
    Empty = -1  # a sentinel value meaning fill this reagent spot - used in selection
    To_Fill = None  # a sentinel value meaning this reagent spot is open - used in selection

    def __init__(self, reagent_counts: list[int]):
        """
        Basic Init
        :param reagent_counts: A list of the number of reagents for each site of diversity in the reaction

        For example if the library to search has 3 reactants the first with 10 options, the second with 20
        and the third with 34 then reagent_counts would be the list [10, 20, 34]
        """

        self._initial_reagent_counts = np.array(reagent_counts)
        self._reagent_exhaust_counts = self._get_reagent_exhaust_counts()

        # this is where we keep track of the disallowed combinations
        self._disallow_mask: DefaultDict[Tuple[int | None], Set] = defaultdict(set)
        self._n_sampled = 0 ## number of products sampled
        self._total_product_size = np.prod(reagent_counts)

    @property
    def n_cycles(self) -> int:
        """ How many cycles are then in this reaction"""
        return len(self._initial_reagent_counts)

    def get_disallowed_selection_mask(self, current_selection: list[int | None]) -> Set[int]:
        """ Returns the disallowed reagents given the current_selection
        :param current_selection:  list of ints denoting the current selection
        :return: set[int] of the indices that are disallowed

        Current_selection is of length "n_cycles" for the current reaction and filled at each position
        with either Disallow_tracker.Empty, Disallow_tracker.To_Fill, or int >= 0
        additionally, Disallow_tracker.To_Fill can appear only once.
        """

        """Returns the disallowed reagents given the current"""

        if len(current_selection) != self.n_cycles:
            raise ValueError(f"current_selection must be equal in length to number of sites "
                             f"({self.n_cycles} for reaction")
        if len([v for v in current_selection if v == DisallowTracker.To_Fill]) != 1:
            raise ValueError(f"current_selection must have exactly one To_Fill slot.")

        return self._disallow_mask[tuple(current_selection)]

    def retire_one_synthon(self, cycle_id: int, synthon_index: int):
        retire_mask = [self.Empty] * self.n_cycles
        retire_mask[cycle_id] = synthon_index
        self._retire_synthon_mask(retire_mask=retire_mask)

    def _retire_synthon_mask(self, retire_mask: list[int]):
        # get the list of cycles that we need to search for retiring
        if retire_mask.count(self.Empty) == 0:
            # if n_to_fill is one - then we have a completed mask (all spot filled)
            # so say that we sampled this synthon by updating the counts
            self._n_sampled += 1
            # and then update the disallow tracker
            self._update(retire_mask)
        else:
            for cycle_id in [i for i in range(self.n_cycles) if retire_mask[i] == self.Empty]:
                # mark which cycle we are going to search for synthons that can be paired with the synthon we are retiring
                retire_mask[cycle_id] = self.To_Fill
                ts_locations = np.ones(shape=self._initial_reagent_counts[cycle_id])
                # update ts_locations
                disallowed_selections = self.get_disallowed_selection_mask(retire_mask)
                # added this to catch cases where a reaction fails or a reagent doesn't score - PW
                if len(disallowed_selections):
                    ts_locations[np.array(list(disallowed_selections))] = np.NaN
                # anything that is not nan is still in play so we need to denote
                # that pairing it with the synthon we will retire is not allowed
                for synthon_idx in np.argwhere(~np.isnan(ts_locations)).flatten():
                    retire_mask[cycle_id] = synthon_idx
                    self._retire_synthon_mask(retire_mask=retire_mask)

    def update(self, selected: list[int | None]) -> None:
        """
        Updates the disallow tracker with the selected reagents.
        :param selected: list[int]

        This means that this particular reagent combination will not be sampled again.

        Selected is the list of indexes that maps to what reagent was used at what position
        For example selected = [4, 5, 3]
              means reagent 4 at position 0
                    reagent 5 at position 1
                    reagent 3 at position 2
              will not be sampled again

        Two sentinel values are used in this routine:
            to_fill = None
            empty = -1
        """
        if len(selected) != self.n_cycles:
            msg = f"DisallowTracker selected size {len(selected)} but reaction has {self.n_cycles} sites of diversity"
            raise ValueError(msg)
        for site_id, sel, max_size in zip(list(range(self.n_cycles)), selected, self._initial_reagent_counts):
            if sel is not None and sel >= max_size:
                raise ValueError(f"Disallowed given index {sel} for site {site_id} which has {max_size} reagents")

        # all ok so call the internal update
        self._update(selected)

    def sample(self) -> list[int]:
        """ Randomly sample from the reaction without replacement"""
        if self._n_sampled == self._total_product_size:
            raise ValueError(f"Sampled {self._n_sampled} of {self._total_product_size} products in reaction - "
                             f"there are no more left to sample")
        selection_mask: list[int | None] = [self.Empty] * self.n_cycles
        selection_order: list[int] = list(range(self.n_cycles))
        random.shuffle(selection_order)
        for cycle_id in selection_order:
            selection_mask[cycle_id] = DisallowTracker.To_Fill
            selection_candidate_scores = np.random.uniform(size=self._initial_reagent_counts[cycle_id])
            selection_candidate_scores[list(self._disallow_mask[tuple(selection_mask)])] = np.NaN
            selection_mask[cycle_id] = np.nanargmax(selection_candidate_scores).item(0)
        self.update(selection_mask)
        self._n_sampled += 1
        return selection_mask

    def _get_reagent_exhaust_counts(self) -> dict[tuple[int,], int]:
        """
        Returns a dictionary denoting when reagents for a site are exhausted.

        For example is the reagent counts are [3,4,5] then the dictionary looks like this:
            {(0,): 20, (1,): 15, (2,): 12, (0, 1): 5, (0, 2): 4, (1, 2): 3}
        which means that a *particular* reagent at:
            site 0 is exhausted if it has been sampled in 20 (4*5) molecules,
            site 1 is exhausted if it has been sampled in 15 (3*5) molecules,
            ...
        also:
            A *particular pair* of reagents used in sites (0,2) are exhausted if they have been sampled 4 times
            and so on for the other reagents sites

        :return: dict[tuple[int,], int]
        """
        s = range(self.n_cycles)
        all_set = {*list(range(self.n_cycles))}
        power_set = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, self.n_cycles))
        return {p: np.prod(self._initial_reagent_counts[list(all_set - {*list(p)})]) for p in power_set}

    def _update(self, selected: list[int | None]):
        """ Does the updates to the disallow masks w/o the error checking of parameters"""
        # ok now start the disallow fun
        for idx, value in enumerate(selected):
            # save what reagent_index was used at current position and set this
            # index to 'to_fill' then update the dictionary index by selected (w/None)
            # to include the value - meaning that this value can not be selected if the remaining
            # synthons are selected
            #   [4, 5, 3] -> [None, 5, 3]  then set indexed by disallow_mask[[None,5,3]] gets 'value' added
            #    meaning that 4 can not be selected as a reagent when [None, 5, 3] is selected in the select step
            selected[idx] = self.To_Fill
            if value is not None and value not in self._disallow_mask[tuple(selected)]:
                if value != self.Empty:
                    self._disallow_mask[tuple(selected)].add(value)
                    # now we get the counts to see if we need to retire a reagent so
                    # get the key, and then get the count for that key
                    count_key = tuple([r_pos for r_pos, r_idx in enumerate(selected) if r_idx != self.To_Fill])
                    if self._reagent_exhaust_counts[count_key] == len(self._disallow_mask[tuple(selected)]):
                        # Here comes the confusing part.  If we have exhausted a reagent then we need
                        # to update the disallow for 'empty' pairs.
                        # Let say we are updating [None, 5, 3] and have exhausted pair 5,3 then we need to
                        # say that if we get the pattern [empty, 5, to_fill] during the select step
                        # meaning that we spot 0 is empty, 1 has reagent id 5 and 2 is currently being selected
                        # that we do not select a 3 for position 2 or a 5 for position 1.
                        # The following recursive call deals with that.

                        self._update([self.Empty if v == self.To_Fill else v for v in selected])
            selected[idx] = value


class GPUDisallowTracker:
    Empty = -1
    To_Fill = None

    def __init__(self, reagent_counts: list[int]):
        """
        GPU-compatible version of DisallowTracker
        :param reagent_counts: A list of the number of reagents for each site of diversity in the reaction
        """
        # Store original counts as integers
        self._initial_reagent_counts = [int(x) for x in reagent_counts]
        # GPU array version for calculations
        self._initial_reagent_counts_gpu = cp.array(reagent_counts)
        self._reagent_exhaust_counts = self._get_reagent_exhaust_counts()
        
        # Store disallow masks as GPU arrays for each position
        self._disallow_masks = [cp.zeros(count, dtype=bool) for count in self._initial_reagent_counts]
        self._position_pairs_masks = {}  # Store masks for position pairs
        
        self._n_sampled = 0
        self._total_product_size = int(cp.prod(self._initial_reagent_counts_gpu))
        
        # Pre-allocate GPU arrays for batch operations
        self._max_batch_size = 1024  # Can be adjusted based on GPU memory
        self._batch_masks = [cp.zeros((self._max_batch_size, count), dtype=bool) 
                           for count in self._initial_reagent_counts]

    def get_disallowed_selection_mask_batch(self, batch_size: int) -> List[cp.ndarray]:
        \"\"\"Returns disallowed reagents for batch processing\"\"\"
        batch_size = min(int(batch_size), self._max_batch_size)
        return [mask[:batch_size] for mask in self._batch_masks]

    def get_disallowed_selection_mask(self, current_selection: list[Union[int, None]]) -> cp.ndarray:
        \"\"\"Returns disallowed reagents for single selection\"\"\"
        if len(current_selection) != len(self._initial_reagent_counts):
            raise ValueError(f"current_selection must match number of sites: {len(self._initial_reagent_counts)}")
        
        to_fill_pos = None
        fixed_positions = {}
        
        # Find which position needs to be filled and store fixed positions
        for pos, val in enumerate(current_selection):
            if val == self.To_Fill:
                if to_fill_pos is not None:
                    raise ValueError("Only one To_Fill position allowed")
                to_fill_pos = pos
            elif val != self.Empty:
                fixed_positions[pos] = int(val)
        
        if to_fill_pos is None:
            raise ValueError("Must have one To_Fill position")
            
        # Start with base mask for the position
        mask = self._disallow_masks[to_fill_pos].copy()
        
        # Update mask based on fixed positions
        for pos, val in fixed_positions.items():
            pair_key = tuple(sorted([pos, to_fill_pos]))
            if pair_key in self._position_pairs_masks:
                pair_mask = self._position_pairs_masks[pair_key][val]
                mask = mask | pair_mask
                
        return mask

    def update(self, selected: List[int]) -> None:
        \"\"\"Update disallow masks with new selection\"\"\"
        if len(selected) != len(self._initial_reagent_counts):
            raise ValueError(f"selected size {len(selected)} must match number of sites")
            
        # Convert selected indices to integers
        selected = [int(val) for val in selected]
            
        # Update individual position masks
        for pos, val in enumerate(selected):
            if val >= self._initial_reagent_counts[pos]:
                raise ValueError(f"Invalid index {val} for position {pos}")
            self._disallow_masks[pos][val] = True
            
        # Update position pairs masks
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pair_key = (i, j)
                if pair_key not in self._position_pairs_masks:
                    self._position_pairs_masks[pair_key] = [
                        cp.zeros(self._initial_reagent_counts[j], dtype=bool)
                        for _ in range(self._initial_reagent_counts[i])
                    ]
                self._position_pairs_masks[pair_key][selected[i]][selected[j]] = True
                
        self._n_sampled += 1

    def sample_batch(self, batch_size: int) -> cp.ndarray:
        \"\"\"Sample multiple combinations in parallel\"\"\"
        batch_size = int(batch_size)
        if self._n_sampled + batch_size > self._total_product_size:
            raise ValueError("Not enough combinations remaining")
            
        batch_size = min(batch_size, self._max_batch_size)
        n_positions = len(self._initial_reagent_counts)
        
        # Initialize result array
        result = cp.zeros((batch_size, n_positions), dtype=cp.int32)
        
        # Generate random selection order for each batch item
        selection_order = cp.array([
            cp.random.permutation(n_positions).get()  # Convert to CPU for indexing
            for _ in range(batch_size)
        ])
        
        # Process each position
        for pos_idx in range(n_positions):
            # Get current position for each batch item
            current_pos = selection_order[:, pos_idx]
            
            # Generate random scores
            max_reagents = max(self._initial_reagent_counts)
            scores = cp.random.uniform(
                size=(batch_size, max_reagents)
            )
            
            # Apply masks
            for b in range(batch_size):
                pos = int(current_pos[b])
                mask = self.get_disallowed_selection_mask(
                    [self.To_Fill if i == pos else 
                     int(result[b, i]) if i < pos_idx else 
                     self.Empty for i in range(n_positions)]
                )
                scores[b, :len(mask)] = cp.where(mask, -cp.inf, scores[b, :len(mask)])
                scores[b, len(mask):] = -cp.inf  # Mask out invalid indices
                
            # Select best valid option
            selected = cp.argmax(scores, axis=1)
            
            # Store selections
            for b in range(batch_size):
                result[b, int(current_pos[b])] = selected[b]
        
        # Update disallow masks
        for selections in result:
            self.update(selections.get().tolist())
            
        return result

    def _get_reagent_exhaust_counts(self) -> dict:
        \"\"\"Calculate reagent exhaustion counts\"\"\"
        s = range(len(self._initial_reagent_counts))
        all_set = set(s)
        power_set = itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(1, len(self._initial_reagent_counts))
        )
        return {
            p: int(cp.prod(self._initial_reagent_counts_gpu[list(all_set - set(p))]).get())
            for p in power_set
        }