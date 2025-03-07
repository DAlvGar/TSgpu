import numpy as np
import pytest
import cupy as cp

from disallow_tracker import DisallowTracker, GPUDisallowTracker


To_Fill = DisallowTracker.To_Fill
Empty = DisallowTracker.Empty

def test_disallow_tracker_complete():
    sizes = [5, 8, 9]
    total = np.prod(sizes)
    d_tracker = DisallowTracker(sizes)
    s = set()
    for i in range(total):
        res = d_tracker.sample()
        s.add(tuple(res))
    assert len(s) == total

def test_disallow_throws_when_full():
    sizes = [5, 8, 9]
    total = np.prod(sizes)
    d_tracker = DisallowTracker(sizes)
    s = set()
    for i in range(total):
        res = d_tracker.sample()
        s.add(tuple(res))
    assert len(s) == total
    with pytest.raises(ValueError):
        d_tracker.sample()

def test_disallow_simple():
    sizes = [3, 4, 5]

    d_tracker = DisallowTracker(sizes)
    assert d_tracker.get_disallowed_selection_mask([To_Fill, 2, 3]) == set()
    assert d_tracker.get_disallowed_selection_mask([1, To_Fill, 3]) == set()
    assert d_tracker.get_disallowed_selection_mask([1, 2, To_Fill]) == set()

    d_tracker.update([1, 2, 3])

    assert d_tracker.get_disallowed_selection_mask([To_Fill, 2, 3]) == set([1])
    assert d_tracker.get_disallowed_selection_mask([1, To_Fill, 3]) == set([2])
    assert d_tracker.get_disallowed_selection_mask([1, 2, To_Fill]) == set([3])

    d_tracker.update([0, 2, 3])

    assert d_tracker.get_disallowed_selection_mask([To_Fill, 2, 3]) == set([0, 1])
    assert d_tracker.get_disallowed_selection_mask([1, To_Fill, 3]) == set([2])
    assert d_tracker.get_disallowed_selection_mask([1, 2, To_Fill]) == set([3])
    assert d_tracker.get_disallowed_selection_mask([0, To_Fill, 3]) == set([2])
    assert d_tracker.get_disallowed_selection_mask([0, 2, To_Fill]) == set([3])

def test_disallow_reagent_exhausted():
    sizes = [3, 4, 5]

    d_tracker = DisallowTracker(sizes)

    # This will fully exhaust reagent position 0 for the [To_Fill, 1, 1] case
    d_tracker.update([0, 1, 1])
    d_tracker.update([1, 1, 1])
    d_tracker.update([2, 1, 1])

    # The important tests that we propogated to the Empty with To_Fill cases for reagent 0
    assert d_tracker.get_disallowed_selection_mask([Empty, To_Fill, 1]) == set([1])
    assert d_tracker.get_disallowed_selection_mask([Empty, 1, To_Fill]) == set([1])
    # Shouldn't really get here in practice (because of the above check), but good to double check.
    assert d_tracker.get_disallowed_selection_mask([To_Fill, 1, 1]) == set([0, 1, 2])

    # If we select the 0th reagent first, this is just the regular exclusion
    for reagent_0 in range(3):
        assert d_tracker.get_disallowed_selection_mask([reagent_0, 1, To_Fill]) == set([1])
        assert d_tracker.get_disallowed_selection_mask([reagent_0, To_Fill, 1]) == set([1])
        # Nothing propagated to the other cases
        assert d_tracker.get_disallowed_selection_mask([reagent_0, Empty, To_Fill]) == set([])
        assert d_tracker.get_disallowed_selection_mask([reagent_0, To_Fill, Empty]) == set([])


def test_gpu_disallow_tracker_basic():
    sizes = [3, 4, 5]
    tracker = GPUDisallowTracker(sizes)
    
    # Test initial state
    assert tracker._total_product_size == 60
    assert len(tracker._disallow_masks) == 3
    
    # Test mask shapes
    assert tracker._disallow_masks[0].shape == (3,)
    assert tracker._disallow_masks[1].shape == (4,)
    assert tracker._disallow_masks[2].shape == (5,)

def test_gpu_disallow_tracker_batch_sampling():
    sizes = [3, 4, 5]
    tracker = GPUDisallowTracker(sizes)
    
    # Test batch sampling
    batch_size = 10
    results = tracker.sample_batch(batch_size)
    
    # Check results shape and uniqueness
    assert results.shape == (batch_size, len(sizes))
    
    # Convert to numpy for easier testing
    results_np = cp.asnumpy(results)
    
    # Check valid ranges
    assert np.all(results_np[:, 0] < sizes[0])
    assert np.all(results_np[:, 1] < sizes[1])
    assert np.all(results_np[:, 2] < sizes[2])
    
    # Check uniqueness
    unique_combinations = set(map(tuple, results_np))
    assert len(unique_combinations) == batch_size

def test_gpu_disallow_tracker_mask_updates():
    sizes = [3, 4, 5]
    tracker = GPUDisallowTracker(sizes)
    
    # Make a selection
    selection = [1, 2, 3]
    tracker.update(selection)
    
    # Test mask updates
    mask = tracker.get_disallowed_selection_mask([GPUDisallowTracker.To_Fill, 2, 3])
    assert cp.asnumpy(mask)[1]  # Position 1 should be disallowed
    
    # Test batch mask
    batch_masks = tracker.get_disallowed_selection_mask_batch(2)
    assert len(batch_masks) == 3  # One for each position
    assert batch_masks[0].shape == (2, sizes[0])

def test_gpu_disallow_tracker_exhaustion():
    sizes = [2, 2, 2]
    tracker = GPUDisallowTracker(sizes)
    
    # Sample all combinations
    total_combinations = np.prod(sizes)
    results = tracker.sample_batch(total_combinations)
    
    # Try to sample more - should raise error
    with pytest.raises(ValueError):
        tracker.sample_batch(1)

def test_gpu_vs_cpu_consistency():
    sizes = [3, 4, 5]
    gpu_tracker = GPUDisallowTracker(sizes)
    cpu_tracker = DisallowTracker(sizes)
    
    # Make same selection on both
    selection = [1, 2, 3]
    gpu_tracker.update(selection)
    cpu_tracker.update(selection)
    
    # Compare masks
    gpu_mask = gpu_tracker.get_disallowed_selection_mask([GPUDisallowTracker.To_Fill, 2, 3])
    cpu_mask = cpu_tracker.get_disallowed_selection_mask([DisallowTracker.To_Fill, 2, 3])
    
    # Convert GPU mask to CPU for comparison
    gpu_mask_np = cp.asnumpy(gpu_mask)
    assert set(np.where(gpu_mask_np)[0]) == cpu_mask

if __name__ == "__main__":
    pytest.main([__file__])