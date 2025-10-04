#!/usr/bin/env python3
"""
Test script for the symmetric autoregressive sampler.
"""

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric


def test_physicality_table():
    """Test that the physicality table is correctly computed."""
    print("Testing physicality table construction...")
    
    # Create a simple binary Hilbert space
    hilbert = nk.hilbert.Spin(s=1/2, N=4)  # 4 sites, binary (maps to 0,1)
    
    # Test with target_sum = 0 (2 up, 2 down spins -> sum = 0)
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=0.0)
    
    # Check some known values
    phys_table = sampler.phys_table
    print(f"Physicality table shape: {phys_table.shape}")
    print(f"Physicality table:\n{phys_table}")
    
    # For spin-1/2 with 4 sites and target_sum=0:
    # Min sum = -4 (all down), Max sum = +4 (all up)
    # Sum range: [-4, -3, -2, -1, 0, 1, 2, 3, 4] -> indices [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # target_sum=0 -> index 4
    
    # At site 0 with sum=-4, should be reachable (can reach 0 in 4 steps)
    assert phys_table[0, 0] == True, "Should be able to reach target from minimum sum"
    
    # At site 3 with sum=-4, should not be reachable (need +4 in 1 site, impossible)
    assert phys_table[3, 0] == False, "Should not be able to reach target from extreme sum in 1 step"
    
    # At the end (site 4) with exactly target sum, should be reachable
    assert phys_table[4, 4] == True, "Should accept exact target at the end"
    
    print("✓ Physicality table tests passed")


def test_constraint_enforcement():
    """Test that the sampler enforces particle number conservation."""
    print("\nTesting constraint enforcement...")
    
    # Create test system
    hilbert = nk.hilbert.Spin(s=1/2, N=6)
    target_sum = 0.0  # 3 up, 3 down -> sum = 0
    
    # Create a simple autoregressive model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Create symmetric sampler
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_sum)
    
    # Initialize model parameters
    key = jax.random.PRNGKey(1234)
    variables = model.init(key, jnp.zeros((1, hilbert.size)))
    
    # Generate samples
    samples, _ = sampler.sample(model, variables, chain_length=100)
    
    print(f"Generated {samples.shape} samples")
    
    # Check that all samples have exactly target_sum
    sample_sums = jnp.sum(samples, axis=-1)
    
    print(f"Sample sums: {jnp.unique(sample_sums)}")
    print(f"Target sum: {target_sum}")
    
    # All samples should have exactly target_sum
    assert jnp.all(sample_sums == target_sum), \
        f"Constraint violation: found sums {jnp.unique(sample_sums)}"
    
    print("✓ Constraint enforcement tests passed")


def test_sampling_distribution():
    """Test that the sampler produces reasonable distributions."""
    print("\nTesting sampling distribution...")
    
    # Create small system for exact verification
    hilbert = nk.hilbert.Spin(s=1/2, N=4)
    target_sum = 0.0  # 2 up, 2 down -> sum = 0
    
    # Use a simple uniform model (all configurations equally likely)
    class UniformModel:
        def __init__(self, hilbert):
            self.hilbert = hilbert
            
        def init(self, key, sample, index=None, method=None):
            return {}
        
        def apply(self, variables, sample, index=None, method=None, mutable=None):
            if method is None:
                # Full evaluation - return zero (uniform log probability)
                return jnp.zeros(sample.shape[0])
            elif method.__name__ == 'conditional':
                # Return uniform probabilities for conditional
                batch_size = sample.shape[0]
                n_local = len(self.hilbert.local_states)
                uniform_probs = jnp.ones((batch_size, n_local)) / n_local
                return uniform_probs, {}
            elif method.__name__ == 'reorder':
                # Return identity ordering
                return sample
        
        def conditional(self, sample, index):
            pass  # Placeholder
        
        def reorder(self, indices):
            pass  # Placeholder
    
    model = UniformModel(hilbert)
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_sum)
    
    # Generate many samples
    key = jax.random.PRNGKey(5678)
    state = sampler.init_state(model, {}, key)
    samples, _ = sampler.sample(model, {}, state=state, chain_length=1000)
    
    # Count configurations and their sums
    # All valid configurations with exactly target_sum=0 in 4 sites
    # There are C(4,2) = 6 configurations with 2 up and 2 down spins
    configs, counts = jnp.unique(samples.reshape(-1, 4), 
                                axis=0, return_counts=True, size=6)
    
    print(f"Found {len(configs)} unique configurations:")
    for i, (config, count) in enumerate(zip(configs, counts)):
        config_sum = jnp.sum(config)
        print(f"  {config}: {count} samples, sum = {config_sum}")
    
    # Check that all configurations have exactly target_sum
    config_sums = jnp.sum(configs, axis=1)
    assert jnp.all(config_sums == target_sum), \
        "Some configurations don't have the right sum"
    
    print("✓ Sampling distribution tests passed")


def test_comparison_with_rejection_sampling():
    """Compare efficiency with naive rejection sampling."""
    print("\nTesting efficiency comparison...")
    
    hilbert = nk.hilbert.Spin(s=1/2, N=8)
    target_sum = 2.0  # All up spins -> sum = 2
    
    # Create model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=16)
    key = jax.random.PRNGKey(9999)
    variables = model.init(key, jnp.zeros((1, hilbert.size)))
    
    # Our symmetric sampler
    sampler_sym = ARDirectSamplerSymmetric(hilbert, target_sum=target_sum)
    
    # Regular sampler for comparison
    sampler_reg = nk.sampler.ARDirectSampler(hilbert)
    
    # Time both approaches
    import time
    n_samples = 100
    
    # Time symmetric sampler
    start = time.time()
    samples_sym, _ = sampler_sym.sample(model, variables, chain_length=n_samples)
    time_sym = time.time() - start
    
    # Time regular sampler with rejection
    start = time.time()
    samples_reg = []
    attempts = 0
    while len(samples_reg) < n_samples:
        batch, _ = sampler_reg.sample(model, variables, chain_length=n_samples)
        sample_sums = jnp.sum(batch, axis=-1)  # Shape: (n_batches, n_samples)
        valid_mask = sample_sums == target_sum
        # Reshape batch to (n_batches * n_samples, n_sites) and then apply mask
        batch_flat = batch.reshape(-1, batch.shape[-1])
        valid_samples = batch_flat[valid_mask.reshape(-1)]
        samples_reg.extend(valid_samples)
        attempts += n_samples
    
    samples_reg = jnp.array(samples_reg[:n_samples])
    time_reg = time.time() - start
    
    print(f"Symmetric sampler: {time_sym:.4f}s for {n_samples} samples")
    print(f"Rejection sampling: {time_reg:.4f}s for {n_samples} samples ({attempts} attempts)")
    print(f"Speedup factor: {time_reg/time_sym:.2f}x")
    
    # Verify both produce valid samples
    assert jnp.all(jnp.sum(samples_sym, axis=-1) == target_sum)
    assert jnp.all(jnp.sum(samples_reg, axis=-1) == target_sum)
    
    print("✓ Efficiency comparison completed")


def main():
    """Run all tests."""
    print("Testing ARDirectSamplerSymmetric implementation")
    print("=" * 50)
    
    try:
        test_physicality_table()
        test_constraint_enforcement()
        test_sampling_distribution()
        test_comparison_with_rejection_sampling()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()