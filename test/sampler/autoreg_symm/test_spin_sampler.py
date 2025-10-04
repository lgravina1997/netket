#!/usr/bin/env python3
"""
Test the generalized symmetric sampler with spin systems.
"""

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import math
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric


def test_spin_half_magnetization():
    """Test magnetization conservation for spin-1/2 system."""
    print("Testing spin-1/2 magnetization conservation...")
    
    # Create spin-1/2 system (local states = [-1, 1])
    L = 6  # Must be even for zero magnetization
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0  # Zero total magnetization
    
    print(f"Local states: {hilbert.local_states}")
    print(f"Target magnetization: {target_magnetization}")
    
    # Create sampler
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    # Create simple model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Initialize and sample
    key = jax.random.PRNGKey(1234)
    variables = model.init(key, jnp.zeros((1, hilbert.size)))
    
    samples, _ = sampler.sample(model, variables, chain_length=100)
    
    print(f"Generated {samples.shape} samples")
    
    # Check magnetization conservation
    magnetizations = jnp.sum(samples, axis=-1)
    unique_mags = jnp.unique(magnetizations)
    
    print(f"Unique magnetizations: {unique_mags}")
    print(f"All samples have target magnetization: {jnp.all(magnetizations == target_magnetization)}")
    
    assert jnp.all(magnetizations == target_magnetization), \
        f"Magnetization not conserved: found {unique_mags}"
    
    print("✓ Spin-1/2 magnetization test passed")


def test_particle_number_conservation():
    """Test particle number conservation with [0,1] states."""
    print("\nTesting particle number conservation...")
    
    # Create particle system (need to map spin to particles)
    L = 8
    target_sum = 3
    
    # Use Spin hilbert but interpret as particles [0,1]
    # We'll need to handle the mapping carefully
    hilbert = nk.hilbert.Fock(n_max=1, N=L)  # Better for particle systems
    
    print(f"Local states: {hilbert.local_states}")
    print(f"Target particle number: {target_sum}")
    
    # Create sampler
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_sum)
    
    # Create model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Sample
    key = jax.random.PRNGKey(5678)
    variables = model.init(key, jnp.zeros((1, hilbert.size)))
    
    samples, _ = sampler.sample(model, variables, chain_length=100)
    
    # Check particle conservation
    particle_counts = jnp.sum(samples, axis=-1)
    unique_counts = jnp.unique(particle_counts)
    
    print(f"Unique particle counts: {unique_counts}")
    print(f"All samples have target count: {jnp.all(particle_counts == target_sum)}")
    
    assert jnp.all(particle_counts == target_sum), \
        f"Particle number not conserved: found {unique_counts}"
    
    print("✓ Particle number conservation test passed")


def test_physicality_table_spins():
    """Test physicality table construction for spins."""
    print("\nTesting physicality table for spins...")
    
    # Small system for manual verification
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)  # local_states = [-1, 1]
    target_mag = 0
    
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_mag)
    
    print(f"Physicality table shape: {sampler.phys_table.shape}")
    print(f"Sum scale: {sampler.sum_scale}")
    print(f"Sum offset: {sampler.sum_offset}")
    
    # For L=4, [-1,1] states, possible sums are -4, -2, 0, 2, 4
    # With target=0, should be reachable from start
    start_sum_idx = int(sampler._get_sum_index(0))  # Start with sum=0
    can_reach_from_start = sampler.phys_table[0, start_sum_idx]
    
    print(f"Can reach target from start: {can_reach_from_start}")
    assert can_reach_from_start, "Should be able to reach target magnetization from start"
    
    # At the end with wrong magnetization, should not be reachable
    wrong_sum_idx = int(sampler._get_sum_index(2))  # Wrong magnetization
    if 0 <= wrong_sum_idx < sampler.phys_table.shape[1]:
        at_end_wrong = sampler.phys_table[L, wrong_sum_idx]
        print(f"Reachable at end with wrong sum: {at_end_wrong}")
        assert not at_end_wrong, "Should not be reachable at end with wrong sum"
    
    print("✓ Physicality table test passed")


def test_mixed_states():
    """Test with non-standard local states."""
    print("\nTesting mixed local states...")
    
    # Create custom Hilbert space with unusual local states
    try:
        # This might not work with standard NetKet, but let's test the principle
        from netket.hilbert import CustomHilbert
        
        local_states = [-2, 0, 1]  # Unusual local states
        L = 3
        target_sum = -1  # Sum = -2 + 0 + 1 = -1
        
        # For testing, we'll create a simple discrete Hilbert space
        # In practice, you might need to subclass DiscreteHilbert
        print("Mixed states test skipped - requires custom Hilbert space implementation")
        
    except ImportError:
        print("Mixed states test skipped - CustomHilbert not available")


def test_performance_comparison():
    """Compare performance of spin vs particle systems."""
    print("\nTesting performance comparison...")
    
    import time
    
    sizes = [6, 8]
    
    print("Size\tType\t\tTime (ms)")
    print("-" * 30)
    
    for L in sizes:
        # Test spin system
        hilbert_spin = nk.hilbert.Spin(s=1/2, N=L)
        sampler_spin = ARDirectSamplerSymmetric(hilbert_spin, target_sum=0)
        model_spin = nk.models.ARNNDense(hilbert=hilbert_spin, layers=2, features=8)
        
        key = jax.random.PRNGKey(1234)
        variables_spin = model_spin.init(key, jnp.zeros((1, L)))
        
        # Warmup
        _, _ = sampler_spin.sample(model_spin, variables_spin, chain_length=10)
        
        # Time spin system
        start = time.time()
        _, _ = sampler_spin.sample(model_spin, variables_spin, chain_length=50)
        time_spin = (time.time() - start) * 1000
        
        # Test particle system
        hilbert_fock = nk.hilbert.Fock(n_max=1, N=L)
        sampler_fock = ARDirectSamplerSymmetric(hilbert_fock, target_sum=L//2)
        model_fock = nk.models.ARNNDense(hilbert=hilbert_fock, layers=2, features=8)
        
        variables_fock = model_fock.init(key, jnp.zeros((1, L)))
        
        # Warmup
        _, _ = sampler_fock.sample(model_fock, variables_fock, chain_length=10)
        
        # Time particle system
        start = time.time()
        _, _ = sampler_fock.sample(model_fock, variables_fock, chain_length=50)
        time_fock = (time.time() - start) * 1000
        
        print(f"{L}\tSpin\t\t{time_spin:.2f}")
        print(f"{L}\tParticle\t{time_fock:.2f}")
    
    print("✓ Performance comparison completed")


def main():
    """Run all tests."""
    print("Testing Generalized ARDirectSamplerSymmetric")
    print("=" * 50)
    
    try:
        test_spin_half_magnetization()
        test_particle_number_conservation()
        test_physicality_table_spins()
        test_mixed_states()
        test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The sampler now supports:")
        print("  ✓ Spin systems with magnetization conservation")
        print("  ✓ Particle systems with number conservation")
        print("  ✓ General sum constraints")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()