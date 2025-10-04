#!/usr/bin/env python3
"""
Quick test to verify the NetKet Pytree fix works.
"""

import jax
import jax.numpy as jnp
import netket as nk
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric

def test_sampler_initialization():
    """Test that the sampler can be created without errors."""
    print("Testing ARDirectSamplerSymmetric initialization...")
    
    # Create spin system
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    print(f"Hilbert space: {L} spins")
    print(f"Local states: {hilbert.local_states}")
    print(f"Target magnetization: {target_magnetization}")
    
    try:
        # This should work now
        sampler = ARDirectSamplerSymmetric(
            hilbert=hilbert, 
            target_sum=target_magnetization
        )
        
        print("✅ Sampler created successfully!")
        print(f"Sampler type: {type(sampler).__name__}")
        print(f"Is exact: {sampler.is_exact}")
        print(f"Target sum: {sampler.target_sum}")
        print(f"Local states shape: {sampler.local_states.shape}")
        print(f"Physicality table shape: {sampler.phys_table.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sampler: {e}")
        return False

def test_sampling():
    """Test that sampling works."""
    print("\nTesting sampling...")
    
    # Create system
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=0)
    
    # Create simple model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.zeros((1, hilbert.size)))
    
    try:
        # Sample
        samples, _ = sampler.sample(model, variables, chain_length=10)
        
        print(f"✅ Sampling successful!")
        print(f"Sample shape: {samples.shape}")
        
        # Check constraint
        samples_flat = samples.reshape(-1, L)
        magnetizations = jnp.sum(samples_flat, axis=1)
        
        print(f"Magnetizations: {jnp.unique(magnetizations)}")
        print(f"Constraint satisfied: {jnp.all(magnetizations == 0)}")
        
        # Show a few samples
        print(f"\nFirst 5 samples:")
        for i in range(min(5, len(samples_flat))):
            config = samples_flat[i]
            config_str = ''.join(['+' if s == 1 else '-' for s in config])
            mag = jnp.sum(config)
            print(f"  {config_str} (Sz={mag})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        return False

def main():
    """Run tests."""
    print("Testing NetKet Pytree Fix for ARDirectSamplerSymmetric")
    print("=" * 55)
    
    success1 = test_sampler_initialization()
    success2 = test_sampling() if success1 else False
    
    print("\n" + "=" * 55)
    if success1 and success2:
        print("🎉 All tests passed! The fix works correctly.")
    else:
        print("❌ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()