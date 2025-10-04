#!/usr/bin/env python3
"""
Test MCState reset behavior.
"""

import jax
import jax.numpy as jnp
import netket as nk
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric

def test_mcstate_reset():
    """Test if MCState reset affects constraint satisfaction."""
    print("Testing MCState reset behavior...")
    
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    model = nk.models.ARNNDense(hilbert=hilbert, layers=1, features=4)
    
    vs = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=32,
        seed=1234
    )
    
    for trial in range(3):
        print(f"\nTrial {trial + 1}:")
        samples = vs.samples
        magnetizations = jnp.sum(samples, axis=-1).flatten()
        print(f"  Shape: {samples.shape}")
        print(f"  Unique magnetizations: {jnp.unique(magnetizations)}")
        print(f"  Constraint satisfied: {jnp.all(magnetizations == target_magnetization)}")
        
        # Reset for next trial
        if trial < 2:  # Don't reset after last trial
            vs.reset()

if __name__ == "__main__":
    test_mcstate_reset()