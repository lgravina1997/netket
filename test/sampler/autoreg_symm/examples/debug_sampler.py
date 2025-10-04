#!/usr/bin/env python3
"""
Debug the constraint sampler.
"""

import jax
import jax.numpy as jnp
import netket as nk
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric

def debug_simple():
    """Debug with the simplest case."""
    print("Debugging constraint sampler...")
    
    # 4 spins, Sz = 0 (like the notebook)
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    print(f"Hilbert: {L} spins")
    print(f"Local states: {hilbert.local_states}")
    print(f"Target: Sz = {target_magnetization}")
    
    # Create sampler
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    print(f"Physicality table shape: {sampler.phys_table.shape}")
    print(f"Physicality table:")
    print(sampler.phys_table)
    
    # Create simple model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=1, features=4)
    
    # Sample once  
    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.zeros((1, L)))
    
    print(f"Direct sampling with chain_length=20:")
    samples, _ = sampler.sample(model, variables, chain_length=20)
    
    print(f"Samples shape: {samples.shape}")
    
    # Check constraint
    samples_flat = samples.reshape(-1, L)
    magnetizations = jnp.sum(samples_flat, axis=1)
    
    print(f"First 10 samples:")
    for i in range(min(10, len(samples_flat))):
        config = samples_flat[i]
        mag = magnetizations[i]
        config_str = ''.join(['+' if s == 1 else '-' for s in config])
        print(f"  {config_str} -> Sz = {mag}")
    
    print(f"Unique magnetizations: {jnp.unique(magnetizations)}")
    print(f"Target satisfied: {jnp.all(magnetizations == target_magnetization)}")
    
    # Test with larger chain_length
    print(f"\nDirect sampling with chain_length=64:")
    samples_64, _ = sampler.sample(model, variables, chain_length=64)
    samples_64_flat = samples_64.reshape(-1, L)
    magnetizations_64 = jnp.sum(samples_64_flat, axis=1)
    print(f"Large samples shape: {samples_64.shape}")
    print(f"Large unique magnetizations: {jnp.unique(magnetizations_64)}")
    print(f"Large target satisfied: {jnp.all(magnetizations_64 == target_magnetization)}")
    
    # Now test with MCState like in the example
    print("\n" + "="*50)
    print("Testing with MCState (like the example)...")
    
    print(f"Sampler n_batches before MCState: {sampler.n_batches}")
    
    vs = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=64,  # Multiple samples in batch 
        seed=1234
    )
    
    print(f"Sampler n_batches after MCState: {sampler.n_batches}")
    print(f"VS sampler n_batches: {vs.sampler.n_batches}")
    
    print(f"Original sampler n_batches: {sampler.n_batches}")
    print(f"MCState n_samples: {vs.n_samples}")
    print(f"MCState sampler info: n_batches={vs.sampler.n_batches}, n_samples_per_rank={vs.n_samples_per_rank}")
    print(f"Same sampler object? {sampler is vs.sampler}")
    
    # Let's try calling the sampler directly with the MCState's configuration
    print(f"\nTesting direct call with MCState sampler...")
    vs_sampler = vs.sampler
    
    # Use the same variables that MCState uses
    vs_variables = vs.variables
    print(f"Original variables keys: {list(variables.keys()) if hasattr(variables, 'keys') else 'Not a dict'}")
    print(f"MCState variables keys: {list(vs_variables.keys()) if hasattr(vs_variables, 'keys') else 'Not a dict'}")
    
    # Use MCState's variables instead
    direct_samples_mcstate, _ = vs_sampler.sample(model, vs_variables, chain_length=64)
    print(f"Direct MCState sampler shape: {direct_samples_mcstate.shape}")
    direct_mag_mcstate = jnp.sum(direct_samples_mcstate.reshape(-1, L), axis=1)
    print(f"Direct MCState unique magnetizations: {jnp.unique(direct_mag_mcstate)}")
    print(f"Direct MCState constraint satisfied: {jnp.all(direct_mag_mcstate == target_magnetization)}")
    
    # Check the samples from MCState
    print(f"\nChecking MCState cached samples:")
    mcstate_samples = vs.samples
    print(f"MCState samples shape: {mcstate_samples.shape}")
    print(f"MCState samples dtype: {mcstate_samples.dtype}")
    print(f"Sample of first few MCState samples:")
    for i in range(min(5, mcstate_samples.shape[0])):
        sample = mcstate_samples[i]
        sample_sum = jnp.sum(sample)
        print(f"  Sample {i}: {sample} -> sum = {sample_sum}")
    
    # Sum over the sites axis (axis=-1 or axis=1)
    mcstate_magnetizations = jnp.sum(mcstate_samples, axis=-1)  # Sum over sites
    print(f"MCState magnetizations (axis=-1): {mcstate_magnetizations}")
    print(f"MCState magnetizations range: [{jnp.min(mcstate_magnetizations)}, {jnp.max(mcstate_magnetizations)}]")
    print(f"MCState unique magnetizations: {jnp.unique(mcstate_magnetizations)}")
    print(f"MCState constraint satisfied: {jnp.all(mcstate_magnetizations == target_magnetization)}")
    
    # Let's force a resample
    print(f"\nForcing MCState to regenerate samples...")
    vs.reset()  # This should clear any cached samples
    new_mcstate_samples = vs.samples
    print(f"New MCState samples shape: {new_mcstate_samples.shape}")
    
    new_mcstate_magnetizations = jnp.sum(new_mcstate_samples, axis=1)
    print(f"New MCState magnetizations range: [{jnp.min(new_mcstate_magnetizations)}, {jnp.max(new_mcstate_magnetizations)}]")
    print(f"New MCState unique magnetizations: {jnp.unique(new_mcstate_magnetizations)}")
    print(f"New MCState constraint satisfied: {jnp.all(new_mcstate_magnetizations == target_magnetization)}")

if __name__ == "__main__":
    debug_simple()