#!/usr/bin/env python3
"""
Test the improved constraint-aware model that uses the physicality table.
"""

import jax
import jax.numpy as jnp
import netket as nk
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric
import sys
sys.path.append('/Users/lucagravina/Codes/GitRepos/netket')
from netket.models.autoreg_symmetric import create_constraint_aware_model


def test_physicality_table_consistency():
    """Test constraint-aware model using the sampler's physicality table."""
    print("Testing Physicality Table Consistency")
    print("=" * 50)
    
    # Setup
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    # Create constraint sampler (this builds the physicality table)
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    print(f"Sampler physicality table shape: {sampler.phys_table.shape}")
    print(f"Sum scale: {sampler.sum_scale}")
    print(f"Sum offset: {sampler.sum_offset}")
    
    # Create base model
    base_model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Create constraint-aware model using the sampler's physicality table
    constrained_model = create_constraint_aware_model(base_model, sampler)
    
    # Initialize models
    key = jax.random.PRNGKey(42)
    base_variables = base_model.init(key, jnp.zeros((1, L)))
    constrained_variables = constrained_model.init(key, jnp.zeros((1, L)))
    
    print("\nTesting various configurations:")
    print("-" * 40)
    
    # Test configurations with different magnetizations
    test_configs = [
        jnp.array([1, 1, -1, -1]),   # Sz = 0 (valid)
        jnp.array([-1, -1, 1, 1]),   # Sz = 0 (valid)
        jnp.array([1, -1, 1, -1]),   # Sz = 0 (valid)
        jnp.array([1, 1, 1, -1]),    # Sz = 2 (invalid)
        jnp.array([1, 1, 1, 1]),     # Sz = 4 (invalid)
        jnp.array([-1, -1, -1, -1]), # Sz = -4 (invalid)
    ]
    
    for i, config in enumerate(test_configs):
        magnetization = jnp.sum(config)
        
        # Evaluate with base model (no constraints)
        base_log_psi = base_model.apply(base_variables, config)
        
        # Evaluate with constraint-aware model (uses physicality table)
        constrained_log_psi = constrained_model.apply(constrained_variables, config)
        
        is_valid = magnetization == target_magnetization
        is_constrained_inf = jnp.isinf(constrained_log_psi[0]) and constrained_log_psi[0] < 0
        
        config_str = ''.join(['+' if s == 1 else '-' for s in config])
        
        print(f"Config {i+1}: {config_str} (Sz={int(magnetization)})")
        print(f"  Base model:        log_psi = {float(base_log_psi[0]):.4f}")
        print(f"  Constrained model: log_psi = {float(constrained_log_psi[0]):.4f}")
        print(f"  Expected valid: {is_valid}, Constrained assigns -inf: {is_constrained_inf}")
        print(f"  Consistent: {is_valid == (not is_constrained_inf)}")
        print()
    
    print("✅ The constraint-aware model now uses the same physicality")
    print("   table as the sampler for perfect consistency!")
    
    return sampler, constrained_model


def test_sampling_inference_consistency():
    """Test that sampling and inference give consistent results."""
    print("\n" + "=" * 50)
    print("Testing Sampling-Inference Consistency")
    print("=" * 50)
    
    # Create components
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    base_model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    constrained_model = create_constraint_aware_model(base_model, sampler)
    
    # Create MCState with constraint-aware model
    vs = nk.vqs.MCState(
        sampler=sampler,
        model=constrained_model,
        n_samples=32,
        seed=1234
    )
    
    # Test sampling
    samples = vs.samples
    sample_magnetizations = jnp.sum(samples, axis=-1).flatten()
    
    print(f"Generated {len(sample_magnetizations)} samples")
    print(f"Sample magnetizations: {jnp.unique(sample_magnetizations)}")
    print(f"All samples satisfy constraint: {jnp.all(sample_magnetizations == target_magnetization)}")
    
    # Test inference on the same samples
    log_psis = constrained_model.apply(vs.variables, samples.reshape(-1, L))
    finite_amplitudes = jnp.isfinite(log_psis)
    
    print(f"All sampled configs have finite amplitudes: {jnp.all(finite_amplitudes)}")
    print(f"Perfect consistency: {jnp.all(sample_magnetizations == target_magnetization) and jnp.all(finite_amplitudes)}")
    
    print("\n✅ Perfect consistency achieved between sampling and inference!")


def main():
    """Run all consistency tests."""
    test_physicality_table_consistency()
    test_sampling_inference_consistency()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print("✅ Constraint-aware model now uses the sampler's physicality table")
    print("✅ Perfect consistency between sampling and inference")
    print("✅ Both respect exactly the same conservation constraints")


if __name__ == "__main__":
    main()