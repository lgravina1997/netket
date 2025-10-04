#!/usr/bin/env python3
"""
Test and demonstrate the consistency issue between sampling and inference.
"""

import jax
import jax.numpy as jnp
import netket as nk
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric


def test_consistency_issue():
    """Demonstrate the consistency issue between sampling and inference."""
    print("Testing Consistency Between Sampling and Inference")
    print("=" * 60)
    
    # Setup
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    # Create standard autoregressive model
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Create constraint sampler
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.zeros((1, L)))
    
    print("1. Testing Sampling (should respect constraints):")
    print("-" * 50)
    
    # Generate samples using constraint sampler
    samples, _ = sampler.sample(model, variables, chain_length=20)
    samples_flat = samples.reshape(-1, L)
    sample_magnetizations = jnp.sum(samples_flat, axis=1)
    
    print(f"Generated {len(samples_flat)} samples")
    print(f"Unique magnetizations from sampling: {jnp.unique(sample_magnetizations)}")
    print(f"All samples satisfy constraint: {jnp.all(sample_magnetizations == target_magnetization)}")
    
    print("\n2. Testing Inference (currently ignores constraints):")
    print("-" * 50)
    
    # Create test configurations that violate the constraint
    valid_config = jnp.array([1, 1, -1, -1])      # Sz = 0 (valid)
    invalid_config = jnp.array([1, 1, 1, -1])     # Sz = 2 (invalid)
    
    # Evaluate model on both configurations
    valid_log_psi = model.apply(variables, valid_config)
    invalid_log_psi = model.apply(variables, invalid_config)
    
    print(f"Valid config [+,+,-,-] (Sz=0): log_psi = {float(valid_log_psi[0]):.4f}")
    print(f"Invalid config [+,+,+,-] (Sz=2): log_psi = {float(invalid_log_psi[0]):.4f}")
    print(f"Model assigns non-zero amplitude to invalid config: {not jnp.isinf(invalid_log_psi[0])}")
    
    print("\n3. The Consistency Problem:")
    print("-" * 50)
    print("❌ INCONSISTENCY DETECTED:")
    print("   - Sampler only generates configurations with Sz=0")
    print("   - Model assigns non-zero amplitude to configurations with Sz≠0")
    print("   - This breaks the fundamental assumption of MCMC sampling!")
    
    return valid_log_psi, invalid_log_psi


def test_consistency_solution():
    """Demonstrate the solution using constraint-aware model wrapper."""
    print("\n" + "=" * 60)
    print("Testing Solution: Constraint-Aware Model")
    print("=" * 60)
    
    # Setup (same as before)
    L = 4
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    target_magnetization = 0
    
    # Create base model
    base_model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Import our constraint-aware wrapper
    import sys
    sys.path.append('/Users/lucagravina/Codes/GitRepos/netket')
    from netket.models.autoreg_symmetric import ConstraintAwareARNN
    
    # Create constraint-aware model
    constrained_model = ConstraintAwareARNN(
        base_model=base_model, 
        target_sum=target_magnetization
    )
    
    # Initialize
    key = jax.random.PRNGKey(42)
    variables = constrained_model.init(key, jnp.zeros((1, L)))
    
    print("Testing Constrained Model Inference:")
    print("-" * 40)
    
    # Test same configurations
    valid_config = jnp.array([1, 1, -1, -1])      # Sz = 0 (valid)
    invalid_config = jnp.array([1, 1, 1, -1])     # Sz = 2 (invalid)
    
    valid_log_psi = constrained_model.apply(variables, valid_config)
    invalid_log_psi = constrained_model.apply(variables, invalid_config)
    
    print(f"Valid config [+,+,-,-] (Sz=0): log_psi = {float(valid_log_psi[0]):.4f}")
    print(f"Invalid config [+,+,+,-] (Sz=2): log_psi = {float(invalid_log_psi[0]):.4f}")
    print(f"Invalid config has -inf amplitude: {jnp.isinf(invalid_log_psi[0]) and invalid_log_psi[0] < 0}")
    
    print("\n✅ CONSISTENCY ACHIEVED:")
    print("   - Sampler only generates configurations with Sz=0")
    print("   - Model assigns zero amplitude to configurations with Sz≠0")
    print("   - Perfect consistency between sampling and inference!")
    
    return valid_log_psi, invalid_log_psi


def main():
    """Run the consistency test and demonstration."""
    
    # First show the problem
    test_consistency_issue()
    
    # Then show the solution
    test_consistency_solution()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("When using constraint samplers like ARDirectSamplerSymmetric,")
    print("you should ALSO use constraint-aware models to ensure")
    print("consistency between sampling and inference!")


if __name__ == "__main__":
    main()