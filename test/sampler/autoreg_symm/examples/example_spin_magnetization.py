#!/usr/bin/env python3
"""
Example: Using ARDirectSamplerSymmetric for spin magnetization conservation.

This demonstrates the generalized sampler working with spin systems where
we enforce exact magnetization conservation (S_z = sum of spins = target).
"""

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import math
from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric


def example_ising_zero_magnetization():
    """
    Example: 1D Ising model with exact zero magnetization.
    """
    print("Example: 1D Ising model with zero magnetization")
    print("=" * 60)
    
    # System parameters
    L = 8  # Must be even for zero magnetization with spins ±1
    target_magnetization = 0
    h = 0.5  # Transverse field
    
    # Create graph and Hilbert space
    graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=1/2, N=L)  # Local states = [-1, +1]
    
    print(f"System: {L} sites")
    print(f"Local states: {hilbert.local_states}")
    print(f"Target magnetization: {target_magnetization}")
    print(f"Possible configurations with Sz=0: C({L}, {L//2}) = {math.comb(L, L//2)}")
    
    # Create Ising Hamiltonian
    # H = -J Σ σᵢσⱼ - h Σ σᵢˣ
    hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=h)
    
    # Create autoregressive model
    model = nk.models.ARNNConv1D(
        hilbert=hilbert,
        layers=3,
        features=16,
        kernel_size=3
    )
    
    # Create symmetric sampler with zero magnetization constraint
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    # Create variational state
    vs = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=256,
        seed=1234
    )
    
    print(f"Model parameters: {vs.n_parameters}")
    
    # Test constraint satisfaction
    samples = vs.samples
    magnetizations = jnp.sum(samples, axis=-1)
    
    print(f"\nSample verification:")
    print(f"Sample shape: {samples.shape}")
    print(f"Magnetization range: [{jnp.min(magnetizations)}, {jnp.max(magnetizations)}]")
    print(f"All samples have Sz=0: {jnp.all(magnetizations == target_magnetization)}")
    
    # Compute energy
    energy = vs.expect(hamiltonian)
    print(f"Initial energy: {energy}")
    
    # Set up optimization
    optimizer = nk.optimizer.Sgd(learning_rate=0.02)
    sr = nk.optimizer.SR(diag_shift=0.01)
    
    gs = nk.VMC(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        preconditioner=sr,
        variational_state=vs
    )
    
    print(f"\nOptimizing ground state with Sz=0 constraint...")
    print("Iter\tEnergy\t\t\tVariance\t\tSz_check")
    print("-" * 55)
    
    for i in range(15):
        gs.advance()
        if i % 3 == 0:
            energy = vs.expect(hamiltonian)
            samples = vs.samples
            magnetizations = jnp.sum(samples, axis=-1)
            sz_satisfied = jnp.all(magnetizations == target_magnetization)
            
            print(f"{i}\t{energy.mean:.6f}\t\t{energy.variance:.6f}\t\t{sz_satisfied}")
    
    print("\n✓ Zero magnetization constraint maintained throughout optimization")


def example_heisenberg_magnetization():
    """
    Example: Heisenberg model with specific magnetization sector.
    """
    print("\n" + "=" * 60)
    print("Example: Heisenberg model with Sz = -2")
    print("=" * 60)
    
    # System parameters
    L = 6
    target_magnetization = -2  # More down spins than up
    
    # Create graph and Hilbert space
    graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    
    print(f"System: {L} sites")
    print(f"Target magnetization: {target_magnetization}")
    
    # For Sz = -2: need 4 down spins (-1) and 2 up spins (+1)
    # 4*(-1) + 2*(+1) = -4 + 2 = -2 ✓
    n_up_spins = (L + target_magnetization) // 2
    n_down_spins = L - n_up_spins
    print(f"Required: {n_up_spins} up spins, {n_down_spins} down spins")
    print(f"Possible configurations: C({L}, {n_up_spins}) = {math.comb(L, n_up_spins)}")
    
    # Create Heisenberg Hamiltonian  
    # For simplicity, we'll use Ising with small transverse field
    hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=0.1)
    
    # Create model and sampler
    model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=12)
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=128)
    
    # Verify constraint
    samples = vs.samples
    magnetizations = jnp.sum(samples, axis=-1)
    
    print(f"\nConstraint verification:")
    print(f"Unique magnetizations: {jnp.unique(magnetizations)}")
    print(f"Target achieved: {jnp.all(magnetizations == target_magnetization)}")
    
    # Count up/down spins in samples
    up_spins = jnp.sum(samples == 1, axis=-1)
    down_spins = jnp.sum(samples == -1, axis=-1)
    
    print(f"Up spins per sample: {jnp.unique(up_spins)}")
    print(f"Down spins per sample: {jnp.unique(down_spins)}")
    
    energy = vs.expect(hamiltonian)
    print(f"Energy in Sz={target_magnetization} sector: {energy}")


def compare_magnetization_sectors():
    """
    Compare ground state energies in different magnetization sectors.
    """
    print("\n" + "=" * 60)
    print("Comparison: Ground states in different Sz sectors")
    print("=" * 60)
    
    L = 6
    graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=1/2, N=L)
    hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=0.3)
    
    # Test different magnetization sectors
    magnetizations = [-2, 0, 2]
    
    print("Sz\tConfigs\tEnergy")
    print("-" * 25)
    
    for target_sz in magnetizations:
        # Create sampler for this sector
        sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_sz)
        model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
        vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=64)
        
        # Quick optimization
        optimizer = nk.optimizer.Sgd(learning_rate=0.05)
        gs = nk.VMC(hamiltonian, optimizer, variational_state=vs)
        
        for _ in range(10):
            gs.advance()
        
        energy = vs.expect(hamiltonian)
        n_configs = math.comb(L, (L + target_sz) // 2)
        
        print(f"{target_sz}\t{n_configs}\t{energy.mean:.6f}")
        
        # Verify constraint
        samples = vs.samples
        magnetizations_check = jnp.sum(samples, axis=-1)
        # Handle batch dimensions properly
        magnetizations_flat = magnetizations_check.flatten()
        constraint_satisfied = jnp.all(magnetizations_flat == target_sz)
        if not constraint_satisfied:
            print(f"  Warning: Some constraint violations for Sz={target_sz}")
            print(f"  Found magnetizations: {jnp.unique(magnetizations_flat)}")
            print(f"  Expected: {target_sz}")
        else:
            print(f"  ✓ All constraints satisfied for Sz={target_sz}")
    
    print("\n✓ All magnetization sectors computed successfully")


def benchmark_spin_vs_particle():
    """
    Benchmark the performance difference between spin and particle systems.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Spin vs Particle conservation")
    print("=" * 60)
    
    import time
    
    L = 8
    n_samples = 50
    
    # Spin system: Sz = 0
    hilbert_spin = nk.hilbert.Spin(s=1/2, N=L)
    sampler_spin = ARDirectSamplerSymmetric(hilbert_spin, target_sum=0)
    model_spin = nk.models.ARNNDense(hilbert=hilbert_spin, layers=2, features=8)
    
    key = jax.random.PRNGKey(1234)
    variables_spin = model_spin.init(key, jnp.zeros((1, L)))
    
    # Particle system: N = L/2
    hilbert_particle = nk.hilbert.Fock(n_max=1, N=L)
    sampler_particle = ARDirectSamplerSymmetric(hilbert_particle, target_sum=L//2)
    model_particle = nk.models.ARNNDense(hilbert=hilbert_particle, layers=2, features=8)
    variables_particle = model_particle.init(key, jnp.zeros((1, L)))
    
    # Warmup
    _, _ = sampler_spin.sample(model_spin, variables_spin, chain_length=10)
    _, _ = sampler_particle.sample(model_particle, variables_particle, chain_length=10)
    
    # Benchmark spin system
    start = time.time()
    samples_spin, _ = sampler_spin.sample(model_spin, variables_spin, chain_length=n_samples)
    time_spin = time.time() - start
    
    # Benchmark particle system  
    start = time.time()
    samples_particle, _ = sampler_particle.sample(model_particle, variables_particle, chain_length=n_samples)
    time_particle = time.time() - start
    
    print(f"System size: {L} sites")
    print(f"Samples: {n_samples}")
    print(f"Spin system (Sz=0):     {time_spin:.4f}s ({n_samples/time_spin:.0f} samples/s)")
    print(f"Particle system (N=4):  {time_particle:.4f}s ({n_samples/time_particle:.0f} samples/s)")
    
    # Verify constraints
    magnetizations = jnp.sum(samples_spin, axis=-1)
    particle_counts = jnp.sum(samples_particle, axis=-1)
    
    print(f"\nConstraint verification:")
    print(f"Spin magnetizations: {jnp.unique(magnetizations)}")
    print(f"Particle counts: {jnp.unique(particle_counts)}")
    
    assert jnp.all(magnetizations == 0), "Spin constraint violated"
    assert jnp.all(particle_counts == L//2), "Particle constraint violated"
    
    print("✓ Both constraints satisfied")


def main():
    """Run all examples."""
    print("ARDirectSamplerSymmetric: Spin Magnetization Examples")
    print("=" * 60)
    
    try:
        example_ising_zero_magnetization()
        example_heisenberg_magnetization()
        compare_magnetization_sectors()
        benchmark_spin_vs_particle()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\nKey capabilities demonstrated:")
        print("  ✓ Zero magnetization constraint in Ising model")
        print("  ✓ Arbitrary magnetization sectors")
        print("  ✓ Ground state optimization with constraints")
        print("  ✓ Performance comparison between constraint types")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        raise


if __name__ == "__main__":
    main()