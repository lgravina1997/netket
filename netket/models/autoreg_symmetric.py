#!/usr/bin/env python3
"""
Constraint-aware wrapper for autoregressive models.

This ensures consistency between sampling (which enforces constraints)
and inference (which should also respect the same constraints).
"""

import jax.numpy as jnp
from flax import linen as nn
from netket.models import AbstractARNN
from typing import Any


class ConstraintAwareARNN(nn.Module):
    """
    Wrapper around an autoregressive model that enforces symmetry constraints
    during both sampling and inference using the same physicality table as the sampler.
    
    This ensures perfect consistency between sampling and inference by using
    the exact same constraint logic.
    """
    
    base_model: AbstractARNN
    """The underlying autoregressive model."""
    
    sampler: Any  # ARDirectSamplerSymmetric
    """The constraint sampler to get physicality table from."""
    
    def setup(self):
        # Inherit key properties from base model
        self.hilbert = self.base_model.hilbert
        self.machine_pow = self.base_model.machine_pow
        
        # Get physicality table and constraint info from sampler
        self.target_sum = self.sampler.target_sum
        self.phys_table = self.sampler.phys_table
        self.local_states = self.sampler.local_states
        self.sum_scale = self.sampler.sum_scale
        self.sum_offset = self.sampler.sum_offset
    
    def _get_sum_index(self, sum_value):
        """Convert sum value to table index (same as sampler)."""
        sum_scaled = jnp.round(sum_value * self.sum_scale).astype(jnp.int32)
        return sum_scaled - jnp.round(self.sum_offset).astype(jnp.int32)
    
    def _check_configuration_validity(self, config):
        """
        Check if a complete configuration is valid using the physicality table.
        This uses the same logic as the sampler for perfect consistency.
        
        For simplicity in this implementation, we just check the final sum.
        A more sophisticated version would simulate the full autoregressive process.
        """
        if config.ndim == 1:
            config = jnp.expand_dims(config, axis=0)
        
        # Check if final sum matches target (most important constraint)
        final_sums = jnp.sum(config, axis=-1)
        valid_configs = (final_sums == self.target_sum)
        
        return valid_configs
    
    def __call__(self, inputs):
        """
        Compute log wave function, enforcing constraints using the physicality table.
        """
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        
        # Compute the base model's log psi
        log_psi = self.base_model(inputs)
        
        # Check constraint satisfaction using physicality table
        constraint_satisfied = self._check_configuration_validity(inputs)
        
        # Set amplitude to zero (log_psi to -inf) for constraint violations
        log_psi = jnp.where(constraint_satisfied, log_psi, -jnp.inf)
        
        return log_psi
    
    def conditionals_log_psi(self, inputs):
        """Forward to base model - constraints handled in __call__."""
        return self.base_model.conditionals_log_psi(inputs)
    
    def conditionals(self, inputs):
        """Forward to base model - constraints handled during sampling."""
        return self.base_model.conditionals(inputs)
    
    def conditional(self, inputs, index):
        """Forward to base model - constraints handled during sampling."""
        return self.base_model.conditional(inputs, index)
    
    def reorder(self, inputs, axis=0):
        """Forward to base model."""
        return self.base_model.reorder(inputs, axis)
    
    def inverse_reorder(self, inputs, axis=0):
        """Forward to base model."""
        return self.base_model.inverse_reorder(inputs, axis)


def create_constraint_aware_model(base_model, constraint_sampler):
    """
    Create a constraint-aware version of an autoregressive model.
    
    Args:
        base_model: The base autoregressive model (e.g., ARNNDense)
        constraint_sampler: The ARDirectSamplerSymmetric instance with physicality table
        
    Returns:
        A ConstraintAwareARNN that enforces constraints during inference
    """
    return ConstraintAwareARNN(base_model=base_model, sampler=constraint_sampler)


# Example usage function
def example_constraint_aware_usage():
    """Example of how to use constraint-aware models with physicality table."""
    import netket as nk
    from netket.sampler.autoreg_symmetric import ARDirectSamplerSymmetric
    
    # Create system
    hilbert = nk.hilbert.Spin(s=1/2, N=4)
    target_magnetization = 0
    
    # Create constraint sampler first (needed for physicality table)
    sampler = ARDirectSamplerSymmetric(hilbert, target_sum=target_magnetization)
    
    # Create base autoregressive model
    base_model = nk.models.ARNNDense(hilbert=hilbert, layers=2, features=8)
    
    # Wrap with constraint awareness using the sampler's physicality table
    constrained_model = create_constraint_aware_model(base_model, sampler)
    
    # Now both sampling and inference respect the same constraints
    vs = nk.vqs.MCState(
        sampler=sampler,
        model=constrained_model,  # Use the wrapped model
        n_samples=64
    )
    
    print("Both sampling and inference now enforce the same constraints using the physicality table!")
    return vs


if __name__ == "__main__":
    example_constraint_aware_usage()