# Copyright 2024 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import flax
import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket import config
from netket.hilbert import DiscreteHilbert
from netket.sampler.autoreg import ARDirectSampler
from netket.utils.types import DType
from netket.utils import struct

def normalize_probabilities(p, mask):
    """Apply constraint mask and renormalize probabilities."""
    p_masked = p * mask
    p_sum = jnp.sum(p_masked, axis=-1, keepdims=True)
    p_sum = jnp.maximum(p_sum, 1e-10)  # Avoid division by zero
    return p_masked / p_sum

def compute_constraint_mask(phys_table, local_states, sum_scale, sum_offset, index, current_sums):
    """Compute mask for valid next states given current sums."""
    # Vectorized computation of all possible next sums
    new_sums = current_sums[:, None] + local_states[None, :]
    new_indices = jnp.round(new_sums * sum_scale).astype(jnp.int32) - jnp.int32(sum_offset)
    
    # Check bounds and reachability
    in_bounds = (new_indices >= 0) & (new_indices < phys_table.shape[1])
    safe_indices = jnp.clip(new_indices, 0, phys_table.shape[1] - 1)
    reachable = phys_table[index + 1, safe_indices]

    return (in_bounds & reachable).astype(jnp.float32)

def init_cache(model, σ, key):
    """Initialize cache for autoregressive model."""
    variables = model.init(key, σ, 0, method=model.conditional)
    cache = variables.get("cache")
    return cache

# External JIT-compiled function for constraint sampling
@partial(jax.jit, static_argnames=(
    "model", 
    "chain_length",
    "return_log_probabilities", 
    "n_batches",
    "dtype",
))
def _sample_chain_with_constraints(
    model,
    variables,
    state,
    chain_length,
    return_log_probabilities,
    phys_table,
    sum_scale,
    sum_offset,
    local_states,
    hilbert_local_states,
    n_batches,
    dtype,
):
    """
    Pure JIT-compiled function for autoregressive sampling with constraints.
    
    This function is stateless and gets all constraint data as static arguments,
    eliminating the need for instance-specific compilation signatures.
    """
    if "cache" in variables:
        variables, _ = flax.core.pop(variables, "cache")
    variables_no_cache = variables

    def scan_fun(carry, index):
        σ, cache, key, log_prob, n_count = carry
        
        if cache:
            variables = {**variables_no_cache, "cache": cache}
        else:
            variables = variables_no_cache
        
        new_key, key = jax.random.split(key)

        # Get conditional probabilities from model
        p, mutables = model.apply(
            variables,
            σ,
            index,
            method=model.conditional,
            mutable=["cache"],
        )
        cache = mutables.get("cache")

        constraint_mask = compute_constraint_mask(phys_table, local_states, sum_scale, sum_offset, index, n_count)
        p_normalized = normalize_probabilities(p, constraint_mask)

        local_states_for_choice = jnp.asarray(hilbert_local_states, dtype=dtype)

        if return_log_probabilities:
            new_σ, new_p = nkjax.batch_choice(
                key, local_states_for_choice, p_normalized, return_prob=True
            )
            log_prob = log_prob + jnp.log(jnp.maximum(new_p, 1e-15))
        else:
            new_σ = nkjax.batch_choice(key, local_states_for_choice, p_normalized)

        σ = σ.at[:, index].set(new_σ)
        n_count = n_count + new_σ.astype(jnp.float32)  # Update sum count

        return (σ, cache, new_key, log_prob, n_count), None

    new_key, key_init, key_scan = jax.random.split(state.key, 3)

    batch_size = n_batches * chain_length
    n_sites = phys_table.shape[0] - 1  
    σ = jnp.zeros((batch_size, n_sites), dtype=dtype)

    # Initialize sum count
    n_count = jnp.zeros(batch_size, dtype=jnp.float32)

    # Add sharding support (same as original ARDirectSampler)
    if config.netket_experimental_sharding:
        σ = jax.lax.with_sharding_constraint(
            σ, jax.sharding.PositionalSharding(jax.devices()).reshape(-1, 1)
        )
        n_count = jax.lax.with_sharding_constraint(
            n_count, jax.sharding.PositionalSharding(jax.devices())
        )

    cache = init_cache(model, σ, key_init)
    if cache:
            variables = {**variables_no_cache, "cache": cache}
    else:
        variables = variables_no_cache
            
    log_prob = (
        jnp.zeros(batch_size) if return_log_probabilities else None
    )

    # Get sampling order from model (same as original ARDirectSampler)
    indices = jnp.arange(n_sites)
    indices = model.apply(variables, indices, method=model.reorder)

    (final_σ, _, _, final_log_prob, _), _ = jax.lax.scan(
        scan_fun,
        (σ, cache, key_scan, log_prob, n_count),
        indices,
    )

    final_σ = final_σ.reshape((n_batches, chain_length, n_sites))
    new_state = state.replace(key=new_key)
    if return_log_probabilities:
        final_log_prob = final_log_prob.reshape((n_batches, chain_length))
        return (final_σ, final_log_prob), new_state  
    else:
        return final_σ, new_state



class ARDirectSamplerSymmetric(ARDirectSampler):
    """
    Direct sampler for autoregressive neural networks with sum conservation.
    
    This sampler enforces exact conservation of the sum of local state values during
    the autoregressive sampling process by pruning branches that cannot lead to valid
    final states. Uses dynamic programming to precompute a reachability table for 
    efficient constraint checking.
    
    Examples:
        - Particle number conservation: local_states=[0,1], target_sum=n_particles
        - Spin magnetization: local_states=[-1,1], target_sum=0 (zero magnetization)
        - General linear constraints: sum(coefficients * local_states) = target
    
    The constraint is enforced by masking out impossible choices at each step
    based on the current sum and remaining sites.
    """

    # Declare all attributes for NetKet's Pytree system
    target_sum: float = struct.field(pytree_node=False)
    n_sites: int = struct.field(pytree_node=False)
    local_states: jnp.ndarray = struct.field(pytree_node=False)
    phys_table: jnp.ndarray = struct.field(pytree_node=False)
    sum_scale: float = struct.field(pytree_node=False)
    sum_offset: float = struct.field(pytree_node=False)

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        target_sum: float,
        dtype: DType = None,
    ):
        """
        Construct a symmetric autoregressive direct sampler.

        Args:
            hilbert: The Hilbert space to sample. Must be unconstrained.
            target_sum: Target sum of local state values to conserve.
                       For spins [-1,1]: target_sum=0 means zero magnetization.
                       For particles [0,1]: target_sum=n means n particles.
            dtype: The dtype of the states sampled (default = hilbert dtype).
        """
        super().__init__(hilbert, dtype=dtype)
        
        # Validate inputs
        if not isinstance(hilbert, DiscreteHilbert):
            raise TypeError("ARDirectSamplerSymmetric only supports DiscreteHilbert spaces")
        
        # Store local states for constraint calculation
        self.local_states = jnp.asarray(hilbert.local_states)
        
        self.target_sum = target_sum
        self.n_sites = hilbert.size
        
        # Validate target value is feasible
        min_possible = float(jnp.min(self.local_states)) * self.n_sites
        max_possible = float(jnp.max(self.local_states)) * self.n_sites
        
        if not (min_possible <= target_sum <= max_possible):
            raise ValueError(
                f"target_sum={target_sum} not achievable with local_states={self.local_states}. "
                f"Range: [{min_possible}, {max_possible}]"
            )
        
        # Pre-compute physicality table using dynamic programming
        self.phys_table = self._build_physicality_table()
    
    def _build_physicality_table(self) -> jnp.ndarray:
        """
        Build a reachability table using vectorized dynamic programming.
        
        Uses JAX vectorized operations instead of nested loops for better performance.
        phys_table[i, sum_idx] = True if it's possible to reach the target sum
        when starting from site i with current sum corresponding to sum_idx.
        
        Returns:
            Array of shape (n_sites + 1, sum_range) with boolean values.
        """
        n_sites = self.n_sites
        local_states = self.local_states
 
        # Determine scaling based on local state values
        min_val, max_val = float(local_states.min()), float(local_states.max())
        
        # Smart scaling logic for different value types
        if jnp.allclose(local_states % 1, 0):
            # Integer values
            scale = 1
        elif jnp.allclose(local_states * 2 % 1, 0):
            # Half-integer values (like spin-1/2: [-0.5, 0.5] or [-1, 1])
            scale = 2
        else:
            # General float values - use higher precision
            scale = 10
        
        # Calculate sum range and offset
        min_sum = min_val * n_sites
        max_sum = max_val * n_sites
        sum_range = int((max_sum - min_sum) * scale) + 1
        offset = int(min_sum * scale)
        
        # Store scaling parameters
        self.sum_scale = float(scale)
        self.sum_offset = float(offset)
        
        # Initialize physicality table
        table = jnp.zeros((n_sites + 1, sum_range), dtype=bool)
        
        # Base case: final site must have exactly target sum
        target_idx = int(self.target_sum * scale) - offset
        if 0 <= target_idx < sum_range:
            table = table.at[n_sites, target_idx].set(True)
        
        # Vectorized backward fill
        for i in range(n_sites - 1, -1, -1):
            # Generate all possible current sum values at this site
            sum_indices = jnp.arange(sum_range)
            current_sums = (sum_indices + offset) / scale
            
            # Compute all possible next sums: current_sum + local_state
            # Shape: (sum_range, n_local_states)
            next_sums = current_sums[:, None] + local_states[None, :]
            next_indices = jnp.round(next_sums * scale).astype(jnp.int32) - offset
            
            # Check bounds and lookup reachability in next layer
            valid_mask = (next_indices >= 0) & (next_indices < sum_range)
            reachable = jnp.where(valid_mask, table[i + 1, next_indices], False)
            
            # A sum is reachable if ANY choice of local state leads to reachability
            can_reach = jnp.any(reachable, axis=1)
            table = table.at[i, :].set(can_reach)
        
        return table
    
    def _sample_chain(
        self,
        model,
        variables,
        state,
        chain_length,
        return_log_probabilities: bool = False,
    ):
        """
        Delegate to external pure function with clean signature.
        
        This maintains API compatibility with the base class while leveraging
        the stateless external function for JIT compilation.
        """                
        return _sample_chain_with_constraints(
            model,
            variables,
            state,
            chain_length,
            return_log_probabilities,
            phys_table=self.phys_table,
            sum_scale=self.sum_scale,
            sum_offset=self.sum_offset,
            local_states=self.local_states,
            hilbert_local_states=self.hilbert.local_states,
            n_batches=self.n_batches,
            dtype=self.dtype,
        )


    def _get_sum_index(self, sum_value: float) -> int:
        """
        Convert a sum value to its corresponding index in the physicality table.
        
        Args:
            sum_value: The sum value to convert
            
        Returns:
            The index in the physicality table
        """
        return int(sum_value * self.sum_scale) - int(self.sum_offset)

    @property
    def is_exact(self) -> bool:
        """
        Returns `True` because the sampler generates exact samples from the 
        constrained distribution.
        """
        return True


