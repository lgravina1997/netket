import jax
import jax.numpy as jnp
from functools import partial

from netket.jax import COOArray
from netket.utils.types import Array

@jax.jit
def hamming_distance(x, y):
    """
    Args:
        x, y: 1D JAX arrays of 0/1 values with the same shape.

    Returns:
      d : Hamming distance (scalar)
    """
    return jnp.sum(x != y)
  
  
@partial(jax.jit, static_argnames=['k'])
def select_changes(x: Array, y: Array, k: int = 4):
    diff = x - y  
    size = k // 2
    k_destroy = jnp.argwhere(diff == 1, size=size, fill_value=-1).reshape(-1)
    l_create = jnp.argwhere(diff == -1, size=size, fill_value=-1).reshape(-1)
    return jnp.flip(k_destroy), jnp.flip(l_create)


@jax.jit
def jw_sign_fast(x, k_destroy, l_create):
    """
    Fast and correct Jordan–Wigner sign that matches the original implementation.
    """
    prefix = jnp.cumsum(x) - x # number of ones to the left
    parity_destroy = jnp.sum(prefix[k_destroy])

    xd = x.at[k_destroy].set(0) # apply destruction BEFORE computing create parity
    prefix_d = jnp.cumsum(xd) - xd # prefix after destruction

    parity_create = jnp.sum(prefix_d[l_create])

    total_parity = (parity_destroy + parity_create) & 1 # total parity
    return 1 - 2 * total_parity # (-1)^parity


@partial(jax.jit, static_argnames=['n_fermions'])
@partial(jnp.vectorize, signature="(n)->()", excluded=(0, 1, 3, 4, 5))
def _get_mel_offdiag(
    n_fermions: int,
    x: Array,
    y: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
):
    r"""
    Get the matrix element between two states `x` and `y` for two-body operators
    of the form 
    
    .. math::
        c^\dagger_{i} c^\dagger_{j} c_{k} c_{l}
        
    Args:
        n_fermions: int 
            Number of fermions in the system.
        x: Array
            Initial state (1D array of 0/1 values).
        y: Array
            Final state (1D array of 0/1 values).
        index_array: Array | COOArray | None
            Precomputed index array for the two-body operator.
        create_array: Array | None
            Precomputed creation array for the two-body operator.
        weight_array: Array
            Precomputed weight array for the two-body operator.
            
    Returns:
        mel: Array
            The matrix element connecting `x` to `y`.
    """
    def compute(k_destroy, l_create, ind):
        creates = create_array[ind] # shape (n_max, n_fermions)
        
        mask = jnp.all(creates == l_create[..., None, :], axis=-1)
        idx = jnp.argmax(mask)
        found = mask[idx]
        
        sgn = jw_sign_fast(x, k_destroy, l_create)
        return jnp.where(found, sgn * weight_array[ind, idx], 0.0)

    def case_k4():
        r"""
        Handles the case where the Hamming distance between `x` and `y` is 4,
        corresponding to two-body operator transitions between different sites, i.e
        
        .. math::
            c^\dagger_{i} c^\dagger_{j} c_{k} c_{l} with i,j,k,l all different.
            
        An example of such a transition is:
        x = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        y = [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        """
        k_destroy, l_create = select_changes(x, y, k=4)
        ind = index_array[tuple(k_destroy)]
        return compute(k_destroy, l_create, ind)

    def case_k2():
        r"""
        Handles the case where the Hamming distance between `x` and `y` is 2,
        corresponding to two-body operator transitions that involve the destruction
        and creation of a particle in the same site, i.e
        
        .. math::
            c^\dagger_{i} c^\dagger_{j} c_{j} c_{k}  or  c^\dagger_{i} c^\dagger_{i} c_{k} c_{l}
            
        An example of such a transition is:
        x = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        y = [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        """
        k_destroy_, l_create_ = select_changes(x, y, k=2) # identifies the two sites that are different
        same_sites = jnp.where((x & y), size=n_fermions-1, fill_value=-1)[0] # all sites where a particle could have been both destroyed and created
        
        def f_one_site(i):
            r"""
            Computes the matrix element contribution for a single same-site transition.
            Args:
                i: Index of the site where a particle is both destroyed and created.
            Returns:
                Matrix element contribution for this specific same-site transition.
            """
            k_destroy = jnp.sort(jnp.array([k_destroy_[0], i]), descending=True)
            l_create = jnp.sort(jnp.array([l_create_[0], i]), descending=True)
            ind = index_array[tuple(k_destroy)]
            return compute(k_destroy, l_create, ind)

        mels = jax.vmap(f_one_site)(same_sites) # vectorize over all i in same_sites
        return jnp.sum(mels)

    d = hamming_distance(x, y)
    return jnp.where(d == 4, case_k4(), jnp.where(d == 2, case_k2(), 0.0))


@partial(jax.jit, static_argnames=['n_fermions_per_spin'])
@partial(jnp.vectorize, signature="(n),(n)->()", excluded=(0, 1, 2, 5, 6, 7))
def _get_mel_mixed_offdiag(
    n_fermions_per_spin: tuple[int],    
    x_down: Array,
    x_up: Array,
    y_down: Array,
    y_up: Array,
    index_array: Array,
    create_array: Array,
    weight_array: Array,
):
    """
    Compute matrix element for mixed_offdiag (cross-sector) two-body operators.
    Handles transitions where one sector has a hop and the other has same-site. For example:
    x_down = [1, 1, 0, 0, 0, 0 || 1, 1, 0, 0, 0, 0]
    x_up   = [1, 0, 0, 1, 0, 0 || 0, 1, 0, 1, 0, 0]
    y_down = [1, 0, 0, 1, 0, 0 || 0, 1, 0, 1, 0, 0]
    y_up   = [1, 1, 0, 0, 0, 0 || 1, 1, 0, 0, 0, 0]
    
    or transitions where there is a hop in both sectors:
    x_down = [1, 1, 0, 0, 0, 0 || 1, 1, 0, 0, 0, 0]
    x_up   = [1, 0, 0, 1, 0, 0 || 0, 1, 0, 1, 0, 0]
    y_down = [1, 0, 0, 0, 0, 1 || 1, 1, 0, 0, 0, 0]
    y_up   = [1, 1, 0, 0, 0, 0 || 1, 1, 0, 0, 0, 0]
    
    Args:
        n_fermions_per_spin: Number of fermions in each spin sector.
        x_down, x_up: Initial states for down and up sectors.
        y_down, y_up: Final states for down and up sectors.
        index_array: Precomputed index array for the two-body operator.
        create_array: Precomputed creation array for the two-body operator.
        weight_array: Precomputed weight array for the two-body operator.
    
    Returns:
        mel: The matrix element connecting `x` to `y`.
    """
    
    d_down = hamming_distance(x_down, y_down)
    d_up = hamming_distance(x_up, y_up)
    
    def case_hop_in_down():
        # Hop in down, same-site in up
        k_destroy_down, l_create_down = select_changes(x_down, y_down, k=2)
        same_sites_up = jnp.where((x_up & y_up), size=n_fermions_per_spin[1], fill_value=-1)[0]
        
        def f_one_site(j_up):
            ind = index_array[k_destroy_down[0], j_up]
            creates = create_array[ind]
            target = jnp.array([l_create_down[0], j_up])
            
            mask = jnp.all(creates == target, axis=1)
            idx = jnp.argmax(mask)
            found = mask[idx]
            
            sgn = jw_sign_fast(x_down, k_destroy_down, l_create_down)
            return jnp.where(found, sgn * weight_array[ind, idx], 0.0)
        
        mels = jax.vmap(f_one_site)(same_sites_up)
        return jnp.sum(mels)
    
    def case_hop_in_up():
        # Hop in up, same-site in down
        k_destroy_up, l_create_up = select_changes(x_up, y_up, k=2)
        same_sites_down = jnp.where((x_down & y_down), size=n_fermions_per_spin[0], fill_value=-1)[0]
        
        def f_one_site(j_down):
            ind = index_array[j_down, k_destroy_up[0]]
            creates = create_array[ind]
            target = jnp.array([j_down, l_create_up[0]])
            
            mask = jnp.all(creates == target, axis=1)
            idx = jnp.argmax(mask)
            found = mask[idx]
            
            sgn = jw_sign_fast(x_up, k_destroy_up, l_create_up)
            return jnp.where(found, sgn * weight_array[ind, idx], 0.0)
        
        mels = jax.vmap(f_one_site)(same_sites_down)
        return jnp.sum(mels)
    
    def case_hop_in_both():
        # Hop in both sectors 
        k_destroy_down, l_create_down = select_changes(x_down, y_down, k=2)
        k_destroy_up, l_create_up = select_changes(x_up, y_up, k=2)
        ind = index_array[k_destroy_down[0], k_destroy_up[0]]
        creates = create_array[ind]
        target = jnp.array([l_create_down[0], l_create_up[0]])
        
        mask = jnp.all(creates == target, axis=1)
        idx = jnp.argmax(mask)
        found = mask[idx]
        
        sgn_down = jw_sign_fast(x_down, k_destroy_down, l_create_down)
        sgn_up = jw_sign_fast(x_up, k_destroy_up, l_create_up)
        return jnp.where(found, sgn_down * sgn_up * weight_array[ind, idx], 0.0)

    # Select based on which sector has the hop
    return jnp.where(
        (d_down == 2) & (d_up == 0),
        case_hop_in_down(),
        jnp.where(
            (d_down == 0) & (d_up == 2),
            case_hop_in_up(),
            jnp.where(
                (d_down == 2) & (d_up == 2),
                case_hop_in_both(),
                0.0
            )
        )
    )
