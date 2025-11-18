# this file contains the logic of the operators, essentially _jw_kernel


from functools import partial

import numpy as np
import itertools

import jax
import jax.numpy as jnp

from netket.jax import COOArray
from netket.utils.types import Array

from ._operator_data import PNCOperatorDataType


def _state_to_int(x: Array) -> Array:
    r"""
    Convert occupation vector(s) to integer representation for efficient lookup.

    Args:
        x: occupation vector(s) of shape (..., n_sites)
    Returns:
        integer representation of shape (...)
    """
    powers = 2 ** jnp.arange(x.shape[-1], dtype=jnp.int64)
    return jnp.sum(x.astype(jnp.int64) * powers, axis=-1)


def _int_to_state(state_int: Array, n_sites: int) -> Array:
    r"""
    Convert integer representation(s) back to occupation vector(s).

    Args:
        state_int: integer representation(s), shape (...)
        n_sites: number of sites
    Returns:
        occupation vector(s) of shape (..., n_sites)
    """
    # Use bit operations to extract each bit position
    powers = 2 ** jnp.arange(n_sites, dtype=jnp.int64)
    # Broadcast and extract bits
    return ((state_int[..., None] & powers) > 0).astype(jnp.int8)


def _compute_mel_xy_term(
    x: Array,
    y: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
    n_fermions: int,
) -> Array:
    r"""
    Compute matrix element <y|H_term|x> for a single operator term.

    This function finds ALL (destroy, create) operator pairs within the term
    that connect x to y, and sums their contributions.

    Args:
        x: input state (occupation vector)
        y: target state (occupation vector)
        index_array, create_array, weight_array: sparse operator data for one term
        n_fermions: number of fermions

    Returns:
        matrix element <y|H_term|x> (summed over all contributing paths)
    """
    n_sites = x.shape[0]

    # Determine operator order
    if index_array is not None:
        half_n_ops = index_array.ndim
    else:
        half_n_ops = weight_array.ndim

    # Constant term: y must equal x
    if half_n_ops == 0:
        return jnp.where(jnp.all(x == y), weight_array.reshape(()), 0.0)

    # Diagonal term: accumulate contributions from all occupied site combinations
    is_diagonal_term = (index_array is None)
    if is_diagonal_term:
        is_diagonal = jnp.all(x == y)

        # Get occupied sites and compute weight
        l_occupied = jnp.where(x, size=n_fermions, fill_value=-1)[0]
        k_destroy = _comb(l_occupied, half_n_ops)
        weight = weight_array[tuple(k_destroy)]

        # Compute sign for diagonal term
        sgn = (half_n_ops // 2) % 2
        sign = 1 - 2 * sgn
        mel = sign * weight.sum()

        return jnp.where(is_diagonal, mel, 0.0)

    # Off-diagonal: loop over all destroy combinations and accumulate contributions
    # Get occupied sites in x (potential destroy sites)
    l_occupied = jnp.where(x, size=n_fermions, fill_value=-1)[0]
    k_destroy = _comb(l_occupied, half_n_ops)  # Shape: (half_n_ops, n_destroy_combos)

    if k_destroy.shape[1] == 0:
        return jnp.array(0.0)

    # For each destroy combination, check if any create pattern leads to y
    def check_destroy_combo(k_d):
        # Look up create patterns and weights for this destroy combination
        ind = index_array[tuple(k_d)]
        weight = weight_array[ind]
        l_create = create_array[ind]  # Shape: (n_options, half_n_ops) or (half_n_ops,)

        # Normalize to 2D: (n_options, half_n_ops)
        if l_create.ndim == 1:
            l_create = l_create.reshape(1, -1)
            weight = jnp.array([weight]) if jnp.ndim(weight) == 0 else weight.reshape(-1)

        n_options = l_create.shape[0]

        # Apply destroy operators
        xd = x.at[k_d].set(0)

        # For each create option, check if it produces y
        def check_create_option(opt_idx):
            l_c = l_create[opt_idx]
            w = weight[opt_idx]

            # Apply create operators
            xp = xd.at[l_c].set(1)

            # Check if this produces y
            produces_y = jnp.all(xp == y)

            # Check Pauli exclusion (can't create where particle exists)
            create_was_empty = ~jnp.any(jax.vmap(lambda i: xd[i])(l_c))

            # Compute Jordan-Wigner sign
            m = jnp.arange(n_sites, dtype=k_d.dtype)
            jw_mask_destroy = jax.lax.reduce_xor(k_d[:, None] > m, axes=(0,))
            jw_mask_create = jax.lax.reduce_xor(l_c[:, None] > m, axes=(0,))

            sgn_destroy = jax.lax.reduce_xor(jw_mask_destroy * x, axes=(0,))
            sgn_create = jax.lax.reduce_xor(jw_mask_create * xd, axes=(0,))

            sgn = sgn_create + sgn_destroy
            sgn = jax.lax.bitwise_and(sgn, jnp.ones_like(sgn)).astype(bool)
            sign = 1 - 2 * sgn.astype(np.int8)

            # Return contribution if this path leads to y
            is_valid = produces_y & create_was_empty
            return jnp.where(is_valid, w * sign, 0.0)

        # Sum contributions from all create options for this destroy combination
        contributions = jax.vmap(check_create_option)(jnp.arange(n_options))
        return jnp.sum(contributions)

    # Sum contributions from all destroy combinations
    total_mel = jnp.sum(jax.vmap(check_destroy_combo)(k_destroy.T))

    return total_mel


def _compute_mel_xy_term_interaction_up_down(
    x_down: Array,
    x_up: Array,
    y_down: Array,
    y_up: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
    nelectron_down: int,
    nelectron_up: int,
) -> Array:
    r"""
    Compute matrix element <y|H_term|x> for mixed spin-sector 2-body interaction.

    This handles operators like: Σ w_ijkl c†_{i↓} c†_{j↑} c_{k↑} c_{l↓}

    Args:
        x_down, x_up: input states for down and up spins
        y_down, y_up: target states for down and up spins
        index_array, create_array, weight_array: sparse operator data
        nelectron_down, nelectron_up: number of electrons in each sector

    Returns:
        matrix element <y|H_term|x> (summed over all contributing paths)
    """
    n_sites = x_down.shape[0]

    # Diagonal term: both sectors unchanged
    is_diagonal_term = (index_array is None)
    if is_diagonal_term:
        is_diagonal = jnp.all(x_down == y_down) & jnp.all(x_up == y_up)

        # Get occupied sites in each sector
        down_occupied = jnp.where(x_down, size=nelectron_down, fill_value=-1)[0]
        up_occupied = jnp.where(x_up, size=nelectron_up, fill_value=-1)[0]

        # Meshgrid to get all (down, up) pairs
        k_destroy_down, k_destroy_up = jnp.meshgrid(down_occupied, up_occupied)
        weight = weight_array[k_destroy_down, k_destroy_up]

        sign = 1  # Diagonal terms have positive sign
        mel = sign * weight.sum()

        return jnp.where(is_diagonal, mel, 0.0)

    # Off-diagonal: loop over all (destroy_down, destroy_up) combinations
    down_occupied = jnp.where(x_down, size=nelectron_down, fill_value=-1)[0]
    up_occupied = jnp.where(x_up, size=nelectron_up, fill_value=-1)[0]

    # Create meshgrid of all combinations
    k_destroy_down, k_destroy_up = jnp.meshgrid(down_occupied, up_occupied)
    k_destroy_down = k_destroy_down.ravel()
    k_destroy_up = k_destroy_up.ravel()

    if k_destroy_down.shape[0] == 0:
        return jnp.array(0.0)

    # For each (destroy_down, destroy_up) pair, check if any create pattern leads to y
    def check_destroy_combo(k_d_down, k_d_up):
        # Look up create patterns
        ind = index_array[k_d_down, k_d_up]
        weight = weight_array[ind]
        l_create = create_array[ind]  # Shape: (2,) or (n_options, 2)

        # Normalize to 2D: (n_options, 2)
        if l_create.ndim == 1:
            l_create = l_create.reshape(1, -1)
            weight = jnp.array([weight]) if jnp.ndim(weight) == 0 else weight.reshape(-1)

        n_options = l_create.shape[0]

        # Apply destroy operators
        xd_down = x_down.at[k_d_down].set(0)
        xd_up = x_up.at[k_d_up].set(0)

        # For each create option, check if it produces (y_down, y_up)
        def check_create_option(opt_idx):
            l_c = l_create[opt_idx]
            w = weight[opt_idx]

            # Extract create sites for each sector
            l_c_down = l_c[0]
            l_c_up = l_c[1]

            # Apply create operators
            xp_down = xd_down.at[l_c_down].set(1)
            xp_up = xd_up.at[l_c_up].set(1)

            # Check if this produces (y_down, y_up)
            produces_y = jnp.all(xp_down == y_down) & jnp.all(xp_up == y_up)

            # Check Pauli exclusion
            create_was_empty_down = ~xd_down[l_c_down].astype(bool)
            create_was_empty_up = ~xd_up[l_c_up].astype(bool)
            create_was_empty = create_was_empty_down & create_was_empty_up

            # Compute Jordan-Wigner signs
            m = jnp.arange(n_sites, dtype=jnp.int32)

            # Down sector JW sign
            jw_mask_destroy_down = k_d_down > m
            jw_mask_create_down = l_c_down > m
            sgn_destroy_down = jax.lax.reduce_xor(jw_mask_destroy_down * x_down, axes=(0,))
            sgn_create_down = jax.lax.reduce_xor(jw_mask_create_down * xd_down, axes=(0,))

            # Up sector JW sign
            jw_mask_destroy_up = k_d_up > m
            jw_mask_create_up = l_c_up > m
            sgn_destroy_up = jax.lax.reduce_xor(jw_mask_destroy_up * x_up, axes=(0,))
            sgn_create_up = jax.lax.reduce_xor(jw_mask_create_up * xd_up, axes=(0,))

            # Total sign (combine both sectors)
            sgn = sgn_create_down + sgn_destroy_down + sgn_create_up + sgn_destroy_up
            sgn = jax.lax.bitwise_and(sgn, jnp.ones_like(sgn)).astype(bool)
            sign = 1 - 2 * sgn.astype(np.int8)

            # Return contribution if this path leads to (y_down, y_up)
            is_valid = produces_y & create_was_empty
            return jnp.where(is_valid, w * sign, 0.0)

        # Sum contributions from all create options
        contributions = jax.vmap(check_create_option)(jnp.arange(n_options))
        return jnp.sum(contributions)

    # Sum contributions from all (destroy_down, destroy_up) combinations
    total_mel = jnp.sum(jax.vmap(check_destroy_combo)(k_destroy_down, k_destroy_up))

    return total_mel


def _compute_mel_xy_single_sector(
    x_sector: Array,
    y_sector: Array,
    other_sectors_x: list[Array],
    other_sectors_y: list[Array],
    sector_idx: int,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
    n_fermions: int,
) -> Array:
    r"""
    Compute <y|H_sector|x> for a single-sector operator in a multi-sector system.

    The operator acts only on sector_idx. For the matrix element to be non-zero,
    all OTHER sectors must be unchanged (y_other == x_other).

    Args:
        x_sector, y_sector: occupation vectors for the active sector
        other_sectors_x, other_sectors_y: occupation vectors for all other sectors
        sector_idx: index of the sector being acted upon
        index_array, create_array, weight_array: sparse operator data
        n_fermions: number of fermions in the active sector

    Returns:
        matrix element <y|H_sector|x> (0 if other sectors changed)
    """
    # Check if all other sectors are unchanged
    other_sectors_unchanged = jnp.array(True)
    for x_other, y_other in zip(other_sectors_x, other_sectors_y):
        other_sectors_unchanged = other_sectors_unchanged & jnp.all(x_other == y_other)

    # If other sectors changed, matrix element is zero
    # Otherwise, compute the single-sector matrix element
    mel_sector = _compute_mel_xy_term(
        x_sector, y_sector, index_array, create_array, weight_array, n_fermions
    )

    return jnp.where(other_sectors_unchanged, mel_sector, 0.0)


@partial(jax.jit, static_argnums=(0, 1, 8))
def _get_conn_padded_scan_interaction_up_down(
    nelectron_down: int,
    nelectron_up: int,
    x_down: Array,
    x_up: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
    subspace_states: Array,
    n_total_sites: int,
) -> tuple[Array, Array, Array]:
    r"""
    Memory-efficient version for mixed spin-sector interactions.

    Computes <y|H_interaction|x> directly for each y in subspace.

    Args:
        nelectron_down, nelectron_up: number of electrons in each sector
        x_down, x_up: input states (1D occupation vectors)
        index_array, create_array, weight_array: sparse operator data
        subspace_states: sorted array of valid state integers, shape (n_subspace,)
        n_total_sites: total number of sites (all spin sectors combined)

    Returns:
        y_down: down-sector states, shape (n_subspace, n_sites_per_sector)
        y_up: up-sector states, shape (n_subspace, n_sites_per_sector)
        mels: matrix elements, shape (n_subspace,)
    """
    assert x_down.ndim == 1
    assert x_up.ndim == 1

    n_subspace = subspace_states.shape[0]
    dtype = x_down.dtype

    # Convert subspace integers to full packed occupation vectors
    y_subspace_packed = _int_to_state(subspace_states, n_total_sites)

    # Unpack into spin sectors
    y_subspace_unpacked = unpack_spin_sectors(y_subspace_packed, n_spin_subsectors=2)
    y_down_subspace = y_subspace_unpacked[0]  # Shape: (n_subspace, n_sites_per_sector)
    y_up_subspace = y_subspace_unpacked[1]    # Shape: (n_subspace, n_sites_per_sector)

    # Compute <y|H|x> for each y in subspace using vmap
    mels = jax.vmap(
        lambda y_d, y_u: _compute_mel_xy_term_interaction_up_down(
            x_down, x_up, y_d, y_u,
            index_array, create_array, weight_array,
            nelectron_down, nelectron_up
        )
    )(y_down_subspace, y_up_subspace)

    return y_down_subspace.astype(dtype), y_up_subspace.astype(dtype), mels


@partial(jax.jit, static_argnums=0)
def _get_conn_padded_scan(
    n_fermions: int,
    x: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
    subspace_states: Array,
) -> tuple[Array, Array]:
    r"""
    Memory-efficient version that computes <y|H|x> directly for each y in subspace.

    Args:
        n_fermions: number of electrons
        x: occupation vector (1D)
        index_array, create_array, weight_array: sparse operator data
        subspace_states: sorted array of valid state integers, shape (n_subspace,)

    Returns:
        xp: states of shape (n_subspace, n_sites)
        mels: matrix elements of shape (n_subspace,)
    """
    assert x.ndim == 1

    n_subspace = subspace_states.shape[0]
    n_sites = x.shape[0]
    dtype = x.dtype

    # Convert subspace integers to occupation vectors
    y_subspace = _int_to_state(subspace_states, n_sites)

    # Compute <y|H|x> for each y in subspace using vmap
    mels = jax.vmap(
        lambda y: _compute_mel_xy_term(x, y, index_array, create_array, weight_array, n_fermions)
    )(y_subspace)

    return y_subspace.astype(dtype), mels


def _comb(kl: Array, n: int) -> Array:
    r"""
    compute all combinations of n elements from a set kl
    Args:
        kl: 1d array of elements
        n: size n
    Returns:
        an array containg all combinations of size n of elemenets in kl
    """
    if len(kl) < n:
        return jnp.zeros((n, 0), dtype=kl.dtype)
    c = list(itertools.combinations(np.arange(len(kl)), n))
    return kl[np.array(c, dtype=kl.dtype).T[::-1]]


def _jw_kernel(
    k_destroy: Array, l_create: Array, x: Array
) -> tuple[Array, Array, Array]:
    r"""
    compute all matrix elements :math:`x^\prime` such that :math:`\langle x^\prime | \hat c^\dagger_{l_m} \cdots \hat c^\dagger_{l_1} \hat c_{k_n} \cdots \hat c_{k_1} | x\rangle \neq 0`
    of a batch of states x.

    Args:
        k_destroy: an array of indices of the :math:`\hat c`
        l_create: an array of indices of the :math:`\hat c^\dagger`
        x: an array of states
    Returns:
        xp: matrix elements :math:`x^\prime`
        sign: sign (value) of the matrix element if it is nonzero, arbitrary value otherwise
        create_was_empty: if the matrix element is zero
    """
    # destroy
    xd = jax.vmap(lambda i: x.at[i].set(0))(k_destroy.T)
    # create
    xp = jax.vmap(jax.vmap(lambda x, i: x.at[i].set(1), in_axes=(None, 0)))(
        xd, l_create
    )

    m = jnp.arange(x.shape[-1], dtype=k_destroy.dtype)

    # we apply the destruction operators in descending order,
    # the jordan-wigner sign of an operator does not depend on sites larger than it, therefore,
    # given it is in normal order, we can compute it all in terms of the initial state.
    # (sum the axis which is the one of the indices we destroy/create (size number of operators//2))
    jw_mask_destroy = jax.lax.reduce_xor(k_destroy[..., None] > m, axes=(0,))

    # same for when we create again, except then have to apply it to the state where we already destroyed
    jw_mask_create = jax.lax.reduce_xor(l_create[..., None] > m, axes=(2,))

    create_was_empty = jax.vmap(jax.vmap(lambda x, i: ~x[i].any(), in_axes=(None, 0)))(
        xd, l_create
    )

    sgn_destroy = jax.lax.reduce_xor(
        jw_mask_destroy * x[None], axes=((jw_mask_destroy * x[None]).ndim - 1,)
    )
    sgn_create = jax.lax.reduce_xor(
        jw_mask_create * xd[:, None], axes=((jw_mask_create * xd[:, None]).ndim - 1,)
    )
    sgn = sgn_create + sgn_destroy[:, None]
    sgn = jax.lax.bitwise_and(sgn, jnp.ones_like(sgn)).astype(bool)
    sign = 1 - 2 * sgn.astype(np.int8)

    return xp, sign, create_was_empty


@partial(jax.jit, static_argnums=0)
@partial(jnp.vectorize, signature="(n)->(m,n),(m)", excluded=(0, 2, 3, 4, 5))
def _get_conn_padded(
    n_fermions: int,
    x: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
    subspace_states: Array | None = None,
) -> tuple[Array, Array]:
    r"""
    helper function for the matrix elements functions defined below

    does not know about spin sectors

    Args:
        n_fermions: number of electrons
        x: occupation vectors
        index_array, create_array, weight_array: internal (sparse) operator data representation
        subspace_states: optional array of shape (n_valid_states,) containing integer representations
                        of valid states in the subspace. If provided, returns compacted arrays of size
                        n_valid_states for memory efficiency.
    Returns:
        Connected states and corresponding matrix elements. Returns arrays of shape (n_connected, n_sites) and (n_connected,)
    """
    assert x.ndim == 1

    # Original vectorized path (subspace filtering handled in get_conn_padded_pnc)
    if index_array is not None:
        half_n_ops = index_array.ndim
    else:  # diagonal
        half_n_ops = weight_array.ndim

    if half_n_ops == 0:  # constant
        xp = x[None, :]
        mels = weight_array.reshape(xp.shape[:-1])
    else:
        dtype = x.dtype

        (l_occupied,) = jnp.where(x, size=n_fermions)
        k_destroy = _comb(l_occupied, half_n_ops)

        if index_array is None:  # diagonal
            weight = weight_array[tuple(k_destroy)]
            xp = x[None, :]
            # we first destroy in desc order, then create
            # sites not acted on cancel by the create/destroy pair of the same site,
            # so we can assume they are not there.
            # When we create all smaller sites acted on are 0,
            # therefore the jw sign is determined just from the signs from destroy.
            # Then it' is easy to see that only every other site counts (the rest cancel),
            # and the sign is given by (+1 if there is an even number of other sites, -1 if odd)
            # sign = [+,+,-,-,+,+,-,-,+,+,-,-,...][half_n_ops]
            sgn = (half_n_ops // 2) % 2
            sign = 1 - 2 * sgn
            mels = sign * weight.sum()[None]
        else:
            ind = index_array[tuple(k_destroy)]
            weight = weight_array[ind]
            l_create = create_array[ind]

            xp, sign, create_was_empty = _jw_kernel(k_destroy, l_create, x)
            mels = weight * sign * create_was_empty

            # make sure we don't return states w/ wrong number of electrons
            # because of the padding we check if the mel is 0
            # xp = jnp.where(create_was_empty[..., None], xp, x[..., None, None, :])
            xp = jnp.where((mels == 0)[:, :, None], x[None, None, :], xp)

            xp = jax.lax.collapse(xp, 0, xp.ndim - 1).astype(dtype)
            mels = jax.lax.collapse(mels, 0, mels.ndim)

    return xp, mels


@partial(jax.jit, static_argnames="n_spin_subsectors")
def unpack_spin_sectors(x: Array, n_spin_subsectors: int = 2):
    r"""
    split spin sectors of x

    Args:
        a single stacked array of occupations
    Returns:
        one array for each spin sector
    """
    assert x.shape[-1] % n_spin_subsectors == 0
    x_ = x.reshape(x.shape[:-1] + (n_spin_subsectors, x.shape[-1] // n_spin_subsectors))
    return tuple(x_[..., i, :] for i in range(n_spin_subsectors))


@jax.jit
def pack_spin_sectors(*xs: tuple[Array]) -> Array:
    r"""
    flatten spin sectors of xs

    Args:
        xs: one array of occupations for each spin sector
    Returns:
        a single stacked array
    """
    xs = jnp.broadcast_arrays(*xs)
    xd = xs[0]
    n_spin_subsectors = len(xs)
    res = jnp.zeros(
        xd.shape[:-1]
        + (
            n_spin_subsectors,
            xd.shape[-1],
        ),
        dtype=xd.dtype,
    )
    for i, xi in enumerate(xs):
        res = res.at[..., i, :].set(xi)
    return jax.lax.collapse(res, res.ndim - 2, res.ndim)


@partial(jax.jit, static_argnums=(0, 1))
@partial(jnp.vectorize, signature="(n),(n)->(m,n),(m,n),(m)", excluded=(0, 1, 4, 5, 6))
def _get_conn_padded_interaction_up_down(
    nelectron_down: int,
    nelectron_up: int,
    x_down: Array,
    x_up: Array,
    index_array: Array | COOArray | None,
    create_array: Array | None,
    weight_array: Array,
) -> tuple[Array, Array, Array]:
    r"""
    helper function for the matrix elements for a 2-body interaction term in two different spin sectors
    i.e. of the operator :math:`\sum_{ijkl} w_{ijkl} \hat c^\dagger_{i\downarrow}  \hat c^\dagger_{j_downarrow} \hat c^\dagger_{k_uparrow} \hat c^\dagger_{l_uparrow}`

    Args:
        nelectron_down, nelectron_up: number of electrons in the down and up sector
        x_down, x_up: occupation vectors in both sectors
        index_array, create_array, weight_array: internal (sparse) operator data representation
    Returns:
        connected states and corresponding matrix elements
    """
    dtype = x_down.dtype

    assert x_down.ndim == 1
    if index_array is not None:
        assert index_array.ndim == 2
    else:  # diagonal
        assert weight_array.ndim == 2

    (down_occupied,) = jnp.where(x_down, size=nelectron_down)
    (up_occupied,) = jnp.where(x_up, size=nelectron_up)

    k_destroy_down, k_destroy_up = jnp.meshgrid(down_occupied, up_occupied)

    if index_array is None:  # diagonal
        weight = weight_array[k_destroy_down, k_destroy_up]
        xp_down = x_down[None, :]
        xp_up = x_up[None, :]
        sign = 1
        mels = sign * weight.sum()[None]
    else:
        ind = index_array[k_destroy_down, k_destroy_up].ravel()
        weight = weight_array[ind]
        l_create = create_array[ind]

        k_destroy_down = k_destroy_down.reshape(1, -1)
        k_destroy_up = k_destroy_up.reshape(1, -1)
        l_create_down = l_create[..., :1]
        l_create_up = l_create[..., 1:]

        xp_down, sign_down, down_create_is_not_occupied = _jw_kernel(
            k_destroy_down, l_create_down, x_down
        )
        xp_up, sign_up, up_create_is_not_occupied = _jw_kernel(
            k_destroy_up, l_create_up, x_up
        )

        up_is_diagonal = k_destroy_up[0][:, None] == l_create_up[..., 0]
        down_is_diagonal = k_destroy_down[0][:, None] == l_create_down[..., 0]
        both_not_occupied = (down_create_is_not_occupied | down_is_diagonal) & (
            up_create_is_not_occupied | up_is_diagonal
        )

        sign = sign_up * sign_down
        mels = weight * both_not_occupied * sign

        xp_down = jnp.where((mels == 0)[:, :, None], x_down[None, None, :], xp_down)
        xp_up = jnp.where((mels == 0)[:, :, None], x_up[None, None, :], xp_up)

        xp_down = jax.lax.collapse(xp_down, 0, xp_down.ndim - 1).astype(dtype)
        xp_up = jax.lax.collapse(xp_up, 0, xp_up.ndim - 1).astype(dtype)
        mels = jax.lax.collapse(mels, 0, mels.ndim)
    return xp_down, xp_up, mels


@partial(jax.jit, static_argnames=("n_fermions",))
def get_conn_padded_pnc(
    _operator_data: PNCOperatorDataType,
    x: Array,
    n_fermions: int,
    subspace_states: Array | None = None,
) -> tuple[Array, Array]:
    r"""
    compute the connected elements for ParticleNumberConservingFermioperator2nd

    Args:
        _operator_data: internal sparse operator representation
        x: occupation vectors
        n_fermions: number of electrons
        subspace_states: optional array of shape (n_valid_states,) containing integer representations
                        of valid states in the subspace. If provided, returns compacted arrays for
                        memory efficiency.
    Returns:
        connected states and corresponding matrix elements
        - If subspace_states is None: shape (n_batch, n_connected, n_sites) and (n_batch, n_connected)
        - If subspace_states provided: shape (n_batch, n_valid_states, n_sites) and (n_batch, n_valid_states)
          Output is indexed by position in subspace_states, zeros indicate no connection
    """
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
        x = x.astype(jnp.int8)

    if subspace_states is not None:
        # When filtering to subspace: use direct <y|H|x> computation
        # Sort subspace_states for efficient operations, track permutation
        sort_indices = jnp.argsort(subspace_states)
        subspace_states_sorted = subspace_states[sort_indices]

        # Accumulate contributions from all operator terms
        xp_accum = None
        mels_accum = 0

        # Use vmap to handle batch dimension
        def compute_for_single_sample(x_single):
            xp_local = None
            mels_local = jnp.zeros(subspace_states_sorted.shape[0], dtype=jnp.complex128)

            for k, v in _operator_data["diag"].items():
                xp_term, mels_term = _get_conn_padded_scan(
                    n_fermions, x_single, *v, subspace_states_sorted
                )
                if xp_local is None:
                    xp_local = xp_term
                mels_local = mels_local + mels_term

            for k, v in _operator_data["offdiag"].items():
                xp_term, mels_term = _get_conn_padded_scan(
                    n_fermions, x_single, *v, subspace_states_sorted
                )
                if xp_local is None:
                    xp_local = xp_term
                mels_local = mels_local + mels_term

            return xp_local, mels_local

        # Apply to all samples in batch
        xp_accum, mels_accum = jax.vmap(compute_for_single_sample)(x)

        # Restore original user ordering
        inverse_indices = jnp.argsort(sort_indices)
        xp_accum = xp_accum[:, inverse_indices]
        mels_accum = mels_accum[:, inverse_indices]

        return xp_accum.astype(dtype), mels_accum
    else:
        # When NOT filtering: CONCATENATE (original behavior)
        xp_list = []
        mels_list = []
        xp_diag = None
        mels_diag = 0
        for k, v in _operator_data["diag"].items():
            xp, mels = _get_conn_padded(n_fermions, x, *v, subspace_states)
            xp_diag = xp
            mels_diag = mels_diag + mels
            xp_list = [xp_diag]
            mels_list = [mels_diag]
        for k, v in _operator_data["offdiag"].items():
            xp, mels = _get_conn_padded(n_fermions, x, *v, subspace_states)
            xp_list.append(xp)
            mels_list.append(mels)
        xp = jnp.concatenate(xp_list, axis=-2)
        mels = jnp.concatenate(mels_list, axis=-1)
        return xp.astype(dtype), mels


@partial(jax.jit, static_argnames=("n_fermions_per_spin",))
def get_conn_padded_pnc_spin(
    _operator_data: PNCOperatorDataType,
    x: Array,
    n_fermions_per_spin: tuple[int],
    subspace_states: Array | None = None
) -> tuple[Array, Array]:
    r"""
    compute the connected elements for ParticleNumberAndSpinConservingFermioperator2nd

    Args:
        _operator_data: internal sparse operator representation
        x: occupation vectors (with concatenated spin sectors)
        n_fermions_per_spin: number of electrons in each spin sector
        subspace_states: optional array of shape (n_valid_states,) containing integer representations
                        of valid states in the subspace. If provided, returns compacted arrays of size
                        n_valid_states for memory efficiency.
    Returns:
        connected states and corresponding matrix elements
        - If subspace_states is None: returns arrays of shape (n_connected, n_sites) and (n_connected,)
        - If subspace_states provided: returns arrays of shape (n_valid_states, n_sites) and (n_valid_states,)
    """
    n_spin_subsectors = len(n_fermions_per_spin)
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
        x = x.astype(jnp.int8)

    if subspace_states is not None:
        # When filtering to subspace: use direct <y|H|x> computation
        # Sort subspace_states for efficient operations, track permutation
        sort_indices = jnp.argsort(subspace_states)
        subspace_states_sorted = subspace_states[sort_indices]

        # Get n_total_sites from x
        n_total_sites = x.shape[-1]

        # Use vmap to handle batch dimension
        def compute_for_single_sample(x_single):
            # Unpack spin sectors for x
            xs = unpack_spin_sectors(x_single, n_spin_subsectors)

            # Initialize accumulator
            mels_local = jnp.zeros(subspace_states_sorted.shape[0], dtype=jnp.complex128)
            xp_local = None

            # Initialize xp_local if not done
            if xp_local is None:
                y_subspace_packed = _int_to_state(subspace_states_sorted, n_total_sites)
                xp_local = y_subspace_packed

            # Unpack all y states in subspace into sectors (vmap over batch)
            # This returns a tuple of arrays, each of shape (n_subspace, n_sites_per_sector)
            ys_subspace_tuple = jax.vmap(
                lambda y_packed: unpack_spin_sectors(y_packed, n_spin_subsectors)
            )(y_subspace_packed)

            # Accumulate from diagonal terms (single sector)
            for (k, sectors), v in _operator_data["diag"].items():
                if k == 0:
                    assert sectors == ()
                    sectors = (0,)  # dummy sector

                for i in sectors:
                    # For each y in subspace, compute <y|H_sector_i|x>
                    def compute_mel_for_y(y_sectors):
                        # y_sectors is a tuple of occupation vectors for all sectors
                        y_sector_i = y_sectors[i]
                        # Other sectors (all except i)
                        other_sectors_x = tuple(xs[j] for j in range(n_spin_subsectors) if j != i)
                        other_sectors_y = tuple(y_sectors[j] for j in range(n_spin_subsectors) if j != i)

                        return _compute_mel_xy_single_sector(
                            xs[i], y_sector_i, other_sectors_x, other_sectors_y,
                            i, *v, n_fermions_per_spin[i]
                        )

                    # Transpose ys_subspace_tuple to get list of tuples instead of tuple of lists
                    # Actually, vmap over the first dimension of each array in the tuple
                    mels_term = jax.vmap(compute_mel_for_y)(ys_subspace_tuple)
                    mels_local = mels_local + mels_term

            # Accumulate from mixed diagonal (interactions between sectors)
            for (k, sectors), v in _operator_data["mixed_diag"].items():
                if k != 4:
                    raise NotImplementedError

                for i, j in sectors:
                    assert i > j  # here i>j (e.g., i=up, j=down)
                    # Use the mixed interaction function
                    _, _, mels_term = _get_conn_padded_scan_interaction_up_down(
                        n_fermions_per_spin[j], n_fermions_per_spin[i],
                        xs[j], xs[i], *v,
                        subspace_states_sorted, n_total_sites
                    )
                    mels_local = mels_local + mels_term

            # Accumulate from off-diagonal terms (single sector)
            for (k, sectors), v in _operator_data["offdiag"].items():
                for i in sectors:
                    # For each y in subspace, compute <y|H_sector_i|x>
                    def compute_mel_for_y(y_sectors):
                        y_sector_i = y_sectors[i]
                        # Other sectors (all except i)
                        other_sectors_x = tuple(xs[j] for j in range(n_spin_subsectors) if j != i)
                        other_sectors_y = tuple(y_sectors[j] for j in range(n_spin_subsectors) if j != i)

                        return _compute_mel_xy_single_sector(
                            xs[i], y_sector_i, other_sectors_x, other_sectors_y,
                            i, *v, n_fermions_per_spin[i]
                        )

                    mels_term = jax.vmap(compute_mel_for_y)(ys_subspace_tuple)
                    mels_local = mels_local + mels_term

            # Accumulate from mixed off-diagonal (interactions between sectors)
            for (k, sectors), v in _operator_data["mixed_offdiag"].items():
                if k != 4:
                    raise NotImplementedError

                for i, j in sectors:
                    assert i > j
                    _, _, mels_term = _get_conn_padded_scan_interaction_up_down(
                        n_fermions_per_spin[j], n_fermions_per_spin[i],
                        xs[j], xs[i], *v,
                        subspace_states_sorted, n_total_sites
                    )
                    mels_local = mels_local + mels_term

            return xp_local, mels_local

        # Apply to all samples in batch
        xp_accum, mels_accum = jax.vmap(compute_for_single_sample)(x)

        # Restore original user ordering
        inverse_indices = jnp.argsort(sort_indices)
        xp_accum = xp_accum[:, inverse_indices]
        mels_accum = mels_accum[:, inverse_indices]

        return xp_accum.astype(dtype), mels_accum

    # When NOT filtering: ORIGINAL behavior
    xs = unpack_spin_sectors(x, n_spin_subsectors)
    xs_diag = tuple(a[..., None, :] for a in xs)

    xp_list = []
    mels_list = []
    xp_diag = None
    mels_diag = 0

    # TODO make sectors a jax array and use jax loop here to compile only once ?
    # (requires n_fermions_per_spin to be the same for all sectors)

    for (k, sectors), v in _operator_data["diag"].items():
        if k == 0:
            assert sectors == ()
            sectors = (0,)  # dummy sector

        for i in sectors:
            # act on a single sector: we use the non-spin mel code here for the mels
            _, melsi = _get_conn_padded(n_fermions_per_spin[i], xs[i], *v)
            mels_diag = mels_diag + melsi
            xp_diag = x[..., None, :]

    for (k, sectors), v in _operator_data["mixed_diag"].items():
        if k != 4:
            raise NotImplementedError
        # TODO make sectors a jax array and use jax loop here to compile only once
        for i, j in sectors:
            assert i > j  # here i>j
            # e.g. take operator data to be c_ijkl + c_jilk so that here we only need to sum  ρ > σ (i.e. σ=d, ρ=u)
            *_, melsij = _get_conn_padded_interaction_up_down(
                n_fermions_per_spin[j], n_fermions_per_spin[i], xs[j], xs[i], *v
            )
            mels_diag = mels_diag + melsij
            xp_diag = x[..., None, :]

    # TODO optionally always add zero diagonal?
    if xp_diag is not None:
        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for (k, sectors), v in _operator_data["offdiag"].items():
        # TODO make sectors a jax array and use jax loop here to compile only once ?
        # (requires n_fermions_per_spin to be the same for all sectors)
        for i in sectors:
            # act on a single sector: we use the non-spin mel code here, and just apply an identity in the other sectors
            xpi, melsi = _get_conn_padded(n_fermions_per_spin[i], xs[i], *v)
            xpi = pack_spin_sectors(*xs_diag[:i], xpi, *xs_diag[i + 1 :])
            xp_list.append(xpi)
            mels_list.append(melsi)

    for (k, sectors), v in _operator_data["mixed_offdiag"].items():
        if k != 4:
            raise NotImplementedError
        for i, j in sectors:
            assert i > j  # here i>j
            # e.g. take operator data to be c_ijkl + c_jilk so that here we only need to sum  ρ > σ (i.e. σ=d, ρ=u)
            xpj, xpi, melsij = _get_conn_padded_interaction_up_down(
                n_fermions_per_spin[j], n_fermions_per_spin[i], xs[j], xs[i], *v
            )
            xpij = pack_spin_sectors(
                *xs_diag[:j], xpj, *xs_diag[j + 1 : i], xpi, *xs_diag[i + 1 :]
            )
            xp_list.append(xpij)
            mels_list.append(melsij)
    if len(xp_list) > 0:
        xp = jnp.concatenate(xp_list, axis=-2).astype(dtype)
        mels = jnp.concatenate(mels_list, axis=-1)
    else:
        xp = jnp.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=dtype)
        mels = jnp.zeros(xp.shape[:-1])  # TODO dtype?
    return xp, mels
