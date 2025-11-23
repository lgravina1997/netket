
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import Array

from ._operator_data import PNCOperatorDataType
from ._kernels import get_conn_padded_pnc
from ._matrix_elements import _get_mel_offdiag

@partial(jax.jit, static_argnames=("n_fermions",))
@partial(jnp.vectorize, signature="(n)->(m,n),(m)", excluded=(0, 2, 3))
def get_conn_padded_pnc_truncated(
    _operator_data: PNCOperatorDataType,
    x: Array,
    y: Array,
    n_fermions: int,
) -> tuple[Array, Array]:
    r"""
    Compute the connected states and matrix elements for a particle-number-conserving fermionic operator,
    using an approximate approach where only diagonal terms and up to one-body off-diagonal terms are fully considered. 
    Two-body off-diagonal terms are computed separately and restricted to the truncation set defined by `y`.
    
    Args:
        _operator_data: Precomputed operator data containing diagonal and off-diagonal elements.
        x: Input state array of shape (n,). This argument is vectorized over.
        y: Truncation set of states of shape (m, n).
        n_fermions: Number of fermions in the system.
        
    Returns:
        xp: Array of connected states including those from the truncation set, shape (m + p, n).
        mels: Corresponding matrix elements for the connected states, shape (m + p,).
        
    Above, `p` is the number of connected states generated from diagonal and one-body off-diagonal terms.
    """
    
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
        x = x.astype(jnp.int8)
    
    # Take all diagonal elements and only off-diagonal elements with k < 4
    _operator_data_reduced = {
        'diag': _operator_data.get('diag', {}),
        'offdiag': {k: v for k, v in _operator_data.get('offdiag', {}).items() if k < 4}
    }
    
    xp_reduced, mels_reduced = get_conn_padded_pnc(_operator_data_reduced, x, n_fermions)
    
    # Take only off-diagonal elements with k == 4 (two-body terms) excluded from _operator_data_reduced
    _operator_data_2body_offdiag = {k: v for k, v in _operator_data.get('offdiag', {}).items() if k == 4}

    mels_offdiag = jnp.zeros(y.shape[0], dtype=mels_reduced.dtype)
    for k, v in _operator_data_2body_offdiag.items():
        assert k == 4, "This loop should only process k == 4 terms"
        mels_offdiag += _get_mel_offdiag(x, y, *v)

    # Note that xp_reduced may contain duplicates with respect to y.
    # The mels associated to those duplicates should be summed together outside this function.
    xp = jnp.concatenate([xp_reduced, y], axis=-2)
    mels = jnp.concatenate([mels_reduced, mels_offdiag], axis=-1)
    return xp, mels



