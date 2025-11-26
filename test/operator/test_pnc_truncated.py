"""
Test suite for particle number conserving operators with truncation (without spin conservation).
Tests the correctness of matrix element calculations for offdiag elements and truncated
get_conn_padded functions on molecular systems.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.ops
from copy import deepcopy

from neuralimportancesampling._src.driver.ngd_antoine.grad_sample.models import PCMolecule
from netket.experimental.operator._particle_number_conserving_fermionic._kernels import (
    get_conn_padded_pnc, get_conn_padded_pnc_spin
)
from netket.experimental.operator._particle_number_conserving_fermionic._kernels_truncated import (
    get_conn_padded_pnc_truncated, get_conn_padded_pnc_spin_truncated,
)
from netket.experimental.operator._particle_number_conserving_fermionic._matrix_elements import (
    _get_mel_offdiag, _get_mel_mixed_offdiag,
)


# Test molecules: LiH, N2, CH4
MOLECULE_CIDS = [62714, 947, 297]

def filter_keys(pytree, filter_func: callable):
    """Filter operator data based on k value."""
    result = {}
    for outer_key, inner_dict in pytree.items():
        result[outer_key] = {
            (k, s): v for (k, s), v in inner_dict.items() if filter_func(k, s)
        }
    return result

@pytest.fixture
def molecule_data(request):
    """Fixture to provide molecule data for each test.

    Expects request.param to be a tuple of (cid, conserve_spin).
    """
    cid, conserve_spin = request.param
    mol, mo_coeff, mf = PCMolecule.molecule(cid=cid)
    molecule = PCMolecule(mol=mol, mo_coeff=mo_coeff, conserve_spin=conserve_spin)
    H = molecule.hamiltonian.to_jax_operator()
    hi = molecule.hilbert_space

    # Get a random state
    all_states = hi.all_states()
    x = all_states[np.random.randint(0, len(all_states))]

    return {
        "operator_data": H._operator_data,
        "hilbert": hi,
        "state": x,
        "cid": cid,
    }


@pytest.mark.parametrize("molecule_data", [(cid, False) for cid in MOLECULE_CIDS], indirect=True)
def test_full_truncated_get_conn_pad(molecule_data):
    """Test that full truncated get_conn_padded matches non-truncated version."""
    _operator_data = molecule_data["operator_data"]
    hi = molecule_data["hilbert"]
    x = molecule_data["state"]

    # Compute non-truncated version
    xp, mels = get_conn_padded_pnc(_operator_data, x, hi.n_fermions)
    xp, inverse_indices = jnp.unique(xp, axis=0, return_inverse=True)
    mels = jax.ops.segment_sum(mels, inverse_indices, num_segments=len(xp))

    # Compute truncated version with full truncation set
    _xp, _ = get_conn_padded_pnc(_operator_data, x, hi.n_fermions)
    _xp = jnp.unique(_xp, axis=0)

    xp_truncated, mels_truncated = get_conn_padded_pnc_truncated(
        _operator_data,
        x,
        _xp,
        hi.n_fermions,
    )
    xp_truncated, inverse_indices = jnp.unique(xp_truncated, axis=0, return_inverse=True)
    mels_truncated = jax.ops.segment_sum(mels_truncated, inverse_indices, num_segments=len(xp_truncated))

    np.testing.assert_allclose(xp, xp_truncated)
    np.testing.assert_allclose(mels, mels_truncated)


@pytest.mark.parametrize("molecule_data", [(cid, False) for cid in MOLECULE_CIDS], indirect=True)
def test_partial_truncated_get_conn_pad(molecule_data):
    """Test that partial truncated get_conn_padded matches manual truncation."""
    _operator_data = molecule_data["operator_data"]
    hi = molecule_data["hilbert"]
    x = molecule_data["state"]

    # Get connected states and truncate to first 50
    all_connected, _ = get_conn_padded_pnc(_operator_data, x, hi.n_fermions)
    # Use min to handle molecules with fewer than 50 connected states
    truncation_size = min(50, len(all_connected))
    y_reduced = all_connected[:truncation_size]

    # Create operator_data with all diagonal terms and off-diagonal terms with k<4
    _operator_data_reduced = deepcopy(_operator_data)
    _operator_data_reduced['offdiag'] = {
        k: v for k, v in _operator_data['offdiag'].items() if k < 4
    }

    xp_reduced, mels_reduced = get_conn_padded_pnc(_operator_data_reduced, x, hi.n_fermions)

    # Create operator_data with only k==4 two-body off-diagonal terms
    _operator_data_twobody_offdiag = {
        "diag": {},
        "offdiag": {
            k: v for k, v in _operator_data['offdiag'].items() if k == 4
        },
    }

    # Manual truncation
    xp_twobody_offdiag, mels_twobody_offdiag = get_conn_padded_pnc(
        _operator_data_twobody_offdiag, x, hi.n_fermions
    )
    mask = (xp_twobody_offdiag[:, None] == y_reduced[None, :]).all(axis=-1).any(axis=-1)
    xp_twobody_offdiag_truncated = xp_twobody_offdiag[mask]
    mels_twobody_offdiag_truncated = mels_twobody_offdiag[mask]

    xp = jnp.concatenate([xp_reduced, xp_twobody_offdiag_truncated], axis=-2)
    mels = jnp.concatenate([mels_reduced, mels_twobody_offdiag_truncated], axis=-1)
    xp, inverse_indices = jnp.unique(xp, axis=0, return_inverse=True)
    mels = jax.ops.segment_sum(mels, inverse_indices, num_segments=len(xp))

    # Using get_conn_padded_pnc_truncated
    y_reduced = jnp.unique(y_reduced, axis=0)

    xp_truncated, mels_truncated = get_conn_padded_pnc_truncated(
        _operator_data,
        x,
        y_reduced,
        hi.n_fermions,
    )
    xp_truncated, inverse_indices = jnp.unique(xp_truncated, axis=0, return_inverse=True)
    mels_truncated = jax.ops.segment_sum(mels_truncated, inverse_indices, num_segments=len(xp_truncated))

    np.testing.assert_allclose(xp, xp_truncated)
    np.testing.assert_allclose(mels, mels_truncated)



@pytest.mark.parametrize("molecule_data", [(cid, True) for cid in MOLECULE_CIDS], indirect=True)
def test_full_truncated_get_conn_pad_spin(molecule_data):
    """Test that full truncated get_conn_padded matches non-truncated version."""
    _operator_data = molecule_data["operator_data"]
    hi = molecule_data["hilbert"]
    x = molecule_data["state"]

    # Compute non-truncated version
    xp, mels = get_conn_padded_pnc_spin(_operator_data, x, hi.n_fermions_per_spin)
    xp, inverse_indices = jnp.unique(xp, axis=0, return_inverse=True)
    mels = jax.ops.segment_sum(mels, inverse_indices, num_segments=len(xp))

    # Compute truncated version with full truncation set
    _xp, _ = get_conn_padded_pnc_spin(_operator_data, x, hi.n_fermions_per_spin)
    _xp = jnp.unique(_xp, axis=0)

    xp_truncated, mels_truncated = get_conn_padded_pnc_spin_truncated(
        _operator_data,
        x,
        _xp,
        hi.n_fermions_per_spin,
    )
    xp_truncated, inverse_indices = jnp.unique(xp_truncated, axis=0, return_inverse=True)
    mels_truncated = jax.ops.segment_sum(mels_truncated, inverse_indices, num_segments=len(xp_truncated))

    np.testing.assert_allclose(xp, xp_truncated)
    np.testing.assert_allclose(mels, mels_truncated)


@pytest.mark.parametrize("molecule_data", [(cid, True) for cid in MOLECULE_CIDS], indirect=True)
def test_partial_truncated_get_conn_pad_spin(molecule_data):
    """Test that partial truncated get_conn_padded matches manual truncation."""
    _operator_data = molecule_data["operator_data"]
    hi = molecule_data["hilbert"]
    x = molecule_data["state"]

    # Get connected states and truncate to first 50
    all_connected, _ = get_conn_padded_pnc_spin(_operator_data, x, hi.n_fermions_per_spin)
    # Use min to handle molecules with fewer than 50 connected states
    truncation_size = min(50, len(all_connected))
    y_reduced = all_connected[:truncation_size]

    # Create operator_data with all diagonal terms and off-diagonal terms with k<4
    _operator_data_reduced = deepcopy(_operator_data)
    _operator_data_reduced['offdiag'] = {
        key: val for key, val in _operator_data['offdiag'].items() if key[0] < 4
    }
    _operator_data_reduced['mixed_offdiag'] = {
        key: val for key, val in _operator_data['mixed_offdiag'].items() if key[0] < 4
    }

    xp_reduced, mels_reduced = get_conn_padded_pnc_spin(_operator_data_reduced, x, hi.n_fermions_per_spin)

    # Create operator_data with only k==4 two-body off-diagonal terms
    _operator_data_twobody_offdiag = filter_keys(_operator_data, lambda k, s: k == 4)
    _operator_data_twobody_offdiag['diag'] = {}
    _operator_data_twobody_offdiag['mixed_diag'] = {}

    # Manual truncation
    xp_twobody_offdiag, mels_twobody_offdiag = get_conn_padded_pnc_spin(
        _operator_data_twobody_offdiag, x, hi.n_fermions_per_spin
    )
    mask = (xp_twobody_offdiag[:, None] == y_reduced[None, :]).all(axis=-1).any(axis=-1)
    xp_twobody_offdiag_truncated = xp_twobody_offdiag[mask]
    mels_twobody_offdiag_truncated = mels_twobody_offdiag[mask]

    xp = jnp.concatenate([xp_reduced, xp_twobody_offdiag_truncated], axis=-2)
    mels = jnp.concatenate([mels_reduced, mels_twobody_offdiag_truncated], axis=-1)
    xp, inverse_indices = jnp.unique(xp, axis=0, return_inverse=True)
    mels = jax.ops.segment_sum(mels, inverse_indices, num_segments=len(xp))

    # Using get_conn_padded_pnc_spin_truncated
    y_reduced = jnp.unique(y_reduced, axis=0)

    xp_truncated, mels_truncated = get_conn_padded_pnc_spin_truncated(
        _operator_data,
        x,
        y_reduced,
        hi.n_fermions_per_spin,
    )
    xp_truncated, inverse_indices = jnp.unique(xp_truncated, axis=0, return_inverse=True)
    mels_truncated = jax.ops.segment_sum(mels_truncated, inverse_indices, num_segments=len(xp_truncated))

    np.testing.assert_allclose(xp, xp_truncated)
    np.testing.assert_allclose(mels, mels_truncated)
