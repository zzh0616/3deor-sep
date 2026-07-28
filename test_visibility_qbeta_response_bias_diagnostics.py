from pathlib import Path

import numpy as np
import pytest

from ops_scripts.evaluate_visibility_qbeta_amplitude_phase_surrogates import (
    _localized_bandpower_surrogates,
    _parse_labeled_paths,
    _spectrally_coherent_spatial_phase_surrogates,
)
from visibility_qbeta import build_sky_band_layout, source_bandpowers


def test_parse_labeled_paths_rejects_duplicate_labels() -> None:
    assert _parse_labeled_paths(["wide=/tmp/wide"]) == {
        "wide": Path("/tmp/wide")
    }
    with pytest.raises(ValueError, match="Duplicate"):
        _parse_labeled_paths(["wide=/tmp/first", "wide=/tmp/second"])


def test_localized_surrogates_preserve_each_block_bandpower() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    eor = torch.randn(8, 8, 8)
    edges = np.asarray([0.0, 1.0, 3.0, 10.0], dtype=np.float64)
    surrogates = _localized_bandpower_surrogates(
        torch=torch,
        eor_k=eor,
        k2jy=torch.ones(8, 1, 1),
        kperp_edges=edges,
        dx_mpc=1.0,
        dy_mpc=1.0,
        dpar_mpc=1.0,
        block_count=2,
        repeats=3,
        seed=7,
        real_dtype=torch.float32,
    ).numpy()
    layout = build_sky_band_layout(
        (4, 8, 8),
        dx_mpc=1.0,
        dy_mpc=1.0,
        dpar_mpc=1.0,
        kperp_edges=edges,
        exclude_radial_nyquist=False,
    )
    for block_index in range(2):
        first = 4 * block_index
        stop = first + 4
        truth = source_bandpowers(eor.numpy()[first:stop], layout)
        measured = source_bandpowers(surrogates[:, first:stop], layout)
        assert np.allclose(
            measured,
            truth[None, :],
            rtol=2e-6,
            atol=1e-8,
        )


def test_spectral_coherence_surrogates_preserve_frequency_covariance() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    restricted = torch.randn(6, 8, 8)
    surrogates = _spectrally_coherent_spatial_phase_surrogates(
        torch=torch,
        restricted_k=restricted,
        k2jy=torch.ones(6, 1, 1),
        repeats=3,
        seed=13,
        real_dtype=torch.float32,
    )
    original_spectrum = torch.fft.fftn(
        restricted, dim=(-2, -1), norm="ortho"
    )
    original_covariance = torch.einsum(
        "fyx,gyx->fgyx",
        original_spectrum,
        original_spectrum.conj(),
    )
    for surrogate in surrogates:
        spectrum = torch.fft.fftn(
            surrogate, dim=(-2, -1), norm="ortho"
        )
        covariance = torch.einsum(
            "fyx,gyx->fgyx", spectrum, spectrum.conj()
        )
        assert torch.allclose(
            covariance,
            original_covariance,
            rtol=2e-5,
            atol=2e-6,
        )
