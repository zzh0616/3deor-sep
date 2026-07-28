import numpy as np

from ops_scripts.evaluate_visibility_qbeta_independent_lightcone import (
    _load_sky_cache,
)


def test_independent_lightcone_cache_uses_strict_frequency_subset(
    tmp_path,
) -> None:
    path = tmp_path / "sky.npz"
    np.savez_compressed(
        path,
        frequencies_mhz=np.asarray([100.0, 100.1, 100.2, 100.3]),
        l_cosine=np.asarray([0.0, 0.1]),
        m_cosine=np.asarray([0.0, 0.1]),
        n_minus_one=np.asarray([0.0, -0.01]),
        eor_jy=np.arange(8, dtype=np.float64).reshape(4, 2),
        k2jy_per_pixel=np.asarray([1.0, 2.0, 3.0, 4.0]),
    )

    measured = _load_sky_cache(path, np.asarray([100.1, 100.3]))

    assert np.array_equal(
        measured["parent_frequency_indices"], np.asarray([1, 3])
    )
    assert np.array_equal(
        measured["eor_jy"],
        np.asarray([[2.0, 3.0], [6.0, 7.0]]),
    )
    assert np.array_equal(
        measured["k2jy_per_pixel"], np.asarray([2.0, 4.0])
    )
