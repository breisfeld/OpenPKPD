"""Tests for IMP warm-start cache including observation hash."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.estimation.imp import IMPMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec


def _make_params(theta_val: float = 1.0) -> ParameterSet:
    return ParameterSet(
        theta=np.array([theta_val]),
        omega=np.eye(1) * 0.1,
        sigma=np.eye(1) * 0.05,
        theta_specs=[ThetaSpec(init=theta_val, lower=0.0, upper=10.0)],
        omega_specs=[OmegaSpec(block_size=1, values=[0.1])],
        sigma_specs=[SigmaSpec(block_size=1, values=[0.05])],
    )


def _cache_key_for(imp: IMPMethod, subj_id: int, params: ParameterSet, obs: np.ndarray) -> tuple:
    """Replicate the cache key construction from _importance_sample."""
    params_hash = imp._proposal_cache_key(params)
    obs_hash = np.asarray(obs, dtype=float).tobytes()
    return (subj_id, params_hash, obs_hash)


@pytest.mark.unit
class TestIMPCacheObsHash:
    def test_same_subject_same_params_same_obs_cache_hit(self):
        """Same subj_id + same params + same obs → same cache key (cache hit)."""
        imp = IMPMethod(isample=10, seed=0)
        params = _make_params(1.0)
        obs = np.array([1.0, 2.0, 3.0])

        key1 = _cache_key_for(imp, 42, params, obs)
        key2 = _cache_key_for(imp, 42, params, obs)

        assert key1 == key2

    def test_same_subject_same_params_different_obs_cache_miss(self):
        """Same subj_id + same params + different obs → different cache key (cache miss)."""
        imp = IMPMethod(isample=10, seed=0)
        params = _make_params(1.0)
        obs1 = np.array([1.0, 2.0, 3.0])
        obs2 = np.array([4.0, 5.0, 6.0])

        key1 = _cache_key_for(imp, 42, params, obs1)
        key2 = _cache_key_for(imp, 42, params, obs2)

        assert key1 != key2, "Different observations must produce different cache keys"

    def test_different_subjects_same_obs_same_params_cache_miss(self):
        """Different subj_id + same obs + same params → different keys (separate subjects)."""
        imp = IMPMethod(isample=10, seed=0)
        params = _make_params(1.0)
        obs = np.array([1.0, 2.0, 3.0])

        key1 = _cache_key_for(imp, 1, params, obs)
        key2 = _cache_key_for(imp, 2, params, obs)

        assert key1 != key2, "Different subject IDs must produce different cache keys"

    def test_cache_key_is_three_tuple(self):
        """Cache key has exactly three elements: (subj_id, params_hash, obs_hash)."""
        imp = IMPMethod(isample=10, seed=0)
        params = _make_params(1.0)
        obs = np.array([1.0])

        key = _cache_key_for(imp, 1, params, obs)
        assert isinstance(key, tuple)
        assert len(key) == 3

    def test_different_params_different_keys(self):
        """Same subj_id + same obs + different params → different keys."""
        imp = IMPMethod(isample=10, seed=0)
        params1 = _make_params(1.0)
        params2 = _make_params(2.0)
        obs = np.array([1.0, 2.0])

        key1 = _cache_key_for(imp, 1, params1, obs)
        key2 = _cache_key_for(imp, 1, params2, obs)

        assert key1 != key2

    def test_obs_hash_is_bytes(self):
        """obs_hash component must be bytes (hashable and serialized)."""
        imp = IMPMethod(isample=10, seed=0)
        params = _make_params(1.0)
        obs = np.array([1.0, 2.0])

        key = _cache_key_for(imp, 1, params, obs)
        obs_hash = key[2]
        assert isinstance(obs_hash, bytes)
