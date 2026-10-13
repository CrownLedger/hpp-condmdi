"""Cubic Hermite trajectory reconstruction (handoff sec. 12).

    p(u) = h00(u) p0 + h10(u) m0 + h01(u) p1 + h11(u) m1,   u in [0, 1]
    h00 = 2u^3 - 3u^2 + 1,  h10 = u^3 - 2u^2 + u,  h01 = -2u^3 + 3u^2,  h11 = u^3 - u^2

m0, m1 are the endpoint derivatives with respect to normalised segment time, i.e. the time derivative dp/dt
(per frame) multiplied by the segment duration. Tangents use Fritsch-Carlson (monotone, "PCHIP") estimates per
axis: they are scaled consistently for non-uniform key spacing and keep each segment within the range of its two
key values on every axis, so no overshoot. Keys are reproduced exactly. Deterministic.
"""
import numpy as np


def basis(u):
    u2, u3 = u * u, u * u * u
    return 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2


def fritsch_carlson_slopes(t, y):
    """dy/dt at each key (per axis). t: (K,), y: (K, D)."""
    t = np.asarray(t, dtype=np.float64)
    h = np.diff(t)[:, None]                                  # (K-1,1)
    delta = np.diff(y, axis=0) / h                           # secant slopes (K-1,D)
    K = len(t)
    m = np.zeros_like(y, dtype=np.float64)
    if K == 2:
        m[:] = delta[0]
        return m
    # interior: weighted harmonic mean when secants agree in sign, else 0 (local extremum -> flat)
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    d0, d1 = delta[:-1], delta[1:]
    same = (d0 * d1) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        hm = (w1 + w2) / (w1 / d0 + w2 / d1)
    m[1:-1] = np.where(same, hm, 0.0)
    # endpoints: one-sided three-point estimate, limited to stay shape preserving
    for end, (hh0, hh1, dd0, dd1) in ((0, (h[0], h[1], delta[0], delta[1])), (-1, (h[-1], h[-2], delta[-1], delta[-2]))):
        d = ((2 * hh0 + hh1) * dd0 - hh0 * dd1) / (hh0 + hh1)
        d = np.where(np.sign(d) != np.sign(dd0), 0.0, d)
        d = np.where((np.sign(dd0) != np.sign(dd1)) & (np.abs(d) > np.abs(3 * dd0)), 3 * dd0, d)
        m[end] = d
    return m


def hermite_reconstruct(key_frames, key_values, n_frames):
    """Evaluate the piecewise cubic Hermite curve through (key_frames, key_values) at frames 0..n_frames-1.
    key_values: (K, D). Frames outside [first key, last key] hold the end values."""
    k = np.asarray(key_frames, dtype=np.int64)
    y = np.asarray(key_values, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    assert np.all(np.diff(k) > 0), 'key frames must be strictly increasing'
    out = np.empty((n_frames, y.shape[1]))
    if len(k) == 1:
        out[:] = y[0]
        return out
    m = fritsch_carlson_slopes(k, y)
    out[:k[0]] = y[0]
    out[k[-1]:] = y[-1]
    for i in range(len(k) - 1):
        a, b = k[i], k[i + 1]
        dur = b - a
        u = ((np.arange(a, b + 1) - a) / dur)[:, None]
        h00, h10, h01, h11 = basis(u)
        out[a:b + 1] = h00 * y[i] + h10 * (m[i] * dur) + h01 * y[i + 1] + h11 * (m[i + 1] * dur)
    return out
