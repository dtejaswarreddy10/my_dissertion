"""
Numeric core for the accuracy study (golden/accuracy_study.py).

Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
Stage   : Month 4 / Week 14 - accuracy characterisation

Pure numpy, no torch, so it can be self-tested against golden_model.py with no
extra install:

    python3 golden/quant_core.py          # runs the self-test

Arithmetic modelled (same contract as the RTL / golden_model.py):
    * INT8   : signed 8-bit operands, range -128..127, 32-bit accumulate
    * INT4x2 : signed 4-bit operands, range -8..7; two K-steps per cycle is a
               THROUGHPUT detail only - numerically it is a plain INT4 dot
               product, so accuracy is modelled as INT4
    * 2:4    : keep the 2 largest-magnitude weights of every 4 consecutive
               weights along K; ties go to the lower index (identical to
               golden_model.prune_group_2of4)

Quantisation scheme (one choice, stated so the paper can quote it):
    * weights     : symmetric, per OUTPUT channel (the scale is applied after
                    the array, outside the MAC datapath)
    * activations : symmetric, per tensor, clipped at a calibrated range
"""

import os
import sys

import numpy as np

GROUP = 4
NNZ = 2


def qmax(bits):
    """Largest positive code: 127 for INT8, 7 for INT4."""
    return (1 << (bits - 1)) - 1


def prune_2of4_rows(w):
    """2:4 prune along the LAST axis.

    The last axis is zero-padded to a multiple of 4 (e.g. a 7x7x3 first conv
    has K=147) and trimmed back afterwards. Padding zeros never beat a real
    weight, so they cannot displace a survivor.
    """
    w = np.asarray(w)
    k = w.shape[-1]
    pad = (-k) % GROUP
    if pad:
        wp = np.concatenate([w, np.zeros(w.shape[:-1] + (pad,), dtype=w.dtype)], axis=-1)
    else:
        wp = w
    g = wp.reshape(wp.shape[:-1] + (-1, GROUP))
    order = np.argsort(-np.abs(g), axis=-1, kind="stable")   # ties -> lower index
    keep = np.zeros(g.shape, dtype=bool)
    np.put_along_axis(keep, order[..., :NNZ], True, axis=-1)
    out = np.where(keep, g, 0).reshape(wp.shape)
    return out[..., :k]


def quantize_weights(w, bits):
    """Per-output-channel symmetric quantisation. w is [Cout, ...].

    Returns (q, s): q holds integer codes (as float64), s has shape
    [Cout, 1, ...] so that q * s is the dequantised weight.
    """
    qm = qmax(bits)
    flat = w.reshape(w.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    scale = np.where(amax > 0, amax / qm, 1.0)
    s = scale.reshape((-1,) + (1,) * (w.ndim - 1))
    q = np.clip(np.rint(w / s), -qm - 1, qm)
    return q, s


def quantize_acts(x, clip, bits):
    """Per-tensor symmetric activation quantisation. Returns (q, scale)."""
    qm = qmax(bits)
    scale = max(float(clip), 1e-12) / qm
    q = np.clip(np.rint(np.asarray(x) / scale), -qm - 1, qm)
    return q, scale


def rel_mae(y, ref):
    """Relative mean-absolute error: sum|y-ref| / sum|ref|."""
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.abs(np.asarray(y, dtype=np.float64) - ref).sum() / max(np.abs(ref).sum(), 1e-30))


# ---------------------------------------------------------------------------
# Self-test against the bit-accurate hardware reference
# ---------------------------------------------------------------------------

def selftest(seed=0):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import golden_model as gm

    rng = np.random.RandomState(seed)
    M, K, N = 6, 32, 5

    # 1. INT8: integer GEMM equals the hardware golden GEMM.
    A = rng.randint(-128, 128, (M, K))
    B = rng.randint(-128, 128, (K, N))
    ref = np.array(gm.gemm_golden(A.tolist(), B.tolist()))
    assert np.array_equal(A.astype(np.int64) @ B.astype(np.int64), ref), "INT8 GEMM mismatch"

    # 2. INT4x2: packing two K-steps per byte gives the same sums as plain INT4.
    A4 = rng.randint(-8, 8, (M, K))
    B4 = rng.randint(-8, 8, (K, N))
    Ap = [[gm.pack_int4x2(int(A4[m, 2 * j + 1]), int(A4[m, 2 * j])) for j in range(K // 2)]
          for m in range(M)]
    Bp = [[gm.pack_int4x2(int(B4[2 * j + 1, n]), int(B4[2 * j, n])) for n in range(N)]
          for j in range(K // 2)]
    ref4 = np.array(gm.gemm_golden_int4x2(Ap, Bp))
    assert np.array_equal(A4.astype(np.int64) @ B4.astype(np.int64), ref4), "INT4x2 GEMM mismatch"

    # 3. 2:4 pruning identical to golden_model, including heavy ties and zeros.
    for lo, hi in ((-128, 128), (-3, 4), (0, 2)):
        Bk = rng.randint(lo, hi, (K, 7))
        golden, _ = gm.prune_matrix_2of4(Bk.tolist())
        assert np.array_equal(prune_2of4_rows(Bk.T).T, np.array(golden)), "2:4 mismatch %s" % ((lo, hi),)

    # 4. Padding path (K not a multiple of 4): shape kept, <=2 survivors per group.
    wp = rng.randn(5, 10)
    pr = prune_2of4_rows(wp)
    assert pr.shape == wp.shape
    padded = np.concatenate([pr, np.zeros((5, 2))], axis=1).reshape(5, -1, 4)
    assert ((padded != 0).sum(-1) <= 2).all(), "more than 2 survivors in a group"

    # 5. Fake-quant (what the torch study runs) equals the integer datapath x scales.
    for bits in (8, 4):
        W = rng.randn(N, K)
        X = rng.randn(M, K) * 3
        qw, s = quantize_weights(W, bits)
        xq, sx = quantize_acts(X, np.abs(X).max(), bits)
        fake = (xq * sx) @ (qw * s).T
        integer = (xq.astype(np.int64) @ qw.astype(np.int64).T) * sx * s.reshape(1, -1)
        assert np.allclose(fake, integer, rtol=1e-9, atol=1e-9), "fake-quant != integer path (INT%d)" % bits
        if bits == 8:
            gold = np.array(gm.gemm_golden(xq.astype(int).tolist(), qw.astype(int).T.tolist()))
            assert np.array_equal(xq.astype(np.int64) @ qw.astype(np.int64).T, gold)

    print("quant_core selftest: PASS (INT8, INT4x2, 2:4 incl. ties/padding, fake-quant == integer path)")


if __name__ == "__main__":
    selftest()
