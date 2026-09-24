"""
cocotb testbench for rtl/systolic_array_sparse.sv (full design).

Exercises all three contributions together:
    * INT8 / INT4x2 precision modes           (pe_mixed)
    * 2:4 compressed weights at the north edge (sparse_feeder + decoder)
    * per-PE operand isolation and clock gating (ENABLE_GATING)

DRIVER CONTRACT
    Activations stay dense and are row-skewed exactly as before. Weights arrive
    COMPRESSED: for column j at cycle t the feeder is on k-step (t - j), so the
    group presented must be compressed[(t-j)//4][j] and it must stay stable for
    the four cycles of that group. feed_en[j] is asserted from cycle j onward,
    which is what gives column j its injection delay.

REFERENCE
    The array computes A x B_pruned, NOT A x B. Pruning is a property of the
    weights, so the golden reference is gemm_golden(A, B_pruned).

GATING EQUIVALENCE
    ENABLE_GATING is a compile-time parameter. Results must be bit-identical
    for both settings - only switching activity changes. Run:
        make TEST=array_sparse GATING=0
        make TEST=array_sparse GATING=1
    and confirm both pass.

Run:  cd tb && make TEST=array_sparse
"""

import os
import pathlib
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "golden"))
from golden_model import (  # noqa: E402
    MODE_INT4X2,
    MODE_INT8,
    gemm_golden,
    gemm_golden_int4x2,
    make_sparse_matrix,
    pack_int4x2,
    prune_matrix_2of4,
    to_signed,
)

N = int(os.environ.get("TB_N", "8"))
K = int(os.environ.get("TB_K", "8"))          # must be a multiple of 4
DATA_WIDTH = 8
ACC_WIDTH = 32
DATA_MASK = (1 << DATA_WIDTH) - 1
ACC_MASK = (1 << ACC_WIDTH) - 1
CLK_PERIOD_NS = 10
SEED = int(os.environ.get("TB_SEED", "1"))
NUM_MATRICES = int(os.environ.get("TB_MATRICES", "20"))
# Activation sparsity for the power-characterisation stimulus (M3.6).
SPARSITY = float(os.environ.get("TB_SPARSITY", "0.5"))
POWER_TILES = int(os.environ.get("TB_POWER_TILES", "16"))

assert K % 4 == 0, "TB_K must be a multiple of 4 for the 2:4 contract"


def read_c(dut, i, j):
    raw = int(dut.c_out.value)
    return to_signed((raw >> ((i * N + j) * ACC_WIDTH)) & ACC_MASK, ACC_WIDTH)


def check_result(dut, C_exp, ctx=""):
    bad = []
    for i in range(N):
        for j in range(N):
            got = read_c(dut, i, j)
            if got != C_exp[i][j]:
                bad.append((i, j, got, C_exp[i][j]))
    assert not bad, (
        f"{ctx}: {len(bad)} mismatch(es), first 5: "
        + ", ".join(f"C[{i}][{j}] RTL={g} golden={e}" for i, j, g, e in bad[:5])
    )


async def reset_dut(dut, mode=MODE_INT8):
    dut.rst.value = 1
    dut.mode.value = int(mode)
    dut.acc_clear.value = 0
    dut.a_west.value = 0
    dut.valid_west.value = 0
    dut.grp_nz.value = 0
    dut.grp_meta.value = 0
    dut.feed_en.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def clear_accumulators(dut):
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.acc_clear.value = 0


async def stream_sparse(dut, mode, A, compressed, depth):
    """Drive dense activations + compressed weights with the required skew."""
    dut.mode.value = int(mode)

    for t in range(N + depth - 1):
        a_vec = v_vec = 0
        nz_vec = meta_vec = en_vec = 0

        for i in range(N):
            k = t - i
            if 0 <= k < depth:
                a_vec |= (A[i][k] & DATA_MASK) << (i * DATA_WIDTH)
                v_vec |= 1 << i

        for j in range(N):
            k = t - j
            if 0 <= k < depth:
                nz_word, meta = compressed[k // 4][j]
                nz_vec |= (nz_word & ((1 << (2 * DATA_WIDTH)) - 1)) << (j * 2 * DATA_WIDTH)
                meta_vec |= (meta & 0xF) << (j * 4)
                en_vec |= 1 << j

        dut.a_west.value = a_vec
        dut.valid_west.value = v_vec
        dut.grp_nz.value = nz_vec
        dut.grp_meta.value = meta_vec
        dut.feed_en.value = en_vec

        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)

        # Metadata legality is checked live: an illegal group would silently
        # drop a survivor and corrupt the result.
        assert int(dut.meta_legal.value) == (1 << N) - 1, (
            f"cycle {t}: illegal 2:4 metadata on some column "
            f"(meta_legal=0b{int(dut.meta_legal.value):0{N}b})"
        )

    dut.a_west.value = 0
    dut.valid_west.value = 0
    dut.feed_en.value = 0
    for _ in range(2 * N + depth + 4):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


def make_sparse_problem(rng, mode):
    """Build (A, B_pruned, compressed) for the requested precision mode."""
    if mode == MODE_INT8:
        A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(N)]
        B = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
    else:
        A = [[pack_int4x2(rng.randint(-8, 7), rng.randint(-8, 7)) for _ in range(K)]
             for _ in range(N)]
        B = [[pack_int4x2(rng.randint(-8, 7), rng.randint(-8, 7)) for _ in range(N)]
             for _ in range(K)]

    B_pruned, compressed = prune_matrix_2of4(B, DATA_WIDTH)
    return A, B_pruned, compressed


# ---------------------------------------------------------------------------
# Functional correctness
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_sparse_int8_random(dut):
    """Compressed-weight INT8 GEMM == dense GEMM against the pruned matrix."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED)
    await reset_dut(dut, MODE_INT8)

    for n in range(NUM_MATRICES):
        A, B_pruned, comp = make_sparse_problem(rng, MODE_INT8)
        await clear_accumulators(dut)
        await stream_sparse(dut, MODE_INT8, A, comp, K)
        check_result(dut, gemm_golden(A, B_pruned), ctx=f"sparse int8[{n}]")

    dut._log.info("Sparse INT8 GEMM passed: %d matrix pairs", NUM_MATRICES)


@cocotb.test()
async def test_sparse_int4x2_random(dut):
    """Compressed weights combined with packed INT4x2 lanes."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 1)
    await reset_dut(dut, MODE_INT4X2)

    for n in range(NUM_MATRICES):
        A, B_pruned, comp = make_sparse_problem(rng, MODE_INT4X2)
        await clear_accumulators(dut)
        await stream_sparse(dut, MODE_INT4X2, A, comp, K)
        check_result(dut, gemm_golden_int4x2(A, B_pruned), ctx=f"sparse int4[{n}]")

    dut._log.info("Sparse INT4x2 GEMM passed: %d matrix pairs", NUM_MATRICES)


@cocotb.test()
async def test_pruned_weights_are_half_zero(dut):
    """Sanity: the streamed weight matrix really is 50% zero (2 of every 4)."""
    rng = random.Random(SEED + 2)
    _, B_pruned, _ = make_sparse_problem(rng, MODE_INT8)

    for j in range(N):
        for g in range(K // 4):
            grp = [B_pruned[g * 4 + p][j] for p in range(4)]
            assert sum(1 for v in grp if v != 0) <= 2, (
                f"column {j} group {g} violates the 2:4 contract: {grp}"
            )

    total = sum(1 for kk in range(K) for j in range(N) if B_pruned[kk][j] == 0)
    dut._log.info(
        "Pruned weight matrix is %.1f%% zero (2:4 contract upper bound 50%%)",
        100.0 * total / (K * N),
    )


@cocotb.test()
async def test_zero_weight_group_gates_cleanly(dut):
    """An all-zero weight column must yield an all-zero result, not X or stale data."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 3)
    await reset_dut(dut, MODE_INT8)

    A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(N)]
    B = [[0] * N for _ in range(K)]
    B_pruned, comp = prune_matrix_2of4(B, DATA_WIDTH)

    await clear_accumulators(dut)
    await stream_sparse(dut, MODE_INT8, A, comp, K)
    check_result(dut, [[0] * N for _ in range(N)], ctx="all-zero weights")

    dut._log.info("Fully gated array returns clean zeros (no stale accumulate)")


@cocotb.test()
async def test_back_to_back_sparse_tiles(dut):
    """Consecutive tiles: feeder phase and accumulators must both re-align."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 4)
    await reset_dut(dut, MODE_INT8)

    for tile in range(6):
        A, B_pruned, comp = make_sparse_problem(rng, MODE_INT8)
        await clear_accumulators(dut)
        await stream_sparse(dut, MODE_INT8, A, comp, K)
        check_result(dut, gemm_golden(A, B_pruned), ctx=f"sparse tile[{tile}]")

    dut._log.info("Back-to-back sparse tiles passed")


# ---------------------------------------------------------------------------
# M3.6 power-characterisation stimulus
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_power_stimulus(dut):
    """Long run at a controlled ACTIVATION sparsity, for VCD -> SAIF power.

    Not a pass/fail correctness test in the usual sense - it still checks every
    tile against the golden model, but its purpose is to generate realistic
    switching activity at a known sparsity level. Run it with waves enabled:

        make TEST=array_sparse WAVES=1 TB_SPARSITY=0.00
        make TEST=array_sparse WAVES=1 TB_SPARSITY=0.25
        make TEST=array_sparse WAVES=1 TB_SPARSITY=0.50
        make TEST=array_sparse WAVES=1 TB_SPARSITY=0.75

    then convert each dump with synth/fpga/vcd2saif.sh and annotate power in
    Vivado via synth/fpga/power_sweep.tcl.
    """
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 100)
    await reset_dut(dut, MODE_INT8)

    dut._log.info(
        "Power stimulus: activation sparsity = %.0f%%, %d tiles, N=%d K=%d",
        SPARSITY * 100, POWER_TILES, N, K,
    )

    zero_ops = 0
    for tile in range(POWER_TILES):
        A = make_sparse_matrix(rng, N, K, SPARSITY)
        B = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(K)]
        B_pruned, comp = prune_matrix_2of4(B, DATA_WIDTH)
        zero_ops += sum(1 for row in A for v in row if v == 0)

        await clear_accumulators(dut)
        await stream_sparse(dut, MODE_INT8, A, comp, K)
        check_result(dut, gemm_golden(A, B_pruned), ctx=f"power tile[{tile}]")

    measured = 100.0 * zero_ops / (POWER_TILES * N * K)
    dut._log.info(
        "Power stimulus complete: %d tiles verified, measured activation "
        "sparsity %.1f%% (target %.0f%%)",
        POWER_TILES, measured, SPARSITY * 100,
    )
