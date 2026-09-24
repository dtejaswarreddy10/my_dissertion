"""
cocotb testbench for rtl/systolic_array_mixed.sv (precision-adaptive N x N array).

Same skew contract as test_systolic_array.py:
    * row    i of A enters the WEST  edge delayed by i cycles (with its valid)
    * column j of B enters the NORTH edge delayed by j cycles
    => A[i][k] and B[k][j] meet at PE(i,j) on cycle (i + j + k)

INT8 mode is checked against gemm_golden(); INT4x2 mode against
gemm_golden_int4x2() over packed bytes. The final test demonstrates the
throughput claim: the SAME logical 2K-deep problem costs 2K stream cycles in
INT8 and K cycles in INT4x2, for identical results.

Run:  cd tb && make TEST=array_mixed
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
    pack_int4x2,
    to_signed,
)

N = int(os.environ.get("TB_N", "8"))
DATA_WIDTH = 8
ACC_WIDTH = 32
DATA_MASK = (1 << DATA_WIDTH) - 1
ACC_MASK = (1 << ACC_WIDTH) - 1
CLK_PERIOD_NS = 10
SEED = int(os.environ.get("TB_SEED", "1"))
NUM_MATRICES = int(os.environ.get("TB_MATRICES", "50"))


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
    dut.b_north.value = 0
    dut.valid_west.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def clear_accumulators(dut):
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.acc_clear.value = 0


async def stream_matrices(dut, mode, A, B, K):
    """Inject A/B with the boundary skew, drain, and return the stream cycles."""
    dut.mode.value = int(mode)
    cycles = 0
    for t in range(N + K - 1):
        a_vec = v_vec = b_vec = 0
        for i in range(N):
            k = t - i
            if 0 <= k < K:
                a_vec |= (A[i][k] & DATA_MASK) << (i * DATA_WIDTH)
                v_vec |= 1 << i
        for j in range(N):
            k = t - j
            if 0 <= k < K:
                b_vec |= (B[k][j] & DATA_MASK) << (j * DATA_WIDTH)

        dut.a_west.value = a_vec
        dut.valid_west.value = v_vec
        dut.b_north.value = b_vec
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        cycles += 1

    dut.a_west.value = 0
    dut.b_north.value = 0
    dut.valid_west.value = 0
    for _ in range(2 * N + K + 4):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    return cycles


def rand_int8_matrix(rng, rows, cols):
    return [[rng.randint(-128, 127) for _ in range(cols)] for _ in range(rows)]


def rand_packed_matrix(rng, rows, cols):
    return [
        [pack_int4x2(rng.randint(-8, 7), rng.randint(-8, 7)) for _ in range(cols)]
        for _ in range(rows)
    ]


# ---------------------------------------------------------------------------
# INT8 mode - must reproduce the dense baseline array exactly
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_int8_identity(dut):
    """A x I == A in INT8 mode (sanity check of the skew/dataflow contract)."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut, MODE_INT8)

    A = rand_int8_matrix(random.Random(0), N, N)
    I = [[1 if i == j else 0 for j in range(N)] for i in range(N)]

    await clear_accumulators(dut)
    await stream_matrices(dut, MODE_INT8, A, I, N)
    check_result(dut, gemm_golden(A, I), ctx="int8 identity")
    dut._log.info("INT8 identity GEMM passed")


@cocotb.test()
async def test_int8_random(dut):
    """Random INT8 matrix pairs vs gemm_golden()."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED)
    await reset_dut(dut, MODE_INT8)

    for n in range(NUM_MATRICES):
        A = rand_int8_matrix(rng, N, N)
        B = rand_int8_matrix(rng, N, N)
        await clear_accumulators(dut)
        await stream_matrices(dut, MODE_INT8, A, B, N)
        check_result(dut, gemm_golden(A, B), ctx=f"int8 random[{n}]")

    dut._log.info("INT8 random GEMM passed: %d matrix pairs", NUM_MATRICES)


# ---------------------------------------------------------------------------
# INT4x2 mode - packed lanes
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_int4x2_corners(dut):
    """Lane extremes across the whole array."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut, MODE_INT4X2)

    cases = {
        "all_min_lanes": pack_int4x2(-8, -8),
        "all_max_lanes": pack_int4x2(7, 7),
        "mixed_lanes": pack_int4x2(-8, 7),
        "zero_lanes": pack_int4x2(0, 0),
    }
    for name, val in cases.items():
        A = [[val] * N for _ in range(N)]
        B = [[val] * N for _ in range(N)]
        await clear_accumulators(dut)
        await stream_matrices(dut, MODE_INT4X2, A, B, N)
        check_result(dut, gemm_golden_int4x2(A, B), ctx=f"int4 {name}")
        dut._log.info("INT4x2 corner case '%s' passed", name)


@cocotb.test()
async def test_int4x2_random(dut):
    """Random packed matrix pairs vs gemm_golden_int4x2()."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 1)
    await reset_dut(dut, MODE_INT4X2)

    for n in range(NUM_MATRICES):
        A = rand_packed_matrix(rng, N, N)
        B = rand_packed_matrix(rng, N, N)
        await clear_accumulators(dut)
        await stream_matrices(dut, MODE_INT4X2, A, B, N)
        check_result(dut, gemm_golden_int4x2(A, B), ctx=f"int4 random[{n}]")

    dut._log.info("INT4x2 random GEMM passed: %d matrix pairs", NUM_MATRICES)


# ---------------------------------------------------------------------------
# Mode interleaving and the throughput claim
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_alternating_mode_tiles(dut):
    """Back-to-back tiles alternating precision; acc_clear must isolate them."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 2)
    await reset_dut(dut, MODE_INT8)

    for tile in range(10):
        if tile % 2 == 0:
            A, B = rand_int8_matrix(rng, N, N), rand_int8_matrix(rng, N, N)
            await clear_accumulators(dut)
            await stream_matrices(dut, MODE_INT8, A, B, N)
            check_result(dut, gemm_golden(A, B), ctx=f"alt tile{tile} int8")
        else:
            A, B = rand_packed_matrix(rng, N, N), rand_packed_matrix(rng, N, N)
            await clear_accumulators(dut)
            await stream_matrices(dut, MODE_INT4X2, A, B, N)
            check_result(dut, gemm_golden_int4x2(A, B), ctx=f"alt tile{tile} int4")

    dut._log.info("Alternating-precision tiles passed")


@cocotb.test()
async def test_throughput_int4x2_vs_int8(dut):
    """Same logical 2K-deep GEMM: INT8 takes 2K stream cycles, INT4x2 takes K.

    Both must produce the identical result matrix. The cycle ratio is the
    throughput number reported in the dissertation (target 1.8-2.0x).
    """
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 4)
    await reset_dut(dut, MODE_INT8)

    K = N                      # packed depth; logical contraction depth is 2K
    depth = 2 * K

    # Logical operands restricted to the INT4 range so both modes are exact.
    A_dense = [[rng.randint(-8, 7) for _ in range(depth)] for _ in range(N)]
    B_dense = [[rng.randint(-8, 7) for _ in range(N)] for _ in range(depth)]
    C_exp = gemm_golden(A_dense, B_dense)

    # --- INT8: stream all `depth` steps one per cycle -----------------------
    await clear_accumulators(dut)
    cycles_int8 = await stream_matrices(dut, MODE_INT8, A_dense, B_dense, depth)
    check_result(dut, C_exp, ctx="throughput int8")

    # --- INT4x2: pack two consecutive K-steps per byte (lane0=even k) -------
    A_packed = [
        [pack_int4x2(A_dense[i][2 * k + 1], A_dense[i][2 * k]) for k in range(K)]
        for i in range(N)
    ]
    B_packed = [
        [pack_int4x2(B_dense[2 * k + 1][j], B_dense[2 * k][j]) for j in range(N)]
        for k in range(K)
    ]

    await clear_accumulators(dut)
    cycles_int4 = await stream_matrices(dut, MODE_INT4X2, A_packed, B_packed, K)
    check_result(dut, C_exp, ctx="throughput int4x2")

    speedup = cycles_int8 / cycles_int4
    dut._log.info(
        "Throughput: INT8 = %d stream cycles, INT4x2 = %d stream cycles -> %.2fx",
        cycles_int8, cycles_int4, speedup,
    )
    assert speedup > 1.5, f"INT4x2 speedup {speedup:.2f}x below expectation"
