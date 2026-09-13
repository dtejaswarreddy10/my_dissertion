"""
cocotb testbench for rtl/systolic_array.sv (N x N output-stationary INT8 array).

Drives the west/north edges with the skew required by the timing contract in
systolic_array.sv:

    * row    i of A enters the WEST  edge delayed by i cycles (with its valid)
    * column j of B enters the NORTH edge delayed by j cycles
    => A[i][k] and B[k][j] meet at PE(i,j) on cycle (i + j + k)

Results are compared against golden/golden_model.py:gemm_golden().

Run:  cd tb && make TEST=array
"""

import os
import pathlib
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "golden"))
from golden_model import gemm_golden, to_signed  # noqa: E402

N = int(os.environ.get("TB_N", "8"))            # array dimension (must match RTL)
DATA_WIDTH = 8
ACC_WIDTH = 32
DATA_MASK = (1 << DATA_WIDTH) - 1
ACC_MASK = (1 << ACC_WIDTH) - 1
CLK_PERIOD_NS = 10
SEED = int(os.environ.get("TB_SEED", "1"))
NUM_MATRICES = int(os.environ.get("TB_MATRICES", "50"))


def read_c(dut, i, j):
    """Unpack C[i][j] out of the flattened c_out bus."""
    raw = int(dut.c_out.value)
    word = (raw >> ((i * N + j) * ACC_WIDTH)) & ACC_MASK
    return to_signed(word, ACC_WIDTH)


async def reset_dut(dut):
    dut.rst.value = 1
    dut.acc_clear.value = 0
    dut.a_west.value = 0
    dut.b_north.value = 0
    dut.valid_west.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def clear_accumulators(dut):
    """Pulse acc_clear so every PE starts a fresh tile (clear-then-stream)."""
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.acc_clear.value = 0


async def stream_matrices(dut, A, B, K):
    """Inject A and B with the required boundary skew, then drain the pipeline."""
    for t in range(N + K - 1):
        a_vec = 0
        v_vec = 0
        b_vec = 0
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

    # Drain: the last product reaches PE(N-1,N-1) at cycle 2N+K-3 and is
    # accumulated one cycle later, so flush generously with valid deasserted.
    dut.a_west.value = 0
    dut.b_north.value = 0
    dut.valid_west.value = 0
    for _ in range(2 * N + K + 4):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


def check_result(dut, C_exp, ctx=""):
    mismatches = []
    for i in range(N):
        for j in range(N):
            got = read_c(dut, i, j)
            if got != C_exp[i][j]:
                mismatches.append((i, j, got, C_exp[i][j]))
    assert not mismatches, (
        f"{ctx}: {len(mismatches)} mismatch(es), first 5: "
        + ", ".join(
            f"C[{i}][{j}] RTL={g} golden={e}" for i, j, g, e in mismatches[:5]
        )
    )


@cocotb.test()
async def test_gemm_identity(dut):
    """A x I == A  (directed sanity check of the skew/dataflow contract)."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    A = [[random.Random(0).randint(-128, 127) for _ in range(N)] for _ in range(N)]
    I = [[1 if i == j else 0 for j in range(N)] for i in range(N)]

    await clear_accumulators(dut)
    await stream_matrices(dut, A, I, N)
    check_result(dut, gemm_golden(A, I), ctx="identity")
    dut._log.info("Identity GEMM passed")


@cocotb.test()
async def test_gemm_corners(dut):
    """Saturating-magnitude operands: all +127, all -128, and mixed signs."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    cases = {
        "all_max": ([[127] * N] * N, [[127] * N] * N),
        "all_min": ([[-128] * N] * N, [[-128] * N] * N),
        "zeros": ([[0] * N] * N, [[0] * N] * N),
        "mixed": (
            [[127 if (i + j) % 2 else -128 for j in range(N)] for i in range(N)],
            [[-128 if (i + j) % 2 else 127 for j in range(N)] for i in range(N)],
        ),
    }

    for name, (A, B) in cases.items():
        await clear_accumulators(dut)
        await stream_matrices(dut, A, B, N)
        check_result(dut, gemm_golden(A, B), ctx=name)
        dut._log.info("Corner case '%s' passed", name)


@cocotb.test()
async def test_gemm_random(dut):
    """Random NxN matrix pairs vs gemm_golden(). Zero mismatches required."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED)
    await reset_dut(dut)

    for n in range(NUM_MATRICES):
        A = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(N)]
        B = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(N)]

        await clear_accumulators(dut)
        await stream_matrices(dut, A, B, N)
        check_result(dut, gemm_golden(A, B), ctx=f"random[{n}]")

    dut._log.info("Random GEMM passed: %d matrix pairs, 0 mismatches", NUM_MATRICES)


@cocotb.test()
async def test_back_to_back_tiles(dut):
    """Consecutive tiles without reset: acc_clear must isolate tile results."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 7)
    await reset_dut(dut)

    for n in range(5):
        A = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(N)]
        B = [[rng.randint(-128, 127) for _ in range(N)] for _ in range(N)]
        await clear_accumulators(dut)
        await stream_matrices(dut, A, B, N)
        check_result(dut, gemm_golden(A, B), ctx=f"tile[{n}]")

    dut._log.info("Back-to-back tile isolation passed")
