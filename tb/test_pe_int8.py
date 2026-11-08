"""
cocotb testbench for rtl/pe_int8.sv (baseline INT8 MAC processing element).

Checks the PE against golden/golden_model.py:pe_golden_int8() using a
cycle-accurate shadow model of the 2-stage pipeline:

    stage 1 : product_q <= a_in * b_in ;  valid_q <= valid_in
    stage 2 : acc_clear ? 0 : (valid_q ? acc_q + product_q : acc_q)

Run:  cd tb && make TEST=pe
"""

import os
import pathlib
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "golden"))
from golden_model import pe_golden_int8, to_signed  # noqa: E402

DATA_WIDTH = 8
ACC_WIDTH = 32
DATA_MASK = (1 << DATA_WIDTH) - 1
CLK_PERIOD_NS = 10  # 100 MHz
SEED = int(os.environ.get("TB_SEED", "1"))


class PEModel:
    """Cycle-accurate shadow of pe_int8.sv (one call == one posedge)."""

    def __init__(self):
        # operands whose product is currently sitting in product_q
        self.op_a = 0
        self.op_b = 0
        self.valid_q = 0
        self.acc = 0
        self.a_out = 0
        self.b_out = 0
        self.valid_out = 0

    def step(self, a_in, b_in, valid_in, acc_clear):
        # The accumulator update uses the PREVIOUS cycle's product_q / valid_q,
        # and acc_clear has priority over the accumulate (see pe_int8.sv).
        if acc_clear:
            next_acc = 0
        elif self.valid_q:
            next_acc = pe_golden_int8(self.op_a, self.op_b, self.acc)
        else:
            next_acc = self.acc

        self.op_a = a_in
        self.op_b = b_in
        self.valid_q = valid_in
        self.acc = next_acc
        self.a_out = a_in
        self.b_out = b_in
        self.valid_out = valid_in


def read_signed(sig, width):
    return to_signed(int(sig.value), width)


async def reset_dut(dut, model):
    dut.rst.value = 1
    dut.valid_in.value = 0
    dut.acc_clear.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    model.__init__()
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def drive_and_check(dut, model, a, b, valid, acc_clear, ctx=""):
    dut.a_in.value = a & DATA_MASK
    dut.b_in.value = b & DATA_MASK
    dut.valid_in.value = int(valid)
    dut.acc_clear.value = int(acc_clear)

    await RisingEdge(dut.clk)
    model.step(a & DATA_MASK, b & DATA_MASK, int(valid), int(acc_clear))
    await FallingEdge(dut.clk)

    # TEMP DEBUG - remove once the Questa compat-flow timing question is
    # settled. Reads pe_int8's internal stage-1 registers directly (visible
    # because vlog compiled with +acc) to check whether product_q/valid_q
    # themselves update on the schedule pe_int8.sv documents, independent of
    # whatever cocotb/VPI is doing on the input-driving side.
    dut._log.info(
        f"{ctx}: product_q={read_signed(dut.product_q, ACC_WIDTH)} "
        f"valid_q={int(dut.valid_q.value)} acc_q={read_signed(dut.acc_q, ACC_WIDTH)}"
    )

    got = read_signed(dut.acc_result, ACC_WIDTH)
    assert got == model.acc, (
        f"{ctx} acc_result mismatch: RTL={got} golden={model.acc} "
        f"(a={a}, b={b}, valid={valid}, acc_clear={acc_clear})"
    )
    assert int(dut.a_out.value) == model.a_out, f"{ctx} a_out forwarding mismatch"
    assert int(dut.b_out.value) == model.b_out, f"{ctx} b_out forwarding mismatch"
    assert int(dut.valid_out.value) == model.valid_out, f"{ctx} valid_out mismatch"


@cocotb.test()
async def test_pe_directed_corners(dut):
    """Boundary operands: 0, +127, -128, +/-1 in every combination."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    model = PEModel()
    await reset_dut(dut, model)

    corners = [0, 1, -1, 127, -128]
    for a in corners:
        for b in corners:
            await drive_and_check(dut, model, 0, 0, 0, 1, ctx="clear")
            for _ in range(4):  # accumulate the same product repeatedly
                await drive_and_check(dut, model, a, b, 1, 0, ctx="corner")
            await drive_and_check(dut, model, 0, 0, 0, 0, ctx="drain")

    dut._log.info("Directed corner test passed (%d combinations)", len(corners) ** 2)


@cocotb.test()
async def test_pe_random_mac(dut):
    """100 random operand pairs accumulated in place, vs pe_golden_int8()."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    random.seed(SEED)
    model = PEModel()
    await reset_dut(dut, model)

    await drive_and_check(dut, model, 0, 0, 0, 1, ctx="clear")
    for n in range(100):
        a = random.randint(-128, 127)
        b = random.randint(-128, 127)
        await drive_and_check(dut, model, a, b, 1, 0, ctx=f"rand[{n}]")

    # Independent end-to-end check against the golden model.
    await drive_and_check(dut, model, 0, 0, 0, 0, ctx="drain")
    dut._log.info("Random MAC test passed; final acc = %d", model.acc)


@cocotb.test()
async def test_pe_valid_gating_and_clear(dut):
    """valid_in=0 must not accumulate; acc_clear must zero the accumulator."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    random.seed(SEED + 1)
    model = PEModel()
    await reset_dut(dut, model)

    await drive_and_check(dut, model, 0, 0, 0, 1, ctx="clear")
    for n in range(200):
        a = random.randint(-128, 127)
        b = random.randint(-128, 127)
        valid = random.random() < 0.6
        clear = random.random() < 0.05
        await drive_and_check(dut, model, a, b, valid, clear, ctx=f"mixed[{n}]")

    await drive_and_check(dut, model, 0, 0, 0, 1, ctx="final clear")
    assert model.acc == 0, "acc_clear did not zero the accumulator"
    dut._log.info("valid gating / acc_clear test passed")

# ---------------------------------------------------------------------------
# Power stimulus - controlled INT8 operand sparsity for SAIF estimation
# ---------------------------------------------------------------------------

POWER_SPARSITY = float(os.environ.get("TB_SPARSITY", "0.50"))
POWER_IDLE = float(os.environ.get("TB_IDLE", "0.0"))
POWER_CYCLES = int(os.environ.get("TB_POWER_CYCLES", "4000"))


@cocotb.test()
async def test_power_stimulus(dut):
    """
    INT8 power workload with controlled A-input zero sparsity.

    Defaults:
      TB_SPARSITY      = 0.50
      TB_IDLE          = 0.0
      TB_POWER_CYCLES  = 4000
    """
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    model = PEModel()
    await reset_dut(dut, model)

    rng = random.Random(SEED + 100)

    dut._log.info(
        "INT8 power stimulus: sparsity=%.0f%% idle=%.0f%% cycles=%d",
        POWER_SPARSITY * 100,
        POWER_IDLE * 100,
        POWER_CYCLES,
    )

    n_valid = 0
    n_zero = 0

    for cyc in range(POWER_CYCLES):
        clear = 1 if cyc % 32 == 31 else 0

        if rng.random() < POWER_IDLE:
            await drive_and_check(
                dut, model, 0, 0, 0, clear, ctx="power idle"
            )
            continue

        a = 0 if rng.random() < POWER_SPARSITY else rng.randint(1, 255)
        b = rng.randint(1, 255)

        n_valid += 1
        n_zero += int(a == 0)

        await drive_and_check(
            dut, model, a, b, 1, clear, ctx="power"
        )

    measured = 100.0 * n_zero / max(n_valid, 1)

    dut._log.info(
        "INT8 power stimulus complete: %d valid cycles, "
        "measured A-zero sparsity %.1f%% (target %.0f%%)",
        n_valid,
        measured,
        POWER_SPARSITY * 100,
    )


@cocotb.test()
async def test_power_capture(dut):
    """INT8 gate-level power capture without functional assertions."""

    cocotb.start_soon(
        Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start()
    )

    dut.rst.value = 1
    dut.valid_in.value = 0
    dut.acc_clear.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0

    for _ in range(2):
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.rst.value = 0

    rng = random.Random(SEED + 100)

    n_valid = 0
    n_zero = 0
    n_idle = 0

    dut._log.info(
        "INT8 power capture: sparsity=%.0f%% idle=%.0f%% cycles=%d",
        POWER_SPARSITY * 100,
        POWER_IDLE * 100,
        POWER_CYCLES,
    )

    for cyc in range(POWER_CYCLES):
        clear = 1 if cyc % 32 == 31 else 0

        if rng.random() < POWER_IDLE:
            a, b, valid = 0, 0, 0
            n_idle += 1
        else:
            a = 0 if rng.random() < POWER_SPARSITY else rng.randint(1, 255)
            b = rng.randint(1, 255)
            valid = 1
            n_valid += 1
            n_zero += int(a == 0)

        dut.a_in.value = a
        dut.b_in.value = b
        dut.valid_in.value = valid
        dut.acc_clear.value = clear

        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)

    measured = 100.0 * n_zero / max(n_valid, 1)

    dut._log.info(
        "INT8 power capture complete: cycles=%d valid=%d idle=%d "
        "zero_operands=%d measured_sparsity=%.2f%%",
        POWER_CYCLES, n_valid, n_idle, n_zero, measured,
    )
