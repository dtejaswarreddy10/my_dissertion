"""
cocotb testbench for rtl/pe_mixed.sv (precision-adaptive INT8 / INT4x2 PE).

Checks the PE against golden/golden_model.py using a cycle-accurate shadow of
the 2-stage pipeline. The product registered at edge N is formed from the
operands AND the mode present at edge N, so the model captures mode alongside
the operands - this is what catches a mode change that leaks into the wrong
product.

Run:  cd tb && make TEST=pe_mixed
      cd tb && make TEST=pe_mixed TB_EXHAUSTIVE=0    # skip the 65536-vector sweeps
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
    pack_int4x2,
    pe_golden,
    to_signed,
)

DATA_WIDTH = 8
ACC_WIDTH = 32
DATA_MASK = (1 << DATA_WIDTH) - 1
CLK_PERIOD_NS = 10
SEED = int(os.environ.get("TB_SEED", "1"))
EXHAUSTIVE = os.environ.get("TB_EXHAUSTIVE", "1") == "1"


class PEMixedModel:
    """Cycle-accurate shadow of pe_mixed.sv (one call == one posedge)."""

    def __init__(self):
        # operands + mode whose product is currently sitting in product_q
        self.op_a = 0
        self.op_b = 0
        self.op_mode = MODE_INT8
        self.valid_q = 0
        self.acc = 0
        self.a_out = 0
        self.b_out = 0
        self.valid_out = 0

    def step(self, mode, a_in, b_in, valid_in, acc_clear):
        # Stage 2 consumes the PREVIOUS cycle's product_q / valid_q;
        # acc_clear has priority over the accumulate (see pe_mixed.sv).
        if acc_clear:
            next_acc = 0
        elif self.valid_q:
            next_acc = pe_golden(self.op_mode, self.op_a, self.op_b, self.acc)
        else:
            next_acc = self.acc

        self.op_a = a_in
        self.op_b = b_in
        self.op_mode = mode
        self.valid_q = valid_in
        self.acc = next_acc
        self.a_out = a_in
        self.b_out = b_in
        self.valid_out = valid_in


async def reset_dut(dut, model, mode=MODE_INT8):
    dut.rst.value = 1
    dut.mode.value = int(mode)
    dut.valid_in.value = 0
    dut.acc_clear.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    model.__init__()
    model.op_mode = int(mode)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def drive_and_check(dut, model, mode, a, b, valid, acc_clear, ctx=""):
    dut.mode.value = int(mode)
    dut.a_in.value = a & DATA_MASK
    dut.b_in.value = b & DATA_MASK
    dut.valid_in.value = int(valid)
    dut.acc_clear.value = int(acc_clear)

    await RisingEdge(dut.clk)
    model.step(int(mode), a & DATA_MASK, b & DATA_MASK, int(valid), int(acc_clear))
    await FallingEdge(dut.clk)

    got = to_signed(int(dut.acc_result.value), ACC_WIDTH)
    assert got == model.acc, (
        f"{ctx} acc mismatch: RTL={got} golden={model.acc} "
        f"(mode={mode}, a=0x{a & DATA_MASK:02x}, b=0x{b & DATA_MASK:02x}, "
        f"valid={valid}, clear={acc_clear})"
    )
    assert int(dut.a_out.value) == model.a_out, f"{ctx} a_out forwarding mismatch"
    assert int(dut.b_out.value) == model.b_out, f"{ctx} b_out forwarding mismatch"
    assert int(dut.valid_out.value) == model.valid_out, f"{ctx} valid_out mismatch"


# ---------------------------------------------------------------------------
# INT8 mode - must remain bit-identical to the pe_int8 baseline behaviour
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_int8_corners(dut):
    """INT8 boundary operands: 0, +-1, +127, -128 in every combination."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT8)

    corners = [0, 1, -1, 127, -128]
    for a in corners:
        for b in corners:
            await drive_and_check(dut, model, MODE_INT8, 0, 0, 0, 1, ctx="clear")
            for _ in range(4):
                await drive_and_check(dut, model, MODE_INT8, a, b, 1, 0, ctx="int8 corner")
            await drive_and_check(dut, model, MODE_INT8, 0, 0, 0, 0, ctx="drain")

    dut._log.info("INT8 corner test passed (%d combinations)", len(corners) ** 2)


@cocotb.test()
async def test_int8_exhaustive(dut):
    """All 256x256 INT8 operand pairs against the golden model."""
    if not EXHAUSTIVE:
        dut._log.warning("TB_EXHAUSTIVE=0 - skipping INT8 exhaustive sweep")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT8)
    await drive_and_check(dut, model, MODE_INT8, 0, 0, 0, 1, ctx="clear")

    for a in range(256):
        for b in range(256):
            await drive_and_check(dut, model, MODE_INT8, a, b, 1, 0, ctx="int8 exh")

    dut._log.info("INT8 exhaustive passed: 65536 operand pairs, 0 mismatches")


@cocotb.test()
async def test_int8_random_accumulate(dut):
    """Long random INT8 accumulation (exercises accumulator wraparound)."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED)
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT8)

    await drive_and_check(dut, model, MODE_INT8, 0, 0, 0, 1, ctx="clear")
    for n in range(500):
        await drive_and_check(
            dut, model, MODE_INT8,
            rng.randint(-128, 127), rng.randint(-128, 127), 1, 0,
            ctx=f"int8 rand[{n}]",
        )
    dut._log.info("INT8 random accumulate passed; final acc = %d", model.acc)


# ---------------------------------------------------------------------------
# INT4x2 mode - two packed signed 4-bit lanes per byte
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_int4x2_corners(dut):
    """INT4 lane extremes: -8 and +7 in both lanes, plus zero lanes."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT4X2)

    lanes = [0, 1, -1, 7, -8]
    for a_hi in lanes:
        for a_lo in lanes:
            a = pack_int4x2(a_hi, a_lo)
            for b_hi in lanes:
                for b_lo in lanes:
                    b = pack_int4x2(b_hi, b_lo)
                    await drive_and_check(dut, model, MODE_INT4X2, 0, 0, 0, 1, ctx="clear")
                    await drive_and_check(
                        dut, model, MODE_INT4X2, a, b, 1, 0,
                        ctx=f"int4 corner a=({a_hi},{a_lo}) b=({b_hi},{b_lo})",
                    )
                    await drive_and_check(dut, model, MODE_INT4X2, 0, 0, 0, 0, ctx="drain")

    dut._log.info("INT4x2 corner test passed (%d combinations)", len(lanes) ** 4)


@cocotb.test()
async def test_int4x2_exhaustive(dut):
    """All 256x256 packed byte pairs == every (a_hi,a_lo,b_hi,b_lo) lane combo."""
    if not EXHAUSTIVE:
        dut._log.warning("TB_EXHAUSTIVE=0 - skipping INT4x2 exhaustive sweep")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT4X2)
    await drive_and_check(dut, model, MODE_INT4X2, 0, 0, 0, 1, ctx="clear")

    for a in range(256):
        for b in range(256):
            await drive_and_check(dut, model, MODE_INT4X2, a, b, 1, 0, ctx="int4 exh")

    dut._log.info("INT4x2 exhaustive passed: 65536 lane combinations, 0 mismatches")


@cocotb.test()
async def test_int4x2_lane_independence(dut):
    """A zero lane must contribute nothing - proves the lanes do not bleed."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 3)
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT4X2)

    for n in range(200):
        hi, lo = rng.randint(-8, 7), rng.randint(-8, 7)

        # Only the low lane is active.
        await drive_and_check(dut, model, MODE_INT4X2, 0, 0, 0, 1, ctx="clear")
        await drive_and_check(
            dut, model, MODE_INT4X2,
            pack_int4x2(0, lo), pack_int4x2(rng.randint(-8, 7), hi), 1, 0,
            ctx=f"lo-only[{n}]",
        )
        await drive_and_check(dut, model, MODE_INT4X2, 0, 0, 0, 0, ctx="drain")
        assert model.acc == lo * hi, (
            f"low-lane isolation broken: acc={model.acc} expected {lo * hi}"
        )

    dut._log.info("INT4x2 lane independence passed")


# ---------------------------------------------------------------------------
# Mode transitions - the accumulator must never mix precisions within a tile
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_mode_transition_with_clear(dut):
    """Tile-granular mode switching: acc_clear at every precision change."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 5)
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT8)

    for tile in range(40):
        mode = MODE_INT8 if tile % 2 == 0 else MODE_INT4X2

        # Clear on the boundary, then stream a short tile in this precision.
        await drive_and_check(dut, model, mode, 0, 0, 0, 1, ctx=f"tile{tile} clear")
        expected = 0
        for _ in range(8):
            if mode == MODE_INT8:
                a, b = rng.randint(-128, 127), rng.randint(-128, 127)
                expected += a * b
            else:
                a_hi, a_lo = rng.randint(-8, 7), rng.randint(-8, 7)
                b_hi, b_lo = rng.randint(-8, 7), rng.randint(-8, 7)
                expected += a_hi * b_hi + a_lo * b_lo
                a, b = pack_int4x2(a_hi, a_lo), pack_int4x2(b_hi, b_lo)
            await drive_and_check(dut, model, mode, a, b, 1, 0, ctx=f"tile{tile}")

        await drive_and_check(dut, model, mode, 0, 0, 0, 0, ctx=f"tile{tile} drain")
        got = to_signed(int(dut.acc_result.value), ACC_WIDTH)
        assert got == expected, (
            f"tile {tile} (mode={mode}) result {got} != independently computed {expected}"
        )

    dut._log.info("Mode-transition test passed over 40 alternating tiles")


@cocotb.test()
async def test_mode_change_does_not_corrupt_inflight(dut):
    """A mode change must apply to new operands only, not to the in-flight product."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    rng = random.Random(SEED + 11)
    model = PEMixedModel()
    await reset_dut(dut, model, MODE_INT8)

    # Randomly toggle mode while streaming; the shadow model tracks which mode
    # each in-flight product was formed under, so any pipeline leak shows up.
    await drive_and_check(dut, model, MODE_INT8, 0, 0, 0, 1, ctx="clear")
    for n in range(300):
        mode = rng.choice([MODE_INT8, MODE_INT4X2])
        a, b = rng.randint(0, 255), rng.randint(0, 255)
        valid = rng.random() < 0.8
        clear = rng.random() < 0.05
        await drive_and_check(dut, model, mode, a, b, valid, clear, ctx=f"churn[{n}]")

    dut._log.info("In-flight mode-change test passed")
