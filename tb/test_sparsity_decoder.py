"""
cocotb testbench for rtl/sparsity_decoder.sv (2:4 metadata decoder).

The DUT is purely combinational, so there is no clock here - inputs are driven,
a delta is allowed to settle, and outputs are compared against
golden/golden_model.py:expand_group_2of4().

Coverage of the metadata space is EXHAUSTIVE: all 16 encodings of
meta = {idx1, idx0}, including the 4 illegal ones where the indices collide.

Run:  cd tb && make TEST=decoder
"""

import os
import pathlib
import random
import sys

import cocotb
from cocotb.triggers import Timer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "golden"))
from golden_model import expand_group_2of4, pack_nz_pair, to_signed  # noqa: E402

DATA_WIDTH = 8
DATA_MASK = (1 << DATA_WIDTH) - 1
SETTLE_NS = 1
SEED = int(os.environ.get("TB_SEED", "1"))


def read_dense(dut):
    """Unpack the flattened dense_out bus into four signed values."""
    raw = int(dut.dense_out.value)
    return [
        to_signed((raw >> (g * DATA_WIDTH)) & DATA_MASK, DATA_WIDTH)
        for g in range(4)
    ]


async def apply(dut, nz0, nz1, meta):
    dut.nz_in.value = pack_nz_pair(nz0, nz1, DATA_WIDTH)
    dut.meta.value = meta & 0xF
    await Timer(SETTLE_NS, units="ns")


def check(dut, nz0, nz1, meta, ctx=""):
    idx0 = meta & 0x3
    idx1 = (meta >> 2) & 0x3

    exp_dense = expand_group_2of4(nz0, nz1, meta, DATA_WIDTH)
    got_dense = read_dense(dut)
    assert got_dense == exp_dense, (
        f"{ctx} dense_out mismatch: RTL={got_dense} golden={exp_dense} "
        f"(nz0={nz0}, nz1={nz1}, meta=0x{meta:x})"
    )

    exp_mask = 0
    for g in range(4):
        if idx0 == g or idx1 == g:
            exp_mask |= 1 << g
    got_mask = int(dut.nz_mask.value)
    assert got_mask == exp_mask, (
        f"{ctx} nz_mask mismatch: RTL=0b{got_mask:04b} golden=0b{exp_mask:04b}"
    )

    exp_legal = int(idx0 != idx1)
    got_legal = int(dut.meta_legal.value)
    assert got_legal == exp_legal, (
        f"{ctx} meta_legal mismatch: RTL={got_legal} expected={exp_legal}"
    )


@cocotb.test()
async def test_metadata_exhaustive(dut):
    """All 16 metadata encodings x representative operand values."""
    values = [0, 1, -1, 127, -128, 42, -99]

    n = 0
    for meta in range(16):
        for nz0 in values:
            for nz1 in values:
                await apply(dut, nz0, nz1, meta)
                check(dut, nz0, nz1, meta, ctx=f"meta=0x{meta:x}")
                n += 1

    dut._log.info("Exhaustive metadata sweep passed: %d vectors, 16/16 encodings", n)


@cocotb.test()
async def test_legal_2of4_placement(dut):
    """For every legal group, exactly two slots are non-pruned and correctly placed."""
    rng = random.Random(SEED)

    legal = [(i0, i1) for i0 in range(4) for i1 in range(4) if i0 != i1]
    assert len(legal) == 12

    for idx0, idx1 in legal:
        for _ in range(20):
            # Non-zero payloads so a dropped survivor cannot hide as a zero.
            nz0 = rng.choice([v for v in range(-128, 128) if v != 0])
            nz1 = rng.choice([v for v in range(-128, 128) if v != 0])
            meta = (idx1 << 2) | idx0

            await apply(dut, nz0, nz1, meta)
            check(dut, nz0, nz1, meta, ctx=f"legal({idx0},{idx1})")

            dense = read_dense(dut)
            assert dense[idx0] == nz0, f"survivor 0 misplaced: {dense} idx0={idx0}"
            assert dense[idx1] == nz1, f"survivor 1 misplaced: {dense} idx1={idx1}"

            pruned = [g for g in range(4) if g not in (idx0, idx1)]
            for g in pruned:
                assert dense[g] == 0, (
                    f"pruned slot {g} must be an explicit zero, got {dense[g]}"
                )
            assert int(dut.meta_legal.value) == 1

    dut._log.info("Legal 2:4 placement verified for all 12 index pairs")


@cocotb.test()
async def test_illegal_metadata_flagged(dut):
    """Colliding indices must assert meta_legal=0 and let nz0 win the slot."""
    for idx in range(4):
        meta = (idx << 2) | idx
        await apply(dut, 55, -77, meta)
        assert int(dut.meta_legal.value) == 0, (
            f"meta=0x{meta:x} has idx0==idx1 and must be flagged illegal"
        )
        dense = read_dense(dut)
        assert dense[idx] == 55, "nz0 must win a collided slot"
        assert sum(1 for v in dense if v != 0) == 1

    dut._log.info("Illegal metadata correctly flagged for all 4 collisions")


@cocotb.test()
async def test_sparsity_rate_is_exactly_two_of_four(dut):
    """The 2:4 contract itself: legal metadata always yields exactly 2 survivors."""
    rng = random.Random(SEED + 1)

    for _ in range(500):
        idx0, idx1 = rng.sample(range(4), 2)
        meta = (idx1 << 2) | idx0
        nz0 = rng.choice([v for v in range(-128, 128) if v != 0])
        nz1 = rng.choice([v for v in range(-128, 128) if v != 0])

        await apply(dut, nz0, nz1, meta)
        assert bin(int(dut.nz_mask.value)).count("1") == 2, (
            "a legal 2:4 group must expose exactly two survivors"
        )
        assert sum(1 for v in read_dense(dut) if v != 0) == 2

    dut._log.info("2:4 density contract holds over 500 random legal groups")
