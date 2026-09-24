"""
Golden reference models for the baseline INT8 systolic array.

Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
Stage   : Month 1 / Weeks 2-3 deliverable

These are pure-Python, bit-accurate references for:
    rtl/pe_int8.sv         -> pe_golden_int8()
    rtl/systolic_array.sv  -> gemm_golden()

Modeled hardware semantics:
    * signed DATA_WIDTH operands (default 8-bit, range -128..127)
    * full-precision signed product (a * b)
    * ACC_WIDTH (default 32-bit) two's-complement accumulator with
      wraparound, matching `acc_result <= acc_result + product`
      (the PE does NOT saturate).

For the 8x8 INT8 case the products sum to at most 8*127*128 = 130048, which is
far inside int32, so wraparound never actually triggers - but it is modeled so
the reference stays exact if WIDTH/N are changed later.

No simulator or external setup is required to use this module; it is the
golden side that a (separately owned) testbench compares the RTL against.
"""

DATA_WIDTH = 8
ACC_WIDTH = 32


def to_signed(value, width):
    """Interpret the low `width` bits of `value` as two's complement."""
    mask = (1 << width) - 1
    value &= mask
    if value & (1 << (width - 1)):
        value -= (1 << width)
    return value


def pe_golden_int8(a, b, acc_in, data_width=DATA_WIDTH, acc_width=ACC_WIDTH):
    """One PE MAC step.

    acc_out = wrap_to_int(acc_in + signed(a) * signed(b))

    Args:
        a, b   : operands (interpreted as signed `data_width`-bit).
        acc_in : current accumulator (interpreted as signed `acc_width`-bit).
    Returns:
        acc_out as a signed Python int in the `acc_width` range.
    """
    a_s = to_signed(int(a), data_width)
    b_s = to_signed(int(b), data_width)
    acc_s = to_signed(int(acc_in), acc_width)
    return to_signed(acc_s + a_s * b_s, acc_width)


def gemm_golden(A, B, data_width=DATA_WIDTH, acc_width=ACC_WIDTH):
    """Signed integer GEMM with the PE's accumulate semantics.

    Computes C = A * B where A is (M x K) and B is (K x N). Each output element
    is accumulated exactly as the hardware does it (per-step two's-complement
    wrap in the `acc_width`-bit accumulator).

    Args:
        A : iterable of rows, each row an iterable of ints (M x K).
        B : iterable of rows, each row an iterable of ints (K x N).
    Returns:
        C : list of lists of signed Python ints (M x N).
    """
    A = [[to_signed(int(x), data_width) for x in row] for row in A]
    B = [[to_signed(int(x), data_width) for x in row] for row in B]

    m = len(A)
    k = len(A[0]) if m else 0
    k_b = len(B)
    n = len(B[0]) if k_b else 0
    assert k == k_b, "inner dimensions must match: A is {}x{}, B is {}x{}".format(m, k, k_b, n)

    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for jj in range(n):
            acc = 0
            for kk in range(k):
                acc = to_signed(acc + A[i][kk] * B[kk][jj], acc_width)
            C[i][jj] = acc
    return C


# ---------------------------------------------------------------------------
# Month 2: precision-adaptive PE (INT8 / INT4x2)
# ---------------------------------------------------------------------------
# Packing contract (locked - rtl/pe_mixed.sv implements exactly this):
#
#   MODE_INT8   : a_in[7:0] is one signed 8-bit operand.
#                 acc += a * b                       -> 1 MAC / cycle
#
#   MODE_INT4X2 : a_in[7:4] = lane 1 (signed 4-bit, range -8..+7)
#                 a_in[3:0] = lane 0 (signed 4-bit, range -8..+7)
#                 acc += a_hi*b_hi + a_lo*b_lo       -> 2 MACs / cycle
#
# The 2x throughput comes from packing two K-steps into one byte: lane 0 holds
# the even k index, lane 1 the odd one, so a K-deep dot product costs K/2 cycles.

MODE_INT8 = 0
MODE_INT4X2 = 1

NIBBLE_WIDTH = 4


def pack_int4x2(hi, lo):
    """Pack two signed 4-bit lanes into one byte (hi = upper nibble)."""
    assert -8 <= hi <= 7, "hi lane out of signed 4-bit range: {}".format(hi)
    assert -8 <= lo <= 7, "lo lane out of signed 4-bit range: {}".format(lo)
    return ((hi & 0xF) << 4) | (lo & 0xF)


def unpack_int4x2(byte):
    """Split a byte into (hi, lo) signed 4-bit lanes."""
    byte = int(byte) & 0xFF
    hi = to_signed(byte >> 4, NIBBLE_WIDTH)
    lo = to_signed(byte, NIBBLE_WIDTH)
    return hi, lo


def pe_golden_int4x2(a, b, acc_in, acc_width=ACC_WIDTH):
    """One PE step in INT4x2 mode: acc_out = acc_in + a_hi*b_hi + a_lo*b_lo."""
    a_hi, a_lo = unpack_int4x2(a)
    b_hi, b_lo = unpack_int4x2(b)
    acc_s = to_signed(int(acc_in), acc_width)
    return to_signed(acc_s + a_hi * b_hi + a_lo * b_lo, acc_width)


def pe_golden(mode, a, b, acc_in, data_width=DATA_WIDTH, acc_width=ACC_WIDTH):
    """Mode-dispatching PE reference (matches rtl/pe_mixed.sv)."""
    if mode == MODE_INT8:
        return pe_golden_int8(a, b, acc_in, data_width, acc_width)
    if mode == MODE_INT4X2:
        return pe_golden_int4x2(a, b, acc_in, acc_width)
    raise ValueError("unknown mode: {}".format(mode))


def gemm_golden_int4x2(A, B, acc_width=ACC_WIDTH):
    """INT4x2 GEMM over packed bytes.

    A is (M x K) and B is (K x N) of PACKED bytes; each byte carries two 4-bit
    lanes, so the logical contraction depth is 2K while only K cycles of
    operands are streamed. Returns C (M x N) with the hardware's accumulate
    semantics.
    """
    m = len(A)
    k = len(A[0]) if m else 0
    assert len(B) == k, "inner dimensions must match: A is {}x{}, B has {} rows".format(m, k, len(B))
    n = len(B[0]) if k else 0

    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for jj in range(n):
            acc = 0
            for kk in range(k):
                acc = pe_golden_int4x2(A[i][kk], B[kk][jj], acc, acc_width)
            C[i][jj] = acc
    return C


def gemm_golden_mode(mode, A, B, data_width=DATA_WIDTH, acc_width=ACC_WIDTH):
    """Mode-dispatching array-level reference."""
    if mode == MODE_INT8:
        return gemm_golden(A, B, data_width, acc_width)
    if mode == MODE_INT4X2:
        return gemm_golden_int4x2(A, B, acc_width)
    raise ValueError("unknown mode: {}".format(mode))


# ---------------------------------------------------------------------------
# Month 3: 2:4 structured sparsity
# ---------------------------------------------------------------------------
# Contract (NVIDIA Ampere style): in every group of FOUR weights along the
# contraction dimension K, exactly TWO survive. Storage carries the two
# survivors plus a 4-bit metadata word holding their positions, so weight
# bandwidth halves. rtl/sparsity_decoder.sv scatters them back and drives an
# explicit zero at the two pruned positions.
#
# Metadata layout matches the RTL exactly:  meta = {idx1[1:0], idx0[1:0]}

GROUP_SIZE = 4
NNZ_PER_GROUP = 2


def prune_group_2of4(group):
    """Keep the two largest-magnitude values of a 4-element group.

    Returns (nz0, nz1, idx0, idx1) with idx0 < idx1, so the metadata is
    canonical and reproducible. Ties break toward the lower index.
    """
    assert len(group) == GROUP_SIZE, "2:4 groups must have exactly 4 elements"
    # Sort by (-|value|, index) so magnitude wins and ties go to the lower index.
    ranked = sorted(range(GROUP_SIZE), key=lambda g: (-abs(group[g]), g))
    keep = sorted(ranked[:NNZ_PER_GROUP])
    idx0, idx1 = keep[0], keep[1]
    return group[idx0], group[idx1], idx0, idx1


def pack_meta_2of4(idx0, idx1):
    """Build the 4-bit metadata word {idx1, idx0} used by the RTL."""
    assert 0 <= idx0 < GROUP_SIZE and 0 <= idx1 < GROUP_SIZE
    return ((idx1 & 0x3) << 2) | (idx0 & 0x3)


def pack_nz_pair(nz0, nz1, data_width=DATA_WIDTH):
    """Build the {nz1, nz0} compressed operand word used by the RTL."""
    mask = (1 << data_width) - 1
    return ((int(nz1) & mask) << data_width) | (int(nz0) & mask)


def expand_group_2of4(nz0, nz1, meta, data_width=DATA_WIDTH):
    """Inverse of the packing: scatter two survivors back into four slots.

    Mirrors rtl/sparsity_decoder.sv bit for bit, including the malformed-input
    behaviour where a collided index lets nz0 win the slot.
    """
    idx0 = meta & 0x3
    idx1 = (meta >> 2) & 0x3
    out = [0] * GROUP_SIZE
    for g in range(GROUP_SIZE):
        if idx0 == g:
            out[g] = to_signed(nz0, data_width)
        elif idx1 == g:
            out[g] = to_signed(nz1, data_width)
        else:
            out[g] = 0
    return out


def prune_column_2of4(col, data_width=DATA_WIDTH):
    """Prune one length-K column to 2:4. K must be a multiple of 4.

    Returns (dense_pruned, compressed) where dense_pruned is the length-K list
    the hardware actually multiplies (half of it zeros), and compressed is a
    list of (nz_word, meta) tuples - one per group of four - in the exact form
    the RTL ports expect.
    """
    k = len(col)
    assert k % GROUP_SIZE == 0, "K must be a multiple of 4 for 2:4 sparsity"

    dense = []
    compressed = []
    for base in range(0, k, GROUP_SIZE):
        group = col[base:base + GROUP_SIZE]
        nz0, nz1, idx0, idx1 = prune_group_2of4(group)
        meta = pack_meta_2of4(idx0, idx1)
        compressed.append((pack_nz_pair(nz0, nz1, data_width), meta))
        dense.extend(expand_group_2of4(nz0, nz1, meta, data_width))
    return dense, compressed


def prune_matrix_2of4(B, data_width=DATA_WIDTH):
    """Prune a (K x N) weight matrix to 2:4 along K.

    Returns (B_pruned, compressed) where:
      B_pruned[k][j]   - the dense post-pruning matrix (the mathematical
                         reference the array must reproduce)
      compressed[g][j] - (nz_word, meta) for column j, group g
    """
    k = len(B)
    n = len(B[0]) if k else 0
    assert k % GROUP_SIZE == 0, "K must be a multiple of 4 for 2:4 sparsity"

    B_pruned = [[0] * n for _ in range(k)]
    compressed = [[None] * n for _ in range(k // GROUP_SIZE)]

    for j in range(n):
        col = [to_signed(B[kk][j], data_width) for kk in range(k)]
        dense, comp = prune_column_2of4(col, data_width)
        for kk in range(k):
            B_pruned[kk][j] = dense[kk]
        for g, entry in enumerate(comp):
            compressed[g][j] = entry

    return B_pruned, compressed


def make_sparse_matrix(rng, rows, cols, sparsity, lo=-128, hi=127):
    """Random matrix with a target fraction of exact zeros.

    Used to sweep the power-vs-sparsity curve at 0 / 25 / 50 / 75%. This is
    unstructured zeroing - it models activation sparsity feeding the PE zero
    detection, as distinct from the structured 2:4 weight pruning above.
    """
    assert 0.0 <= sparsity <= 1.0
    return [
        [0 if rng.random() < sparsity else rng.randint(lo, hi) for _ in range(cols)]
        for _ in range(rows)
    ]


if __name__ == "__main__":
    # Minimal usage example (sanity-check the reference itself, no RTL involved).
    import random

    # Single-PE accumulate example.
    acc = 0
    for _ in range(4):
        a = random.randint(-128, 127)
        b = random.randint(-128, 127)
        acc = pe_golden_int8(a, b, acc)
    print("pe_golden_int8 running accumulate ->", acc)

    # Small GEMM example, cross-checked against a plain integer multiply.
    N = 8
    A = [[random.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    B = [[random.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    C = gemm_golden(A, B)
    ref = [[sum(A[i][t] * B[t][j] for t in range(N)) for j in range(N)] for i in range(N)]
    print("gemm_golden matches plain int GEMM:", C == ref)

    # --- INT4x2 -----------------------------------------------------------
    hi_a, lo_a = random.randint(-8, 7), random.randint(-8, 7)
    hi_b, lo_b = random.randint(-8, 7), random.randint(-8, 7)
    pa, pb = pack_int4x2(hi_a, lo_a), pack_int4x2(hi_b, lo_b)
    print("pack/unpack round-trip:", unpack_int4x2(pa) == (hi_a, lo_a))
    print(
        "pe_golden_int4x2 matches two independent MACs:",
        pe_golden_int4x2(pa, pb, 0) == hi_a * hi_b + lo_a * lo_b,
    )

    # An INT4x2 GEMM over K packed bytes must equal a dense 2K-deep INT4 GEMM.
    K = 4
    Ap = [[pack_int4x2(random.randint(-8, 7), random.randint(-8, 7)) for _ in range(K)]
          for _ in range(N)]
    Bp = [[pack_int4x2(random.randint(-8, 7), random.randint(-8, 7)) for _ in range(N)]
          for _ in range(K)]
    Cp = gemm_golden_int4x2(Ap, Bp)

    # Expand the packed operands into their logical 2K-deep dense form.
    A_dense = [[v for byte in row for v in reversed(unpack_int4x2(byte))] for row in Ap]
    B_dense = [[0] * N for _ in range(2 * K)]
    for kk in range(K):
        for j in range(N):
            b_hi, b_lo = unpack_int4x2(Bp[kk][j])
            B_dense[2 * kk][j] = b_lo
            B_dense[2 * kk + 1][j] = b_hi
    ref4 = [[sum(A_dense[i][t] * B_dense[t][j] for t in range(2 * K)) for j in range(N)]
            for i in range(N)]
    print("gemm_golden_int4x2 matches dense 2K-deep INT4 GEMM:", Cp == ref4)

    # --- 2:4 structured sparsity -----------------------------------------
    grp = [3, -50, 7, 120]
    nz0, nz1, i0, i1 = prune_group_2of4(grp)
    meta = pack_meta_2of4(i0, i1)
    exp = expand_group_2of4(nz0, nz1, meta)
    print("prune_group_2of4 keeps the two largest magnitudes:",
          (nz0, nz1) == (-50, 120) and exp == [0, -50, 0, 120])

    Kp = 8
    Bfull = [[random.randint(-128, 127) for _ in range(N)] for _ in range(Kp)]
    B_pruned, comp = prune_matrix_2of4(Bfull)
    ok_rate = all(
        sum(1 for kk in range(g * 4, g * 4 + 4) if B_pruned[kk][j] != 0) <= 2
        for g in range(Kp // 4) for j in range(N)
    )
    print("prune_matrix_2of4 honours the 2:4 contract:", ok_rate)

    # The RTL must reproduce GEMM against the PRUNED matrix, not the original.
    A8 = [[random.randint(-128, 127) for _ in range(Kp)] for _ in range(N)]
    print("sparse GEMM reference is well defined:",
          gemm_golden(A8, B_pruned) is not None and len(comp) == Kp // 4)


