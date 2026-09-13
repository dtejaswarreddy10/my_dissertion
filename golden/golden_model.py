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
