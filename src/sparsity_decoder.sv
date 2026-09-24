//-----------------------------------------------------------------------------
// sparsity_decoder - 2:4 structured sparsity metadata decoder (COMBINATIONAL)
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 3 (Contribution 2 - 2:4 sparsity front-end)
//
// The 2:4 contract (as used by NVIDIA Ampere): in every group of four weights
// along the contraction (K) dimension, exactly TWO are non-zero. Storage and
// bandwidth therefore carry only the two surviving values plus 4 bits of
// metadata (a 2-bit position index per surviving element) instead of four full
// words - a 50% weight-memory saving at DATA_WIDTH=8.
//
// This block performs the inverse mapping at the array boundary: it SCATTERS
// the two compressed values back to their original positions within the group
// and drives an explicit zero at the two pruned positions.
//
//   nz_in  = {nz1, nz0}          two surviving values
//   meta   = {idx1, idx0}        their positions, 2 bits each
//   dense_out[idx0] = nz0
//   dense_out[idx1] = nz1
//   dense_out[other] = 0
//   nz_mask[g] = 1 where a surviving value was placed
//
// The explicit zeros are the point: they propagate into the array and trip the
// per-PE zero detection, which is what suppresses the MAC switching activity
// and gates the clock (see pe_mixed.sv, ENABLE_GATING).
//
// PURELY COMBINATIONAL BY DESIGN - no registers anywhere in this module. A
// pipeline stage here would desynchronise the systolic skew contract and would
// force a more complex controller, so `always_comb` is used throughout and the
// absence of inferred flops is checked in synthesis.
//
// MALFORMED METADATA: a legal 2:4 group has idx0 != idx1. If the two indices
// collide (illegal input), nz0 wins the position and nz1 is dropped; the
// `meta_legal` output flags the violation for the testbench/assertions rather
// than silently corrupting the stream.
//-----------------------------------------------------------------------------
module sparsity_decoder #(
    parameter int DATA_WIDTH = 8
)(
    input  logic [2*DATA_WIDTH-1:0] nz_in,      // {nz1, nz0}
    input  logic [3:0]              meta,       // {idx1, idx0}
    output logic [4*DATA_WIDTH-1:0] dense_out,  // expanded group of four
    output logic [3:0]              nz_mask,    // 1 = position holds a survivor
    output logic                    meta_legal  // 0 = idx0 == idx1 (illegal 2:4)
);

    logic [DATA_WIDTH-1:0] nz0, nz1;
    logic [1:0]            idx0, idx1;

    assign nz0  = nz_in[0 +: DATA_WIDTH];
    assign nz1  = nz_in[DATA_WIDTH +: DATA_WIDTH];
    assign idx0 = meta[1:0];
    assign idx1 = meta[3:2];

    assign meta_legal = (idx0 != idx1);

    // Two 1-of-4 scatters sharing one comparator per position: for each of the
    // four slots, emit nz0 if it is idx0's target, else nz1 if it is idx1's,
    // else zero. Flattened output keeps the port list tool-portable.
    genvar g;
    generate
        for (g = 0; g < 4; g++) begin : g_slot
            logic hit0, hit1;
            assign hit0 = (idx0 == 2'(g));
            assign hit1 = (idx1 == 2'(g));

            assign dense_out[g*DATA_WIDTH +: DATA_WIDTH] =
                       hit0 ? nz0 :
                       hit1 ? nz1 : '0;

            assign nz_mask[g] = hit0 | hit1;
        end
    endgenerate

endmodule
