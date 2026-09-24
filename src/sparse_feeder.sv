//-----------------------------------------------------------------------------
// sparse_feeder - north-edge 2:4 weight expander for one array column
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 3 (2:4 sparsity front-end integration)
//
// Sits at the array BOUNDARY, outside the systolic mesh, and turns a
// compressed 2:4 weight stream into the dense per-cycle stream the column
// expects. Because it is a boundary block, it adds no stage to the mesh and
// the skew contract of systolic_array.sv is untouched.
//
//   compressed in : {nz1, nz0} + 4-bit metadata   (2 words per 4 k-steps)
//   dense out     : one word per enabled cycle    (4 words per 4 k-steps,
//                                                  two of which are zero)
//
// The zeros it emits are the whole point: they reach the PE, trip its zero
// detection, and gate the MAC. Bandwidth into the chip halves; the power
// saving appears inside the array.
//
// SOURCE CONTRACT: grp_nz/grp_meta must stay stable for the four enabled
// cycles of a group. grp_advance strobes on the last enabled cycle of the
// group to tell the source to present the next one. `en` lets a column start
// late, which is exactly what the column-j injection skew requires.
//
// The expansion itself is the purely combinational sparsity_decoder; the only
// state here is the 2-bit phase counter that walks the group.
//-----------------------------------------------------------------------------
module sparse_feeder #(
    parameter int DATA_WIDTH = 8
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    en,          // advance one k-step
    input  logic [2*DATA_WIDTH-1:0] grp_nz,      // {nz1, nz0}
    input  logic [3:0]              grp_meta,    // {idx1, idx0}
    output logic [DATA_WIDTH-1:0]   k_data,      // dense weight for this k-step
    output logic                    k_nonzero,   // 1 = survivor, 0 = pruned
    output logic                    grp_advance, // strobe: present the next group
    output logic                    meta_legal
);

    logic [1:0]              phase;
    logic [4*DATA_WIDTH-1:0] dense_group;
    logic [3:0]              nz_mask;

    sparsity_decoder #(
        .DATA_WIDTH (DATA_WIDTH)
    ) u_dec (
        .nz_in      (grp_nz),
        .meta       (grp_meta),
        .dense_out  (dense_group),
        .nz_mask    (nz_mask),
        .meta_legal (meta_legal)
    );

    always_ff @(posedge clk) begin
        if (rst)
            phase <= 2'd0;
        else if (en)
            phase <= phase + 2'd1;
    end

    assign k_data      = dense_group[phase*DATA_WIDTH +: DATA_WIDTH];
    assign k_nonzero   = nz_mask[phase];
    assign grp_advance = en && (phase == 2'd3);

endmodule
