//-----------------------------------------------------------------------------
// sparse_feeder - north-edge 2:4 weight expander for one array column
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 3 (2:4 sparsity front-end integration)
//
// Sits at the array BOUNDARY, outside the systolic mesh, and turns a
// compressed 2:4 weight stream into the dense per-cycle stream the column
// expects. It is a boundary block, so it never adds a stage to the MESH
// itself (the i+j+k skew formula in systolic_array_sparse.sv is untouched).
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
// The expansion itself is the purely combinational sparsity_decoder (that
// module has zero registered paths, by design and by the plan's own Week-9
// verification step, and is untouched here); phase is the 2-bit counter that
// walks the group.
//
// OUTPUT REGISTER (added for FPGA timing closure): phase_reg -> decoder ->
// per-slot mux -> PE b_in -> pre-conditioning -> multiply was one long
// unregistered chain that missed 250 MHz on Kintex-7 by ~0.5 ns / 115+
// endpoints (see synth/fpga/reports/timing_sparse_nogate_uniform50.rpt from
// before this fix). k_data/k_nonzero/grp_advance/meta_legal are now captured
// in one register together, so they stay aligned with each other.
//
// NO DRIVER-SIDE COMPENSATION IS NEEDED for this register. k_data_c and
// meta_legal_c are combinational from THIS CYCLE's grp_nz/grp_meta (and, for
// k_data_c, `phase`, whose own value already accounts for cumulative `en`
// correctly) - never from another register's output a cycle upstream. A
// driver using the standard "set inputs, await the clock edge, then read"
// idiom - the idiom every testbench in this project already uses - reads
// this register's value for the SAME cycle's inputs with no extra skew,
// exactly as it would read a plain wire. (This is unlike pe_mixed's
// pipeline, where product_q's input chains through another register -
// m_hh_q - set up a full cycle earlier; that one genuinely needs a
// one-cycle compensation. This module does not.)
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
    logic                    meta_legal_c;

    sparsity_decoder #(
        .DATA_WIDTH (DATA_WIDTH)
    ) u_dec (
        .nz_in      (grp_nz),
        .meta       (grp_meta),
        .dense_out  (dense_group),
        .nz_mask    (nz_mask),
        .meta_legal (meta_legal_c)
    );

    always_ff @(posedge clk) begin
        if (rst)
            phase <= 2'd0;
        else if (en)
            phase <= phase + 2'd1;
    end

    // Stage 0 (combinational): this cycle's slot out of the decoded group.
    logic [DATA_WIDTH-1:0] k_data_c;
    logic                   k_nonzero_c, grp_advance_c;

    assign k_data_c     = dense_group[phase*DATA_WIDTH +: DATA_WIDTH];
    assign k_nonzero_c   = nz_mask[phase];
    assign grp_advance_c = en && (phase == 2'd3);

    // Output register - see the OUTPUT REGISTER note above.
    always_ff @(posedge clk) begin
        if (rst) begin
            k_data      <= '0;
            k_nonzero   <= 1'b0;
            grp_advance <= 1'b0;
            meta_legal  <= 1'b0;
        end else begin
            k_data      <= k_data_c;
            k_nonzero   <= k_nonzero_c;
            grp_advance <= grp_advance_c;
            meta_legal  <= meta_legal_c;
        end
    end

endmodule
