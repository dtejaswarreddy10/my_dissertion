//-----------------------------------------------------------------------------
// systolic_array_sparse - full design: mixed precision + 2:4 sparsity + ICG
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 3 (all three contributions combined - the `full_design`
//           configuration of the Month 4 synthesis sweep)
//
// Combines:
//   Contribution 1  pe_mixed          INT8 / INT4x2 on a shared reduction tree
//   Contribution 2  sparse_feeder     2:4 compressed weight stream at the
//                                     north edge (one feeder per column)
//   Contribution 3  ENABLE_GATING     per-PE operand isolation + ICG
//
// SYNTHESIS SWEEP HOOKS - the four configurations of the dissertation plan are
// reachable without editing RTL:
//   baseline_dense_int8              -> systolic_array.sv       (separate top)
//   precision_adaptive_no_sparsity   -> systolic_array_mixed.sv (separate top)
//   precision_adaptive_sparsity      -> this top, ENABLE_GATING = 0
//   full_design                      -> this top, ENABLE_GATING = 1
//
// DATAFLOW. Activations (A) stay dense and enter the WEST edge exactly as in
// systolic_array.sv. Weights (B) arrive COMPRESSED at the north edge: two
// surviving values plus 4 bits of metadata per group of four k-steps. Each
// column's feeder expands them back to one word per cycle, inserting explicit
// zeros at the pruned positions.
//
// SKEW CONTRACT (unchanged): row i of A is delayed i cycles, column j of B is
// delayed j cycles. Since each column starts at a different cycle, each column
// owns its own feeder phase counter and its own `feed_en` bit - assert
// feed_en[j] from cycle j onward.
//
// WHY SPARSITY BUYS POWER AND NOT THROUGHPUT HERE: the feeder does not compact
// the schedule - a group of four k-steps still costs four cycles, two of them
// streaming zeros. Those zero cycles gate the MAC, which is where the dynamic
// power reduction comes from. Schedule compaction (Ampere-style 2x throughput)
// needs a load-balancing controller and is explicitly out of scope; the
// throughput gain in this work comes solely from INT4x2 packing.
//
// Port packing (flattened for tool / cocotb portability):
//   a_west   : row i    = a_west  [i*DATA_WIDTH     +: DATA_WIDTH]
//   grp_nz   : column j = grp_nz  [j*2*DATA_WIDTH   +: 2*DATA_WIDTH]
//   grp_meta : column j = grp_meta[j*4              +: 4]
//   c_out    : C[i][j]  = c_out   [(i*N + j)*ACC_WIDTH +: ACC_WIDTH]
//-----------------------------------------------------------------------------
module systolic_array_sparse #(
    parameter int N             = 8,
    parameter int DATA_WIDTH    = 8,
    parameter int ACC_WIDTH     = 32,
    parameter bit ENABLE_GATING = 1'b1
)(
    input  logic                        clk,
    input  logic                        rst,
    input  logic                        mode,        // 0 = INT8, 1 = INT4x2
    input  logic                        acc_clear,

    // Dense activations, west edge (row-skewed by the driver).
    input  logic [N*DATA_WIDTH-1:0]     a_west,
    input  logic [N-1:0]                valid_west,

    // Compressed 2:4 weights, north edge (column-skewed by the driver).
    input  logic [N*2*DATA_WIDTH-1:0]   grp_nz,
    input  logic [N*4-1:0]              grp_meta,
    input  logic [N-1:0]                feed_en,
    output logic [N-1:0]                grp_advance,
    output logic [N-1:0]                meta_legal,

    output logic [N*N*ACC_WIDTH-1:0]    c_out,
    output logic [N-1:0]                valid_east
);

    // Expanded (dense) weight entering the top of each column this cycle.
    logic [DATA_WIDTH-1:0] b_dense [N];

    // Mesh interconnect - identical structure to systolic_array.sv.
    logic [DATA_WIDTH-1:0] a_h [N][N+1];
    logic [DATA_WIDTH-1:0] b_v [N+1][N];
    logic                  v_h [N][N+1];

    genvar i, j;
    generate
        //---------------------------------------------------------------------
        // North-edge sparsity front-end: one feeder per column.
        //---------------------------------------------------------------------
        for (j = 0; j < N; j++) begin : g_feed
            logic unused_nonzero;

            sparse_feeder #(
                .DATA_WIDTH (DATA_WIDTH)
            ) u_feeder (
                .clk         (clk),
                .rst         (rst),
                .en          (feed_en[j]),
                .grp_nz      (grp_nz[j*2*DATA_WIDTH +: 2*DATA_WIDTH]),
                .grp_meta    (grp_meta[j*4 +: 4]),
                .k_data      (b_dense[j]),
                .k_nonzero   (unused_nonzero),
                .grp_advance (grp_advance[j]),
                .meta_legal  (meta_legal[j])
            );

            // Column j receives the expanded weight at the top of the mesh.
            assign b_v[0][j] = b_dense[j];
        end

        //---------------------------------------------------------------------
        // Precision-adaptive, optionally clock-gated PE mesh.
        //---------------------------------------------------------------------
        for (i = 0; i < N; i++) begin : g_row
            assign a_h[i][0]     = a_west[i*DATA_WIDTH +: DATA_WIDTH];
            assign v_h[i][0]     = valid_west[i];
            assign valid_east[i] = v_h[i][N];

            for (j = 0; j < N; j++) begin : g_col
                pe_mixed #(
                    .DATA_WIDTH    (DATA_WIDTH),
                    .ACC_WIDTH     (ACC_WIDTH),
                    .ENABLE_GATING (ENABLE_GATING)
                ) u_pe (
                    .clk        (clk),
                    .rst        (rst),
                    .mode       (mode),
                    .valid_in   (v_h[i][j]),
                    .acc_clear  (acc_clear),
                    .a_in       (a_h[i][j]),
                    .b_in       (b_v[i][j]),
                    .a_out      (a_h[i][j+1]),
                    .b_out      (b_v[i+1][j]),
                    .valid_out  (v_h[i][j+1]),
                    .acc_result (c_out[(i*N + j)*ACC_WIDTH +: ACC_WIDTH])
                );
            end
        end
    endgenerate

endmodule
