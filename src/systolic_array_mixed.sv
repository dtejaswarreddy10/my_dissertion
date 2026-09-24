//-----------------------------------------------------------------------------
// systolic_array_mixed - N x N output-stationary precision-adaptive array
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 2 (array integration of the precision-adaptive PE)
//
// Identical topology, port packing and skew contract to systolic_array.sv -
// the only difference is that each node is a `pe_mixed` and the array carries
// a broadcast `mode` select. Kept as a SEPARATE top level so the baseline
// INT8 array stays intact as the `baseline_dense_int8` synthesis config.
//
//   mode = 0 : every PE does one INT8 MAC   per cycle
//   mode = 1 : every PE does two INT4  MACs per cycle (lanes packed per byte)
//
// TIMING / SKEW CONTRACT (unchanged - see systolic_array.sv for the rationale):
//   * Row    i of A is injected at the WEST  edge delayed by i cycles.
//   * Column j of B is injected at the NORTH edge delayed by j cycles.
//   * valid for row i tracks that row's A injection.
//   A[i][k] and B[k][j] meet at PE(i,j) on cycle (i + j + k); the PE pipelines
//   its multiply, so C[i][j] settles ONE CYCLE after the operands arrive.
//
// In INT4x2 mode each streamed byte carries two K-steps, so a 2K-deep dot
// product completes in K cycles - this is where the ~2x throughput comes from.
// `mode` is broadcast rather than pipelined with the operands, so it must only
// change on a tile boundary: assert acc_clear when switching precision.
//
// Port packing (flattened vectors for tool / cocotb portability):
//   a_west  : row i    = a_west [i*DATA_WIDTH +: DATA_WIDTH]
//   b_north : column j = b_north[j*DATA_WIDTH +: DATA_WIDTH]
//   c_out   : C[i][j]  = c_out  [(i*N + j)*ACC_WIDTH +: ACC_WIDTH]
//   valid_west[i] / valid_east[i] : 1 bit per row
//-----------------------------------------------------------------------------
module systolic_array_mixed #(
    parameter int N          = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       mode,        // 0 = INT8, 1 = INT4x2
    input  logic                       acc_clear,
    input  logic [N*DATA_WIDTH-1:0]    a_west,
    input  logic [N-1:0]               valid_west,
    input  logic [N*DATA_WIDTH-1:0]    b_north,
    output logic [N*N*ACC_WIDTH-1:0]   c_out,
    output logic [N-1:0]               valid_east
);

    // Mesh interconnect:
    //   a_h[i][j]   -> PE(i,j).a_in     ; PE(i,j).a_out     -> a_h[i][j+1] (east)
    //   b_v[i][j]   -> PE(i,j).b_in     ; PE(i,j).b_out     -> b_v[i+1][j] (south)
    //   v_h[i][j]   -> PE(i,j).valid_in ; PE(i,j).valid_out -> v_h[i][j+1] (east)
    logic [DATA_WIDTH-1:0] a_h [N][N+1];
    logic [DATA_WIDTH-1:0] b_v [N+1][N];
    logic                  v_h [N][N+1];

    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : g_row
            // West-edge injection for row i (A + valid flow east from here).
            assign a_h[i][0]     = a_west[i*DATA_WIDTH +: DATA_WIDTH];
            assign v_h[i][0]     = valid_west[i];
            assign valid_east[i] = v_h[i][N];

            for (j = 0; j < N; j++) begin : g_col
                // North-edge injection for column j (top row only; B flows south).
                if (i == 0) begin : g_north
                    assign b_v[0][j] = b_north[j*DATA_WIDTH +: DATA_WIDTH];
                end

                pe_mixed #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
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
