//-----------------------------------------------------------------------------
// systolic_array - N x N output-stationary INT8 systolic array (baseline)
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 1 / Week 2 deliverable
//           "Instantiate 8x8 array: A flows east, B flows south, acc stationary."
//
// Composition: N*N pe_int8 instances in a mesh.
//   * A operands enter the WEST edge  (one per row),    flow EAST.
//   * B operands enter the NORTH edge (one per column), flow SOUTH.
//   * valid enters the WEST edge with A and flows EAST.
//   * Each PE(i,j) accumulates C[i][j] in place (output stationary).
//
// TIMING / SKEW CONTRACT (the testbench/driver must honour this):
//   To compute C = A * B (A is N x K, B is K x N; here K = N), operands must be
//   skewed at the boundaries so A[i][k] and B[k][j] reach PE(i,j) on the same
//   cycle:
//     * Row    i of A is injected at the WEST  edge delayed by i cycles.
//     * Column j of B is injected at the NORTH edge delayed by j cycles.
//     * valid for row i tracks that row's A injection (delayed by i cycles).
//   With 1-cycle registered hops, A[i][k] and B[k][j] meet at PE(i,j) on cycle
//   (i + j + k). NOTE: the PE pipelines its multiply, so each product is
//   accumulated one cycle after the operands arrive -> C[i][j] becomes valid
//   ONE CYCLE LATER than a single-cycle PE would give. The input injection skew
//   is unchanged (operand forwarding is still 1 cycle per hop); only the result
//   read-out cycle shifts by +1.
//
// Port packing (flattened vectors for tool / cocotb portability):
//   a_west  : row i    = a_west [i*DATA_WIDTH +: DATA_WIDTH]
//   b_north : column j = b_north[j*DATA_WIDTH +: DATA_WIDTH]
//   c_out   : C[i][j]  = c_out  [(i*N + j)*ACC_WIDTH +: ACC_WIDTH]
//   valid_west[i] / valid_east[i] : 1 bit per row
//-----------------------------------------------------------------------------
module systolic_array #(
    parameter int N          = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                       clk,
    input  logic                       rst,
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

                pe_int8 #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) u_pe (
                    .clk        (clk),
                    .rst        (rst),
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
