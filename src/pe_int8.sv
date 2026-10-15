//-----------------------------------------------------------------------------
// pe_int8 - Output-stationary INT8 MAC processing element
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Baseline INT8 PE (Month 1 / Week 2 deliverable)
//
// Dataflow contract (consumed by systolic_array.sv):
//   * a_in     flows EAST  : registered to a_out     (1 cycle / hop)
//   * b_in     flows SOUTH : registered to b_out     (1 cycle / hop)
//   * valid_in flows EAST with A : registered to valid_out
//   * acc_result (C[i][j]) stays stationary inside this PE
//
// Behaviour (2-stage pipelined MAC -> maps to ONE DSP48 with MREG + PREG):
//   stage 1 : product_q <= signed(a_in) * signed(b_in) ;  valid_q <= valid_in
//   stage 2 : if (valid_q) acc_result <= acc_result + product_q
// A product is therefore accumulated 1 cycle AFTER its operands arrive.
// Operand forwarding (a_out/b_out/valid_out) stays 1 cycle/hop, so the systolic
// skew is UNCHANGED; only the final acc_result becomes valid 1 cycle later.
// acc_clear synchronously zeroes the accumulator (start of a new tile); assert it
// before a fresh accumulation sequence (clear-then-stream).
// Reset is synchronous, active-high (so the accumulator packs into a DSP48).
//-----------------------------------------------------------------------------
module pe_int8 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    valid_in,
    input  logic                    acc_clear,
    input  logic [DATA_WIDTH-1:0]   a_in,
    input  logic [DATA_WIDTH-1:0]   b_in,
    output logic [DATA_WIDTH-1:0]   a_out,
    output logic [DATA_WIDTH-1:0]   b_out,
    output logic                    valid_out,
    output logic [ACC_WIDTH-1:0]    acc_result
);

    // Stage-1 pipeline registers: the product (-> DSP MREG) and its aligned valid.
    logic signed [ACC_WIDTH-1:0] product_q;
    logic                        valid_q;
    // Accumulator kept internal with use_dsp so the multiply and the accumulate
    // FUSE into a single DSP48 (P + A*B, MREG+PREG) -> one DSP per PE, not two.
    (* use_dsp = "yes" *) logic signed [ACC_WIDTH-1:0] acc_q;

    assign acc_result = acc_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            a_out      <= '0;
            b_out      <= '0;
            valid_out  <= 1'b0;
            product_q  <= '0;
            valid_q    <= 1'b0;
            acc_q      <= '0;
        end else begin
            // 1-cycle systolic forwarding (latency unchanged -> array skew intact)
            a_out     <= a_in;
            b_out     <= b_in;
            valid_out <= valid_in;

            // Stage 1: register product + valid (product_q maps to the DSP MREG)
            product_q <= $signed(a_in) * $signed(b_in);
            valid_q   <= valid_in;

            // Stage 2: accumulate the registered product (acc_q is the DSP P-reg)
            if (acc_clear)
                acc_q <= '0;
            else if (valid_q)
                acc_q <= acc_q + product_q;
        end
    end

endmodule
