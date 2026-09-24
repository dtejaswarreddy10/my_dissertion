//-----------------------------------------------------------------------------
// cov_pe_int8 - functional coverage for the baseline INT8 PE
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 1 / Week 4 deliverable ("covergroups, >90% on baseline INT8")
//
// Non-intrusive: bound into pe_int8 at the bottom of this file, so the RTL
// itself stays synthesis-clean. Requires a simulator with SystemVerilog
// covergroup support (Questa) - Icarus cannot run this file.
//
//   make SIM=questa TEST=pe COVERAGE=1
//
// Coverage intent:
//   * operand value classes incl. the INT8 extremes (+127 / -128) and zero
//   * a x b cross, so sign combinations and extreme-by-extreme are hit
//   * zero-operand events  -> this is the trigger for Month-3 ICG, so proving
//     it is exercised now de-risks the clock-gating work later
//   * valid / acc_clear states and transitions (tile boundary behaviour)
//   * accumulator polarity and large-magnitude occupancy
//-----------------------------------------------------------------------------
module cov_pe_int8 #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input logic                   clk,
    input logic                   rst,
    input logic                   valid_in,
    input logic                   acc_clear,
    input logic [DATA_WIDTH-1:0]  a_in,
    input logic [DATA_WIDTH-1:0]  b_in,
    input logic [ACC_WIDTH-1:0]   acc_result
);

    logic signed [DATA_WIDTH-1:0] a_s;
    logic signed [DATA_WIDTH-1:0] b_s;
    logic signed [ACC_WIDTH-1:0]  acc_s;
    logic                         operand_is_zero;

    assign a_s             = $signed(a_in);
    assign b_s             = $signed(b_in);
    assign acc_s           = $signed(acc_result);
    assign operand_is_zero = (a_in == '0) || (b_in == '0);

    covergroup cg_pe @(posedge clk iff (!rst));
        option.per_instance = 1;
        option.name         = "cg_pe_int8";

        cp_a : coverpoint a_s {
            bins zero     = {0};
            bins plus_one = {1};
            bins min_one  = {-1};
            bins max      = {127};
            bins min      = {-128};
            bins pos[3]   = {[2:126]};
            bins neg[3]   = {[-127:-2]};
        }

        cp_b : coverpoint b_s {
            bins zero     = {0};
            bins plus_one = {1};
            bins min_one  = {-1};
            bins max      = {127};
            bins min      = {-128};
            bins pos[3]   = {[2:126]};
            bins neg[3]   = {[-127:-2]};
        }

        // Zero-operand rate is the Month-3 clock-gating trigger.
        cp_zero_operand : coverpoint operand_is_zero {
            bins busy   = {1'b0};
            bins gated  = {1'b1};
        }

        cp_valid : coverpoint valid_in {
            bins idle   = {1'b0};
            bins active = {1'b1};
        }

        cp_clear : coverpoint acc_clear {
            bins run   = {1'b0};
            bins clear = {1'b1};
        }

        // Transitions kept separate so they are not pulled into the crosses.
        cp_valid_tr : coverpoint valid_in {
            bins start = (1'b0 => 1'b1);
            bins stop  = (1'b1 => 1'b0);
            bins hold  = (1'b1 => 1'b1);
        }

        cp_clear_tr : coverpoint acc_clear {
            bins pulse = (1'b0 => 1'b1 => 1'b0);
        }

        cp_acc : coverpoint acc_s {
            bins negative   = {[$:-1]};
            bins zero       = {0};
            bins positive   = {[1:$]};
            bins big_pos    = {[100000:$]};
            bins big_neg    = {[$:-100000]};
        }

        // Sign / extreme combinations of the two operands.
        x_operands   : cross cp_a, cp_b;
        // acc_clear asserted while streaming must be covered (tile boundary).
        x_valid_clear: cross cp_valid, cp_clear;
    endgroup

    cg_pe cg_pe_inst;

    initial begin
        cg_pe_inst = new();
    end

    final begin
        $display("[COVERAGE] %m cg_pe_int8 = %0.2f %%", cg_pe_inst.get_inst_coverage());
    end

endmodule

// Non-intrusive attach to every pe_int8 instance (array PEs included).
bind pe_int8 cov_pe_int8 #(
    .DATA_WIDTH (DATA_WIDTH),
    .ACC_WIDTH  (ACC_WIDTH)
) u_cov (.*);
