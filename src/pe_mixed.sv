//-----------------------------------------------------------------------------
// pe_mixed - Precision-adaptive output-stationary MAC processing element
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 2 (Contribution 1 - fracturable precision-adaptive PE)
//
// Supersedes pe_int8.sv. Runtime-selectable precision on a SHARED partial-
// product array and a SHARED accumulator:
//
//   MODE_INT8   : acc += a * b                      (1 MAC / cycle)
//   MODE_INT4X2 : acc += a_hi*b_hi + a_lo*b_lo      (2 MACs / cycle)
//
// Packing contract (mirrors golden/golden_model.py):
//   a_in[7:4] = lane 1 (signed 4-bit)   a_in[3:0] = lane 0 (signed 4-bit)
// In INT4x2 mode both lanes are signed; in INT8 mode the low nibble is the
// unsigned low half of a signed 8-bit word.
//
// HOW THE SHARING WORKS
// ---------------------
// Split each operand into halves:   a = 2^H * a_hi + a_lo
// Then the INT8 product expands into four H x H sub-products:
//
//   a*b = 2^2H (a_hi*b_hi) + 2^H (a_hi*b_lo + a_lo*b_hi) + (a_lo*b_lo)
//
// and the INT4x2 result is simply the HH and LL terms with no shift:
//
//   a_hi*b_hi + a_lo*b_lo
//
// So BOTH modes reuse the same four sub-multipliers and the same reduction
// tree; only (a) the signedness of the low half and (b) the HH shift amount
// differ. Both are resolved in a PRE-CONDITIONING stage ahead of the array -
// never as a mux on the partial products - so the mode select never lands on
// the critical path through the reduction tree.
//
// The two cross terms (HL, LH) are unused in INT4x2 mode and their operands
// are forced to zero, which isolates those sub-multipliers and suppresses
// their switching activity. This is the same operand-isolation mechanism the
// Month-3 clock-gating work builds on.
//
// TIMING: identical contract to pe_int8.sv - 1 cycle/hop forwarding, product
// registered in stage 1, accumulate in stage 2, so array skew is UNCHANGED.
//
// MODE CHANGES ARE TILE-GRANULAR: `mode` is broadcast, not pipelined with the
// operands. Assert acc_clear when switching mode so no tile mixes precisions.
//
// ENABLE_GATING (Month 3, Contribution 3)
// ---------------------------------------
// When set, the PE detects a zero operand byte, isolates the multiply tree and
// stops the clock to the MAC pipeline registers.
//
// WHAT IS *NOT* GATED: a_out / b_out / valid_out run on the ungated clock and
// always advance. Gating the forwarding path would stall the systolic mesh and
// corrupt every downstream PE, so only product_q and acc_q see the gated clock.
//
// FUNCTIONAL EQUIVALENCE: a zero operand byte makes every sub-product zero, so
// suppressing the accumulate for that cycle yields exactly the same
// accumulator value as adding zero. Results are bit-identical with gating on
// or off - only the switching activity differs. That is what makes the
// power-vs-sparsity sweep a clean single-variable experiment.
//
// SINGLE CLOCK DOMAIN: the gated clock is derived from clk inside the same PE
// and only ever feeds registers in that PE. No clock domain crossing is
// created, so no synchronisers or CDC constraints are required.
//-----------------------------------------------------------------------------
module pe_mixed #(
    parameter int DATA_WIDTH    = 8,
    parameter int ACC_WIDTH     = 32,
    parameter bit ENABLE_GATING = 1'b0
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    mode,        // 0 = INT8, 1 = INT4x2
    input  logic                    valid_in,
    input  logic                    acc_clear,
    input  logic [DATA_WIDTH-1:0]   a_in,
    input  logic [DATA_WIDTH-1:0]   b_in,
    output logic [DATA_WIDTH-1:0]   a_out,
    output logic [DATA_WIDTH-1:0]   b_out,
    output logic                    valid_out,
    output logic [ACC_WIDTH-1:0]    acc_result
);

    localparam logic MODE_INT8   = 1'b0;
    localparam logic MODE_INT4X2 = 1'b1;

    localparam int HALF = DATA_WIDTH / 2;      // 4
    localparam int EXT  = HALF + 1;            // 5 - room for the sign bit
    localparam int PROD = 2 * EXT;             // 10 - one sub-product
    // Worst-case full product plus the shifted partial sums.
    localparam int SUM  = 2 * DATA_WIDTH + 2;  // 18

    generate
        if (DATA_WIDTH % 2 != 0) begin : g_bad_width
            // The nibble split is undefined for odd widths.
            $error("pe_mixed: DATA_WIDTH must be even, got %0d", DATA_WIDTH);
        end
    endgenerate

    logic int4_mode;
    assign int4_mode = (mode == MODE_INT4X2);

    //-------------------------------------------------------------------------
    // Zero detection and operand isolation (Month 3).
    // A zero operand BYTE kills every sub-product in both modes: if a_in is
    // zero then a_hi and a_lo are both zero. Detection is therefore done at
    // byte granularity - conservative, mode-independent, and one OR-reduce
    // deep, so it never threatens the critical path.
    //-------------------------------------------------------------------------
    logic zero_operand;
    logic mac_active;

    assign zero_operand = (a_in == '0) || (b_in == '0);
    assign mac_active   = ENABLE_GATING ? (valid_in && !zero_operand) : valid_in;

    // Isolators hold the multiply tree inputs at zero whenever this cycle
    // cannot contribute, so idle/drain traffic does not toggle the datapath.
    logic [DATA_WIDTH-1:0] a_iso, b_iso;
    assign a_iso = (ENABLE_GATING && !mac_active) ? '0 : a_in;
    assign b_iso = (ENABLE_GATING && !mac_active) ? '0 : b_in;

    //-------------------------------------------------------------------------
    // Stage 0 (combinational): operand pre-conditioning.
    // The high half is always signed. The low half is signed only in INT4x2;
    // in INT8 it is the unsigned low nibble of the signed byte.
    //-------------------------------------------------------------------------
    logic signed [EXT-1:0] a_hi, a_lo, b_hi, b_lo;

    assign a_hi = $signed({a_iso[DATA_WIDTH-1], a_iso[DATA_WIDTH-1 -: HALF]});
    assign b_hi = $signed({b_iso[DATA_WIDTH-1], b_iso[DATA_WIDTH-1 -: HALF]});
    assign a_lo = int4_mode ? $signed({a_iso[HALF-1], a_iso[HALF-1:0]})
                            : $signed({1'b0,         a_iso[HALF-1:0]});
    assign b_lo = int4_mode ? $signed({b_iso[HALF-1], b_iso[HALF-1:0]})
                            : $signed({1'b0,         b_iso[HALF-1:0]});

    // Cross terms exist only in INT8 mode; isolate their operands otherwise so
    // the HL/LH sub-multipliers hold still instead of toggling on dead data.
    logic signed [EXT-1:0] hl_a, hl_b, lh_a, lh_b;
    assign hl_a = int4_mode ? '0 : a_hi;
    assign hl_b = int4_mode ? '0 : b_lo;
    assign lh_a = int4_mode ? '0 : a_lo;
    assign lh_b = int4_mode ? '0 : b_hi;

    //-------------------------------------------------------------------------
    // Shared partial-product array: four HALFxHALF signed sub-multipliers.
    // use_dsp="no" keeps the fracturable structure in fabric so the FPGA area
    // delta is measured against the same implementation style as the baseline;
    // the ASIC flow (Yosys/FreePDK45) is the primary area comparison.
    //-------------------------------------------------------------------------
    (* use_dsp = "no" *) logic signed [PROD-1:0] m_hh, m_hl, m_lh, m_ll;

    assign m_hh = a_hi * b_hi;   // used in BOTH modes
    assign m_hl = hl_a * hl_b;   // INT8 only
    assign m_lh = lh_a * lh_b;   // INT8 only
    assign m_ll = a_lo * b_lo;   // used in BOTH modes

    //-------------------------------------------------------------------------
    // Shared reduction tree. Only the HH shift is mode-dependent (<<2H for the
    // INT8 expansion, <<0 for the INT4x2 lane sum) - a narrow 2:1 select on a
    // single term, not a mux across the whole tree.
    //-------------------------------------------------------------------------
    logic signed [SUM-1:0] hh_term, cross_term, product_c;

    assign hh_term    = int4_mode ? SUM'(m_hh) : (SUM'(m_hh) <<< (2 * HALF));
    assign cross_term = SUM'(m_hl + m_lh) <<< HALF;   // zero in INT4x2 mode
    assign product_c  = hh_term + cross_term + SUM'(m_ll);

    //-------------------------------------------------------------------------
    // Stage 1 / Stage 2 pipeline - identical latency to pe_int8.sv.
    //
    // Two clock nets, one source:
    //   clk       -> forwarding registers (NEVER gated; the mesh depends on it)
    //   gclk_mac  -> product_q            (gated when this cycle cannot MAC)
    //   gclk_acc  -> acc_q                (gated when nothing will be added)
    // valid_q stays on the ungated clock: it is one bit, and it must be able to
    // fall so that a stale product_q left behind by a gated cycle can never be
    // accumulated.
    //-------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] product_q;
    logic                        valid_q;
    logic signed [ACC_WIDTH-1:0] acc_q;

    assign acc_result = acc_q;

    logic gclk_mac, gclk_acc;
    logic acc_active;

    assign acc_active = valid_q || acc_clear;

    generate
        if (ENABLE_GATING) begin : g_icg
            // rst is ORed into both enables: the reset is synchronous, so the
            // gated clock must keep toggling during reset or product_q/acc_q
            // would never be cleared.
            // test_en tied off: DFT insertion is out of scope for this work.
            icg_cell u_icg_mac (
                .clk     (clk),
                .enable  (mac_active | rst),
                .test_en (1'b0),
                .gclk    (gclk_mac)
            );
            icg_cell u_icg_acc (
                .clk     (clk),
                .enable  (acc_active | rst),
                .test_en (1'b0),
                .gclk    (gclk_acc)
            );
        end else begin : g_no_icg
            assign gclk_mac = clk;
            assign gclk_acc = clk;
        end
    endgenerate

    // Forwarding path - ungated, always advances.
    always_ff @(posedge clk) begin
        if (rst) begin
            a_out     <= '0;
            b_out     <= '0;
            valid_out <= 1'b0;
            valid_q   <= 1'b0;
        end else begin
            a_out     <= a_in;
            b_out     <= b_in;
            valid_out <= valid_in;
            valid_q   <= mac_active;
        end
    end

    // Stage 1 - product capture on the gated clock.
    always_ff @(posedge gclk_mac) begin
        if (rst)
            product_q <= '0;
        else
            product_q <= ACC_WIDTH'(product_c);
    end

    // Stage 2 - accumulate on the gated clock (acc_clear keeps priority).
    always_ff @(posedge gclk_acc) begin
        if (rst)
            acc_q <= '0;
        else if (acc_clear)
            acc_q <= '0;
        else if (valid_q)
            acc_q <= acc_q + product_q;
    end

endmodule
