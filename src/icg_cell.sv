//-----------------------------------------------------------------------------
// icg_cell - integrated clock-gating cell wrapper
//
// Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
// Stage   : Month 3 (Contribution 3 - per-PE clock gating)
//
// One wrapper, three implementations, selected at compile time. This is the
// module where the "FPGA power != ASIC power" caveat from the dissertation
// plan physically lives, so the three paths are kept explicit rather than
// hidden behind tool inference:
//
//   default (ASIC / simulation)  latch-based ICG: the enable is captured on
//                                the LOW phase of clk and ANDed with clk, so
//                                the gated clock can never glitch. This is the
//                                structure of a standard-cell ICG and is what
//                                Yosys/FreePDK45 maps and what the gate-level
//                                SAIF power estimate measures.
//
//   ICG_USE_BUFGCE (FPGA)        Xilinx BUFGCE primitive. NOTE: a Kintex-7 has
//                                only ~32 global clock buffers, so an 8x8 array
//                                CANNOT give 64 PEs independent gated clocks.
//                                Use this only for small/PE-level experiments.
//
//   ICG_NO_GATING (FPGA sweep)   pass-through. The PE then falls back to plain
//                                flip-flop clock-enable, which holds state but
//                                does NOT stop the clock and does NOT reduce
//                                clock-tree power. This is the honest FPGA
//                                configuration, and its results must be
//                                reported as "operand isolation + clock enable",
//                                never as an ICG result.
//
//   ICG_STDCELL (ASIC, optional) instantiate a named library clock-gating cell
//                                instead of letting the tool infer the latch.
//                                Cell and pin names differ between libraries,
//                                so they are macro-driven, e.g.:
//                                  -DICG_STDCELL -DICG_CELL_NAME=CLKGATETST_X1
//                                Check your Liberty for the actual cell before
//                                enabling this; the default inferred-latch path
//                                needs no library-specific knowledge.
//
// test_en forces the clock through for scan/DFT and must be tied low in
// functional mode.
//
// GLITCH SAFETY: enable is sampled while clk is low, so it is already settled
// when clk rises. The enable is driven from flip-flop outputs of neighbouring
// PEs (a_in/b_in/valid_in), which settle shortly after the previous rising
// edge, leaving most of the low phase as margin.
//-----------------------------------------------------------------------------
`ifndef ICG_CELL_NAME
  `define ICG_CELL_NAME CLKGATETST_X1
`endif

module icg_cell (
    input  logic clk,
    input  logic enable,
    input  logic test_en,
    output logic gclk
);

`ifdef ICG_NO_GATING
    // Pass-through: gating degenerates to flip-flop clock-enable in the PE.
    assign gclk = clk;
    // verilator lint_off UNUSED
    wire _unused = &{1'b0, enable, test_en};
    // verilator lint_on UNUSED

`elsif ICG_USE_BUFGCE
    BUFGCE u_bufgce (
        .I  (clk),
        .CE (enable | test_en),
        .O  (gclk)
    );

`elsif ICG_STDCELL
    // Library cell; pin names follow the Nangate/FreePDK45 convention.
    `ICG_CELL_NAME u_icg (
        .CK  (clk),
        .E   (enable),
        .SE  (test_en),
        .GCK (gclk)
    );

`else
    // Latch-based ICG (standard-cell structure).
    logic en_latched;

    always_latch begin
        if (!clk)
            en_latched = enable | test_en;
    end

    assign gclk = clk & en_latched;
`endif

endmodule
