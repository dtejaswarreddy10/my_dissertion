# ----------------------------------------------------------------------------
# EXPERIMENTAL variant of synth_pe_mixed.tcl - FreePDK45 track.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 / Section 3.4 follow-up - root-causing the pe_mixed
#           ASIC "wire has no driver" check-assert failure
#
# synth_pe_mixed.tcl (unmodified, kept as the reference/baseline script) hits
# a 49-problem check-assert failure for pe_mixed on BOTH FreePDK45 and sky130:
# a_out, b_out, valid_out and acc_q all come back "used but has no driver"
# after the second (liberty) ABC pass. See logs/synth_pe_mixed_freepdk45.log
# for the full analysis - cocotb already proves these signals are correctly
# driven at the RTL level, so this is a synthesis-tool artifact, not an RTL
# bug. Current theory: the default Yosys abc recipe's sequential-equivalence/
# choice-based passes (scorr, dc2, dretime, &dch, &nf) mis-associate a
# register's original port name during re-integration on pe_mixed's larger
# netlist (~1198 gates vs pe_int8's ~741, same recipe, no problem on pe_int8).
#
# This script is a NEW file (not an edit of synth_pe_mixed.tcl) so the known
# reference run stays reproducible while this fix is tried. Two changes vs.
# the reference script:
#
#   1. The second ABC pass uses a custom script (abc_nolegalize.script, next
#      to this file) that strips the recipe down to `strash; map` - i.e. no
#      scorr/dc2/dretime/&dch/&nf - to test whether removing those passes
#      removes the undriven-wire problem.
#   2. `yosys check -assert` (which raises a FATAL Tcl error that the old
#      script's `catch` could not actually catch - the gated config never
#      ran) is replaced with plain `yosys check` (no -assert). This never
#      aborts, so BOTH configs always get a netlist written, whether or not
#      the ABC-script change actually resolves the warnings - that result is
#      exactly what tells us if the theory above is right.
#
# Run:
#   yosys -c synth_pe_mixed_abcfix.tcl
#
# Environment:
#   FREEPDK45_LIB  path to the FreePDK45 / Nangate Liberty (required)
#
# Outputs (in ./reports, suffixed _abcfix to not clobber the reference run):
#   pe_mixed_nogate_abcfix_stat.txt / pe_mixed_gated_abcfix_stat.txt
#   pe_mixed_nogate_abcfix.v        / pe_mixed_gated_abcfix.v
# ----------------------------------------------------------------------------

if {[info exists ::env(FREEPDK45_LIB)]} {
    set lib $::env(FREEPDK45_LIB)
} else {
    set lib "/path/to/FreePDK45/stdcells.lib"   ;# <-- EDIT if FREEPDK45_LIB is unset
}

set here      [file normalize [file dirname [info script]]]
set src       [file normalize $here/../../src]
set abcscript [file normalize $here/abc_nolegalize.script]
file mkdir $here/reports

if {![file exists $lib]} {
    puts "ERROR: Liberty file not found: $lib"
    puts "       setenv FREEPDK45_LIB /path/to/NangateOpenCellLibrary_typical.lib"
    exit 1
}

if {![file exists $abcscript]} {
    puts "ERROR: ABC script not found: $abcscript"
    exit 1
}

# cfg_name -> ENABLE_GATING value
set configs [list nogate 0 gated 1]

foreach {name gating} $configs {

    puts "=============================================================="
    puts "INFO: synthesising pe_mixed  ENABLE_GATING=$gating  ($name)  (abcfix)"
    puts "=============================================================="

    yosys design -reset

    yosys read_verilog -sv $src/icg_cell.sv
    yosys read_verilog -sv $src/pe_mixed.sv

    yosys chparam -set ENABLE_GATING $gating pe_mixed
    yosys hierarchy -check -top pe_mixed

    yosys synth -top pe_mixed -flatten

    yosys dfflibmap -liberty $lib
    yosys abc -liberty $lib -script $abcscript
    yosys opt_clean

    # No -assert: always write the netlist so the ABC-script change can be
    # judged by whether the warning COUNT below actually changed, not by
    # whether the script survived.
    yosys check

    yosys tee -o $here/reports/pe_mixed_${name}_abcfix_stat.txt stat -liberty $lib
    yosys write_verilog -noattr $here/reports/pe_mixed_${name}_abcfix.v

    puts "INFO: -> reports/pe_mixed_${name}_abcfix_stat.txt"
    puts "INFO: -> reports/pe_mixed_${name}_abcfix.v"
}

puts ""
puts "=============================================================="
puts "COMPARE AGAINST THE REFERENCE RUN"
puts "  Reference (unmodified recipe): yosys -c synth_pe_mixed.tcl"
puts "    -> 49 problems reported by 'check -assert', run aborts after"
puts "       'nogate', 'gated' never produced."
puts "  This script: if the 'check' output above now reports 0 problems for"
puts "  both configs, the scorr/dc2/dretime/&dch/&nf removal fixed it - the"
puts "  theory in the header comment is confirmed. If the same 4 signals"
puts "  (a_out/b_out/valid_out/acc_q) are still undriven, the cause is"
puts "  elsewhere and this ABC-script change should be reverted/discarded."
puts "=============================================================="
