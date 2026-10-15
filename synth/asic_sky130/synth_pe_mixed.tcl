# ----------------------------------------------------------------------------
# Yosys ASIC synthesis of the precision-adaptive PE on sky130_fd_sc_hd,
# ICG OFF and ICG ON.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 / Section 3.4 - "the ASIC flow is where the clock-gating
#           claim actually lives", sky130 track
#
# sky130 counterpart of ../asic/synth_pe_mixed.tcl. icg_cell.sv is read with
# NO macro defined, so it takes its DEFAULT path - a latch-based inferred ICG
# structure (see rtl/icg_cell.sv) - which Yosys maps onto sky130_fd_sc_hd
# cells. sky130_fd_sc_hd does provide a real clock-gating cell
# (sky130_fd_sc_hd__dlclkp_*); wiring ICG_STDCELL/ICG_CELL_NAME directly to it
# is a later refinement once its exact pinout is confirmed, not done here.
#
# An FPGA cannot host 64 independently gated PE clocks (a Kintex-7 has ~32
# BUFGs), so Vivado degrades per-PE gating into flip-flop clock-enables. This
# script produces the ASIC counterpart: two gate-level netlists of the SAME PE,
# identical except for ENABLE_GATING, so a gate-level activity-annotated power
# run can attribute the delta to integrated clock gating alone.
#
# Run:
#   yosys -c synth_pe_mixed.tcl
#
# Liberty resolution order (same as synth_pe.tcl in this folder):
#   SKY130_LIB -> $PDK_ROOT/sky130A/.../sky130_fd_sc_hd__tt_025C_1v80.lib -> placeholder
#
# Outputs (in ./reports):
#   pe_mixed_nogate_stat.txt / pe_mixed_gated_stat.txt   gate counts + area
#   pe_mixed_nogate.v        / pe_mixed_gated.v          gate-level netlists
#
# The netlists feed sta_pe.tcl (timing + power) and the gate-level simulation
# that produces the SAIF. Because the two netlists share ports with the RTL,
# the SAME cocotb testbench re-runs on them - which also proves RTL/netlist
# equivalence for free.
# ----------------------------------------------------------------------------

if {[info exists ::env(SKY130_LIB)]} {
    set lib $::env(SKY130_LIB)
} elseif {[info exists ::env(PDK_ROOT)]} {
    set lib "$::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
} else {
    set lib "/path/to/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
    ;# <-- EDIT if neither SKY130_LIB nor PDK_ROOT is set
}

set here [file normalize [file dirname [info script]]]
set rtl  [file normalize $here/../../src]
file mkdir $here/reports

if {![file exists $lib]} {
    puts "ERROR: Liberty file not found: $lib"
    puts "       Set SKY130_LIB (or PDK_ROOT), or edit the path in this script."
    exit 1
}

puts "INFO: liberty=$lib"

# cfg_name -> ENABLE_GATING value
set configs [list nogate 0 gated 1]

foreach {name gating} $configs {

    puts "=============================================================="
    puts "INFO: synthesising pe_mixed  ENABLE_GATING=$gating  ($name)"
    puts "=============================================================="

    yosys design -reset

    yosys read_verilog -sv $rtl/icg_cell.sv
    yosys read_verilog -sv $rtl/pe_mixed.sv

    # Bake the gating parameter in before elaboration so the ICG instances are
    # either present or structurally absent - not optimised away later.
    yosys chparam -set ENABLE_GATING $gating pe_mixed
    yosys hierarchy -check -top pe_mixed

    yosys synth -top pe_mixed -flatten

    # Map sequential elements. dfflibmap also handles the ICG enable latch when
    # the Liberty provides one; if it does not, the check below will say so.
    yosys dfflibmap -liberty $lib
    yosys abc -liberty $lib
    yosys opt_clean

    # --- Guard: unmapped generic cells would invalidate the power numbers ----
    # Anything still named $_... never reached a real standard cell, so its
    # capacitance is unknown and the gate-level power estimate would be wrong.
    if {[catch {yosys check -assert} msg]} {
        puts "WARN: yosys check reported issues for '$name': $msg"
    }

    yosys tee -o $here/reports/pe_mixed_${name}_stat.txt stat -liberty $lib
    yosys write_verilog -noattr $here/reports/pe_mixed_${name}.v

    puts "INFO: -> reports/pe_mixed_${name}_stat.txt"
    puts "INFO: -> reports/pe_mixed_${name}.v"
}

puts ""
puts "=============================================================="
puts "NEXT STEPS"
puts "  1. Critical path + power:  sta  sta_pe.tcl   (per netlist)"
puts "  2. Gate-level activity  :  cd ../../tb && make TEST=pe_mixed_gl ..."
puts "  3. ICG delta            :  ./asic_power_sweep.sh"
puts ""
puts "  Compare pe_mixed_nogate_stat.txt vs pe_mixed_gated_stat.txt for the"
puts "  AREA cost of gating; the power BENEFIT comes from step 3."
puts "=============================================================="
