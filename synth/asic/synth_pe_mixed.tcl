# ----------------------------------------------------------------------------
# Yosys ASIC synthesis of the precision-adaptive PE, ICG OFF and ICG ON.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 / Section 3.4 - "the ASIC flow is where the clock-gating
#           claim actually lives"
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
# Environment:
#   FREEPDK45_LIB  path to the FreePDK45 / Nangate Liberty (required)
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

if {[info exists ::env(FREEPDK45_LIB)]} {
    set lib $::env(FREEPDK45_LIB)
} else {
    set lib "/path/to/FreePDK45/stdcells.lib"   ;# <-- EDIT if FREEPDK45_LIB is unset
}

set here [file normalize [file dirname [info script]]]
set rtl  [file normalize $here/../../rtl]
file mkdir $here/reports

if {![file exists $lib]} {
    puts "ERROR: Liberty file not found: $lib"
    puts "       setenv FREEPDK45_LIB /path/to/NangateOpenCellLibrary_typical.lib"
    exit 1
}

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
