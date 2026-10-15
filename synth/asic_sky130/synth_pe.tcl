# ----------------------------------------------------------------------------
# Yosys ASIC synthesis of a SINGLE PE on SkyWater sky130 (sky130_fd_sc_hd):
# gate count + area + timing proxy.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 1 / Week 3 deliverable, sky130 track
#           "Run Yosys synthesis on single PE only: gate count, critical path."
#
# This is the sky130 counterpart of ../asic/synth_pe.tcl (FreePDK45/Nangate45).
# Both tracks are kept side by side rather than one replacing the other.
#
# Run as a Tcl script so the Liberty path can come from the environment:
#   yosys -c synth_pe.tcl
#
# Liberty resolution order:
#   1. SKY130_LIB, if set                 (explicit override)
#   2. $PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
#      - the typical (tt, 25C, 1.8V) corner, derived from PDK_ROOT if an
#        OpenLane/volare install has already set it in your environment.
#   3. a placeholder path (edit below) if neither is set.
#
# Critical-path note: Yosys' `abc -liberty` prints a longest-path delay
# estimate in its log. For publication-quality timing, feed the written
# netlist to OpenSTA (sta_pe.tcl in this folder).
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

# --- Read + elaborate the PE only --------------------------------------------
yosys read_verilog -sv $rtl/pe_int8.sv
yosys hierarchy -check -top pe_int8

# --- Generic synthesis, then map to the standard-cell library ----------------
yosys synth -top pe_int8 -flatten
yosys dfflibmap -liberty $lib
yosys abc -liberty $lib
yosys opt_clean

# --- Reports ------------------------------------------------------------------
# Gate count (per cell type) + total cell area in the target library:
yosys tee -o $here/reports/pe_stat.txt stat -liberty $lib
# Gate-level netlist for downstream OpenSTA timing / gate-level power:
yosys write_verilog -noattr $here/reports/pe_netlist_sky130.v

puts "INFO: PE gate-count/area report -> $here/reports/pe_stat.txt"
puts "INFO: gate-level netlist        -> $here/reports/pe_netlist_sky130.v"
puts "INFO: for critical path, run OpenSTA (sta_pe.tcl) on the netlist above."
