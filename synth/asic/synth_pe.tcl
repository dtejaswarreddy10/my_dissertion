# ----------------------------------------------------------------------------
# Yosys ASIC synthesis of a SINGLE PE at 45nm (gate count + area + timing proxy).
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 1 / Week 3 deliverable
#           "Run Yosys synthesis on single PE only: gate count, critical path."
#
# Run as a Tcl script so the Liberty path can come from the environment:
#   yosys -c synth_pe.tcl
#
# Provide the FreePDK45 (or Nangate45) Liberty via env var FREEPDK45_LIB,
# e.g.  setenv FREEPDK45_LIB /path/to/NangateOpenCellLibrary_typical.lib
# or edit the fallback path below.
#
# Critical-path note: Yosys' `abc -liberty` prints a longest-path delay estimate
# in its log. For publication-quality timing, feed the written netlist to
# OpenSTA (per the dissertation plan, Section 3.4).
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
    puts "       Set FREEPDK45_LIB or edit the path in this script."
    exit 1
}

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
yosys write_verilog -noattr $here/reports/pe_netlist_freepdk45.v

puts "INFO: PE gate-count/area report -> $here/reports/pe_stat.txt"
puts "INFO: gate-level netlist        -> $here/reports/pe_netlist_freepdk45.v"
puts "INFO: for critical path, run OpenSTA on the netlist above."
