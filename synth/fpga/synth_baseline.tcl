# ----------------------------------------------------------------------------
# Vivado non-project synthesis + implementation for the baseline INT8 array.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 1 / Week 3 deliverable
#           "Synthesize baseline: record LUTs, FFs, DSP48s, Fmax, power."
#
# Usage:
#   vivado -mode batch -source synth_baseline.tcl
#   vivado -mode batch -source synth_baseline.tcl -tclargs <part> <clk_period_ns>
#
# Defaults: Kintex-7 xc7k70tfbv676-1, 4.0 ns clock (250 MHz target).
# Change the part to an Artix-7 device (e.g. xc7a200tfbg484-1) if your Vivado
# license does not cover Kintex-7.
# ----------------------------------------------------------------------------

set part [lindex $argv 0]
if {$part eq ""} { set part xc7k70tfbv676-1 }

set clk_ns [lindex $argv 1]
if {$clk_ns eq ""} { set clk_ns 4.0 }

set here    [file normalize [file dirname [info script]]]
set rtl_dir [file normalize $here/../../rtl]
set out_dir [file normalize $here/reports]
file mkdir $out_dir

puts "INFO: part=$part  clk_period=$clk_ns ns  rtl=$rtl_dir"

# --- Read RTL -----------------------------------------------------------------
read_verilog -sv $rtl_dir/pe_int8.sv
read_verilog -sv $rtl_dir/systolic_array.sv

# --- Synthesize (OUT OF CONTEXT) ----------------------------------------------
# The array exposes its 64 accumulators (8*8*32 = 2048 bits) plus the operand
# buses as top-level ports => ~2195 signals, far more than any FPGA package has
# physical pins (xc7k70t-fbv676 has only 300). For block-level area / Fmax /
# power characterization we synthesize OUT OF CONTEXT so Vivado does not insert
# I/O buffers or try to map these signals onto device pins.
synth_design -top systolic_array -part $part -mode out_of_context

# Clock constraint (applied to the netlist for timing-driven implementation).
create_clock -name clk -period $clk_ns [get_ports clk]

# --- Implement (place + route for post-route Fmax / power) --------------------
opt_design
place_design
route_design

# --- Optional: approximate the plan's "uniform 50% toggle" for vectorless power
# Wrapped in catch so an unsupported flag on your Vivado version cannot abort the flow.
if {[catch {
    set_switching_activity -default_toggle_rate 0.5 -default_static_probability 0.5
} msg]} {
    puts "WARN: could not set uniform switching activity (using Vivado defaults): $msg"
}

# --- Reports ------------------------------------------------------------------
report_utilization     -file $out_dir/util_post_route.rpt
report_timing_summary  -file $out_dir/timing_post_route.rpt
report_power           -file $out_dir/power_post_route.rpt

# --- Console summary ----------------------------------------------------------
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
puts "INFO: clock period      = $clk_ns ns"
puts "INFO: worst setup slack = $wns ns"
if {$wns ne "" && $wns >= 0} {
    set fmax [expr {1000.0 / ($clk_ns - $wns)}]
    puts "INFO: achievable Fmax  ~= [format %.1f $fmax] MHz"
} else {
    puts "INFO: timing not met at this period; relax clk_period and re-run for Fmax."
}
puts "INFO: reports written to $out_dir"
