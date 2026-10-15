# ----------------------------------------------------------------------------
# OpenSTA: critical path + activity-annotated power for a PE gate-level
# netlist, on sky130_fd_sc_hd.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 / Section 3.4 (ASIC timing + the ICG power claim), sky130 track
#
# sky130 counterpart of ../asic/sta_pe.tcl. CLK_NS defaults to 10.0 ns here
# (not 4.0 ns) - sky130 is a 130nm node, meaningfully slower per gate than the
# 45nm assumption the FreePDK45 track's default was tuned for. Treat 10 ns as
# a safe starting point to find the real achievable critical path, then
# tighten once you have a number, rather than reusing a 45nm-tuned constraint
# that could falsely look like a timing failure on a slower process.
#
# Run:
#   sta -exit -f sta_pe.tcl                       # timing only, default netlist
#   NETLIST=reports/pe_mixed_gated.v \
#   SAIF=reports/activity_gated_sp50.saif \
#   sta -exit -f sta_pe.tcl                       # timing + annotated power
#
# Environment:
#   SKY130_LIB     Liberty (required; falls back to PDK_ROOT-derived path)
#   NETLIST        gate-level .v from synth_pe_mixed.tcl (default: gated)
#   TOP            top module            (default pe_mixed)
#   CLK_NS         clock period in ns    (default 10.0)
#   SAIF           optional SAIF for activity-annotated dynamic power
#   SAIF_SCOPE     instance path the SAIF was captured under (default: TOP)
#
# Power reported WITHOUT a SAIF uses OpenSTA's default activity assumptions and
# is only useful as a sanity check. The dissertation number must come from an
# annotated run - that is the whole point of the gate-level flow.
# ----------------------------------------------------------------------------

proc env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default
}

set here [file normalize [file dirname [info script]]]

if {[info exists ::env(SKY130_LIB)]} {
    set lib $::env(SKY130_LIB)
} elseif {[info exists ::env(PDK_ROOT)]} {
    set lib "$::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
} else {
    set lib "/path/to/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
    ;# <-- EDIT if neither SKY130_LIB nor PDK_ROOT is set
}

set top     [env_or TOP           pe_mixed]
set netlist [env_or NETLIST       $here/reports/pe_mixed_gated.v]
set clk_ns  [env_or CLK_NS        10.0]
set saif    [env_or SAIF          ""]

if {![file exists $lib]} {
    puts "ERROR: Liberty not found: $lib  (set SKY130_LIB or PDK_ROOT)"
    exit 1
}
if {![file exists $netlist]} {
    puts "ERROR: netlist not found: $netlist  (run: yosys -c synth_pe_mixed.tcl)"
    exit 1
}

set saif_scope [env_or SAIF_SCOPE $top]
set tag [file rootname [file tail $netlist]]
file mkdir $here/reports

puts "INFO: lib=$lib"
puts "INFO: netlist=$netlist  top=$top  clk=${clk_ns}ns"

read_liberty $lib
read_verilog $netlist
link_design $top

# --- Constraints --------------------------------------------------------------
# clk is the only primary clock. gclk_mac / gclk_acc / gclk_sub are DERIVED
# inside the PE by the ICG, so they are generated clocks, not a second domain
# - declaring them this way is what keeps the design single-clock-domain in
# the timing view and backs the "no CDC" claim in the dissertation.
create_clock -name clk -period $clk_ns [get_ports clk]

set icg_outs [get_pins -quiet -hierarchical *gclk*]
if {$icg_outs ne ""} {
    foreach p $icg_outs {
        catch {
            create_generated_clock -name gclk_[get_name $p] \
                -source [get_ports clk] -divide_by 1 $p
        }
    }
    puts "INFO: declared [llength $icg_outs] gated clock(s) as generated clocks"
}

# Reasonable block-level boundary conditions for a PE characterised standalone.
#set_input_delay  -clock clk [expr {$clk_ns * 0.2}] [remove_from_collection [all_inputs] [get_ports clk]]
foreach port [all_inputs] {
    if {[get_name $port] ne "clk"} {
        set_input_delay -clock clk [expr {$clk_ns * 0.2}] $port
    }
}
set_output_delay -clock clk [expr {$clk_ns * 0.2}] [all_outputs]
catch { set_load 0.005 [all_outputs] }

# --- Timing -------------------------------------------------------------------
report_checks -path_delay max -digits 4 > $here/reports/sta_${tag}_setup.rpt
report_checks -path_delay min -digits 4 > $here/reports/sta_${tag}_hold.rpt
report_check_types -max_slew -max_capacitance -violators \
    > $here/reports/sta_${tag}_drc.rpt

puts "INFO: setup/hold reports -> reports/sta_${tag}_*.rpt"
report_checks -path_delay max -digits 4

# --- Power --------------------------------------------------------------------
if {$saif ne "" && [file exists $saif]} {
    puts "INFO: annotating activity from $saif (scope $saif_scope)"
    if {[catch {read_saif -scope $saif_scope $saif} msg]} {
        puts "WARN: read_saif failed ($msg)"
        puts "WARN: falling back to gl_power.py for the dynamic-power estimate"
    }
    set pwr_file $here/reports/power_${tag}_saif.rpt
} else {
    puts "WARN: no SAIF supplied - power below uses default activity and is NOT"
    puts "WARN: the number to quote. Produce a SAIF via asic_power_sweep.sh."
    set pwr_file $here/reports/power_${tag}_default.rpt
}

if {[catch {
    report_power -digits 6 > $pwr_file
    report_power -digits 6
} msg]} {
    puts "WARN: report_power unavailable in this OpenSTA build ($msg)"
    puts "WARN: use gl_power.py, which derives dynamic power from SAIF + Liberty"
} else {
    puts "INFO: power report -> $pwr_file"
}
