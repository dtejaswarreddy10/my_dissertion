# ----------------------------------------------------------------------------
# Vivado synthesis + implementation sweep across the four design configurations.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 / Month 4 (M2.6 area delta, M3.6 power, Week 13 sweep)
#
# Produces, for every configuration, the numbers the dissertation reports:
#   LUTs, FFs, DSP48s, post-route Fmax, static/dynamic power, and (when a SAIF
#   is supplied) activity-annotated dynamic power.
#
# Usage:
#   vivado -mode batch -source synth_sweep.tcl
#   vivado -mode batch -source synth_sweep.tcl -tclargs <config> <part> <clk_ns> <saif>
#
#   <config> : baseline | mixed | sparse_nogate | full | all   (default: all)
#   <saif>   : optional .saif for activity-annotated power
#
# Configurations map 1:1 onto the plan's sweep:
#   baseline       systolic_array          dense INT8 reference
#   mixed          systolic_array_mixed    + INT8/INT4x2 precision modes
#   sparse_nogate  systolic_array_sparse   + 2:4 sparsity, ENABLE_GATING=0
#   full           systolic_array_sparse   + 2:4 sparsity, ENABLE_GATING=1
#
# The area delta for the "< 20% overhead" claim is (mixed - baseline); the
# power delta for the sparsity claim is (full - sparse_nogate) at matched
# stimulus.
#
# NOTE ON FPGA CLOCK GATING: a Kintex-7 has only ~32 BUFGs, so 64 PEs cannot
# each own a gated clock. icg_cell.sv's DEFAULT (no macro) path is the
# latch-based ICG structure meant for ASIC - on FPGA that is ~128 per-PE
# latch-gated clock nets (2/PE x 64 PEs) fighting for general fabric routing,
# not a defined result. This script therefore always reads the RTL with
# ICG_NO_GATING defined, which forces icg_cell.sv's honest FPGA passthrough
# (gclk = clk) - pe_mixed's operand isolation + the valid_q/acc_active holds
# then degenerate to a plain flip-flop clock-enable, bit-identical to the
# gated result (see pe_mixed.sv). The FPGA "full" number is therefore
# operand-isolation + clock-enable, NOT true ICG - report it as such. True
# ICG power lives in the Yosys/FreePDK45 flow, which must NOT define this
# macro (it needs the real latch-based ICG to synthesize a gated clock).
# ----------------------------------------------------------------------------

set config [lindex $argv 0]
if {$config eq ""} { set config all }

set part [lindex $argv 1]
if {$part eq ""} { set part xc7k70tfbv676-1 }

set clk_ns [lindex $argv 2]
if {$clk_ns eq ""} { set clk_ns 4.0 }

set saif [lindex $argv 3]

set here    [file normalize [file dirname [info script]]]
set rtl_dir [file normalize $here/../../rtl]
set out_dir [file normalize $here/reports]
file mkdir $out_dir

# config -> {top  extra_generics}
dict set cfg baseline      top systolic_array
dict set cfg baseline      gen {}
dict set cfg mixed         top systolic_array_mixed
dict set cfg mixed         gen {}
dict set cfg sparse_nogate top systolic_array_sparse
dict set cfg sparse_nogate gen {ENABLE_GATING=0}
dict set cfg full          top systolic_array_sparse
dict set cfg full          gen {ENABLE_GATING=1}

if {$config eq "all"} {
    set run_list {baseline mixed sparse_nogate full}
} else {
    if {![dict exists $cfg $config]} {
        puts "ERROR: unknown config '$config'. Use baseline|mixed|sparse_nogate|full|all"
        exit 1
    }
    set run_list [list $config]
}

set summary {}

foreach name $run_list {
    set top  [dict get $cfg $name top]
    set gen  [dict get $cfg $name gen]

    puts "=============================================================="
    puts "INFO: config=$name  top=$top  generics='$gen'  part=$part  clk=$clk_ns ns"
    puts "=============================================================="

    # Fresh in-memory design per configuration.
    catch {close_design}
    catch {close_project}

    read_verilog -sv [glob $rtl_dir/*.sv]

    # Out of context: the array exposes 2000+ accumulator/operand bits, far more
    # than any package has pins, so I/O buffer insertion must be suppressed.
    # -verilog_define (not read_verilog -define, which needs compile-unit-mode
    # filesets that this non-project flow does not use) forces ICG_NO_GATING.
    if {$gen eq ""} {
        synth_design -top $top -part $part -mode out_of_context \
            -verilog_define {ICG_NO_GATING}
    } else {
        synth_design -top $top -part $part -mode out_of_context \
            -verilog_define {ICG_NO_GATING} -generic $gen
    }

    create_clock -name clk -period $clk_ns [get_ports clk]

    opt_design
    place_design -directive ExtraTimingOpt
    phys_opt_design
    route_design
    # Post-route physical optimization to claw back sub-100ps residual slack
    # (retiming/replication only - no functional or latency change, so this is
    # safe regardless of what the RTL side is doing in parallel).
    phys_opt_design -directive AggressiveExplore

    # Activity annotation: a SAIF from the cocotb power stimulus if provided,
    # otherwise the plan's uniform 50% toggle placeholder.
    if {$saif ne "" && [file exists $saif]} {
        puts "INFO: annotating switching activity from $saif"
        read_saif $saif
        set act "saif"
    } else {
        if {[catch {
            set_switching_activity -default_toggle_rate 0.5 -default_static_probability 0.5
        } msg]} {
            puts "WARN: could not set uniform switching activity: $msg"
        }
        set act "uniform50"
    }

    report_utilization    -file $out_dir/util_${name}_${act}.rpt
    report_timing_summary -file $out_dir/timing_${name}_${act}.rpt
    report_power          -file $out_dir/power_${name}_${act}.rpt

    # --- Scrape the headline numbers for the comparison table ---------------
    set luts [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT}]]
    set ffs  [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set dsps [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == ARITHMETIC}]]

    set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
    if {$wns ne "" && $wns >= 0} {
        set fmax [format %.1f [expr {1000.0 / ($clk_ns - $wns)}]]
    } else {
        set fmax "TIMING_FAILED"
    }

    set dyn "n/a"
    catch { set dyn [format %.4f [get_property POWER_DYNAMIC [current_design]]] }

    lappend summary [list $name $luts $ffs $dsps $wns $fmax $dyn]
}

# --- Console summary table ----------------------------------------------------
puts ""
puts "=============================================================="
puts "SWEEP SUMMARY  (part=$part, clk=$clk_ns ns)"
puts "=============================================================="
puts [format "%-16s %8s %8s %6s %10s %10s %12s" \
      "config" "LUTs" "FFs" "DSPs" "WNS(ns)" "Fmax(MHz)" "Dyn(W)"]
foreach row $summary {
    lassign $row name luts ffs dsps wns fmax dyn
    puts [format "%-16s %8s %8s %6s %10s %10s %12s" $name $luts $ffs $dsps $wns $fmax $dyn]
}
puts ""
puts "INFO: reports written to $out_dir"
puts "INFO: area overhead claim = (mixed - baseline) / baseline"
puts "INFO: power claim         = (full vs sparse_nogate) at matched SAIF stimulus"
