#!/bin/bash
# ----------------------------------------------------------------------------
# asic_power_sweep.sh - gate-level ICG power characterisation (Section 3.4),
# sky130_fd_sc_hd track.
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
#
# sky130 counterpart of ../asic/asic_power_sweep.sh. Reuses ../asic/gl_power.py
# as-is (it is already PDK-agnostic: it parses whatever Liberty you hand it),
# rather than duplicating it.
#
# THIS is where the clock-gating claim is measured. The FPGA flow cannot host
# 64 independently gated PE clocks, so Vivado converts per-PE gating into
# flip-flop clock-enables - real clock-tree power never drops there. Here the
# ICG cells exist, so the ICG-off vs ICG-on delta is meaningful.
#
# KNOWN GAP (carried over from the FreePDK45 track, not yet fixed): the
# per-sparsity-point loop below passes TB_SPARSITY into the gate-level
# cocotb run, but tb/test_pe_mixed.py has no code path that reads TB_SPARSITY
# (only tb/test_array_sparse.py's test_power_stimulus does). Until a
# sparsity-controlled stimulus is added to test_pe_mixed.py, every "sparsity
# point" below drives the same stimulus - treat a single run as one data
# point, not yet a real 4-point curve.
#
# Flow, per sparsity point:
#   1. Yosys  : pe_mixed -> two sky130 netlists (ENABLE_GATING 0 and 1)
#   2. cocotb : re-run test_pe_mixed on EACH netlist -> VCD
#               (this doubles as RTL/netlist equivalence proof)
#   3. VCD    -> SAIF
#   4. OpenSTA report_power, or gl_power.py as the fallback model
#   5. Report the ICG-off vs ICG-on delta
#
# Usage:
#   export SKY130_LIB=/path/to/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
#   # or just: export PDK_ROOT=/path/to/your/openlane/pdk/root   (auto-derives SKY130_LIB)
#   export SKY130_VERILOG=/path/to/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v
#   ./asic_power_sweep.sh
#   ./asic_power_sweep.sh 0.50            # single sparsity point
#
# NOTE on SKY130_VERILOG: sky130_fd_sc_hd.v `includes other files (e.g.
# primitives.v) in some open_pdks layouts. If gate-level compilation fails on
# a missing `include, point your simulator's include path at the same
# verilog/ directory, or pre-concatenate the needed files - a well-known
# detail of sky130 gate-level sim setups, not specific to this project.
#
# Nothing is installed by this script.
# ----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TB_DIR="$ROOT/tb"
REP="$HERE/reports"
GL_POWER_PY="$HERE/../asic/gl_power.py"
SIM="${SIM:-questa}"
CLK_NS="${CLK_NS:-10.0}"

mkdir -p "$REP"

if [[ -z "${SKY130_LIB:-}" ]]; then
    if [[ -n "${PDK_ROOT:-}" ]]; then
        export SKY130_LIB="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
        echo "INFO: SKY130_LIB not set, derived from PDK_ROOT: $SKY130_LIB"
    else
        echo "ERROR: set SKY130_LIB, or set PDK_ROOT so it can be derived"
        exit 1
    fi
fi
[[ -f "$SKY130_LIB" ]] || { echo "ERROR: Liberty not found: $SKY130_LIB"; exit 1; }

if [[ $# -ge 1 ]]; then SPARSITIES=("$1"); else SPARSITIES=(0.00 0.25 0.50 0.75); fi

command -v yosys >/dev/null 2>&1 || { echo "ERROR: yosys not on PATH"; exit 1; }
command -v cocotb-config >/dev/null 2>&1 || { echo "ERROR: cocotb not active (venv?)"; exit 1; }

echo "=============================================================="
echo " ASIC gate-level ICG power sweep (sky130_fd_sc_hd)"
echo "   liberty : $SKY130_LIB"
echo "   sim     : $SIM     clk: ${CLK_NS} ns"
echo "   points  : ${SPARSITIES[*]}"
echo "=============================================================="

# --- 1. Synthesise both gating configurations --------------------------------
echo ""
echo "[1/5] Yosys synthesis (ENABLE_GATING = 0 and 1)"
( cd "$HERE" && yosys -c synth_pe_mixed.tcl )

for cfg in nogate gated; do
    [[ -f "$REP/pe_mixed_${cfg}.v" ]] || { echo "ERROR: missing $REP/pe_mixed_${cfg}.v"; exit 1; }
done

echo ""
echo "[2/5] Area cost of gating (from the stat reports)"
grep -E "Chip area|Number of cells" "$REP/pe_mixed_nogate_stat.txt" | sed 's/^/   nogate: /' || true
grep -E "Chip area|Number of cells" "$REP/pe_mixed_gated_stat.txt"  | sed 's/^/   gated : /' || true

# --- 2..4. Per sparsity point -------------------------------------------------
for s in "${SPARSITIES[@]}"; do
    tag="sp$(printf '%02d' "$(echo "$s * 100 / 1" | bc)")"
    echo ""
    echo "--------------------------------------------------------------"
    echo " Sparsity ${s}  (tag ${tag})"
    echo "--------------------------------------------------------------"

    for cfg in nogate gated; do
        netlist="$REP/pe_mixed_${cfg}.v"
        saif="$REP/activity_${cfg}_${tag}.saif"

        echo "[3/5] Gate-level simulation: $cfg @ sparsity $s"
        if [[ -z "${SKY130_VERILOG:-}" || ! -f "${SKY130_VERILOG}" ]]; then
            echo "   WARN: SKY130_VERILOG (cell simulation models) not set or missing."
            echo "   WARN: skipping gate-level sim for $cfg; supply it to enable SAIF capture."
            continue
        fi

        pushd "$TB_DIR" >/dev/null
        make clean >/dev/null
        # The netlist keeps pe_mixed's port list, so the RTL testbench runs
        # unchanged - a passing run is also a netlist equivalence check.
        # See the KNOWN GAP note above re: TB_SPARSITY on this test.
        make SIM="$SIM" TEST=pe_mixed_gl WAVES=1 \
             GL_NETLIST="$netlist" GL_LIB_V="$SKY130_VERILOG" \
             TB_SPARSITY="$s" TB_EXHAUSTIVE=0

        dump=""
        for cand in dump.vcd pe_mixed.vcd vsim.wlf; do
            [[ -f "$cand" ]] && dump="$cand" && break
        done
        [[ -n "$dump" ]] || { echo "ERROR: no waveform dump for $cfg"; popd >/dev/null; exit 1; }

        echo "[4/5] SAIF extraction -> $(basename "$saif")"
        if [[ "$dump" == *.wlf ]]; then
            vsim -c -do "dataset open $dump; power add -r /*; power report -all -bsaif $saif; quit -f"
        elif command -v vcd2saif >/dev/null 2>&1; then
            vcd2saif -input "$dump" -output "$saif"
        else
            echo "ERROR: need vcd2saif or a Questa .wlf to produce SAIF"
            popd >/dev/null; exit 1
        fi
        popd >/dev/null

        # --- OpenSTA first, gl_power.py as the documented fallback ----------
        if command -v sta >/dev/null 2>&1; then
            echo "[5/5] OpenSTA timing + power: $cfg"
            NETLIST="$netlist" SAIF="$saif" TOP=pe_mixed CLK_NS="$CLK_NS" \
                sta -exit -f "$HERE/sta_pe.tcl" | tee "$REP/sta_${cfg}_${tag}.log" || true
        else
            echo "   INFO: OpenSTA not on PATH - using gl_power.py only"
        fi

        python3 "$GL_POWER_PY" \
            --liberty "$SKY130_LIB" \
            --netlist "$netlist" \
            --saif    "$saif" \
            --json    "$REP/power_${cfg}_${tag}.json"
    done

    # --- 5. ICG delta for this sparsity point --------------------------------
    if [[ -f "$REP/power_nogate_${tag}.json" && -f "$REP/power_gated_${tag}.json" ]]; then
        echo ""
        python3 "$GL_POWER_PY" --compare \
            "$REP/power_nogate_${tag}.json" "$REP/power_gated_${tag}.json"
    fi
done

echo ""
echo "=============================================================="
echo " Done. For the dissertation:"
echo "   * ICG power benefit  -> the compare blocks above (per sparsity)"
echo "   * ICG area cost      -> pe_mixed_{nogate,gated}_stat.txt"
echo "   * PE critical path   -> reports/sta_pe_mixed_*_setup.rpt"
echo ""
echo " State clearly in the paper that the FPGA number is operand"
echo " isolation + clock ENABLE, that THIS flow is the ICG result, and"
echo " that it is characterised against the sky130 (130nm) node."
echo "=============================================================="
