#!/bin/bash
# ----------------------------------------------------------------------------
# power_sweep.sh - end-to-end power-vs-sparsity characterisation (M3.6)
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 Week 12 - "4-point power vs sparsity curve (paper Figure 3)"
#
# Pipeline, repeated at 0 / 25 / 50 / 75% activation sparsity:
#
#   1. cocotb power stimulus, Questa power tracking live during the run
#      -> SAIF directly   (test_array_sparse.py::test_power_stimulus, tb/power.do)
#   2. SAIF + routed netlist -> annotated dynamic power (synth_sweep.tcl)
#
# (SAIF must come from a live-tracked run, not a post-hoc VCD/.wlf
# conversion after the fact - see tb/power.do's header comment for why.)
#
# Each sparsity point is run for BOTH gating configurations so the reported
# reduction is a single-variable comparison:
#     sparse_nogate : operand isolation off, no gating   (reference)
#     full          : operand isolation + gating on      (measurement)
#
# Usage:
#   ./power_sweep.sh                 # all four sparsity points, both configs
#   ./power_sweep.sh 0.50            # a single sparsity point
#
# Requires: an activated cocotb venv, a simulator (SIM env var, default questa),
# and Vivado on PATH. Nothing here is installed by this script.
# ----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TB_DIR="$ROOT/tb"
OUT_DIR="$HERE/reports/power_sweep"
SIM="${SIM:-questa}"
PART="${PART:-xc7k70tfbv676-1}"
CLK_NS="${CLK_NS:-4.0}"

mkdir -p "$OUT_DIR"

if [[ $# -ge 1 ]]; then
    SPARSITIES=("$1")
else
    SPARSITIES=(0.00 0.25 0.50 0.75)
fi

echo "=============================================================="
echo " Power vs sparsity sweep"
echo "   sim=$SIM  part=$PART  clk=${CLK_NS}ns"
echo "   sparsity points: ${SPARSITIES[*]}"
echo "   output: $OUT_DIR"
echo "=============================================================="

command -v vivado >/dev/null 2>&1 || { echo "ERROR: vivado not on PATH"; exit 1; }
command -v cocotb-config >/dev/null 2>&1 || { echo "ERROR: cocotb not active (venv?)"; exit 1; }

for s in "${SPARSITIES[@]}"; do
    tag="sp$(printf '%02d' "$(echo "$s * 100 / 1" | bc)")"

    echo ""
    echo "--------------------------------------------------------------"
    echo " Sparsity ${s} -> tag ${tag}"
    echo "--------------------------------------------------------------"

    # --- 1+2. Simulate with power tracking enabled, SAIF written directly ---
    # TESTCASE/COCOTB_TESTCASE restricts the run to test_power_stimulus only -
    # without this, TEST=array_sparse runs all 6 tests in the module into the
    # same session, which would mix test_sparse_int8_random/
    # test_zero_weight_group_gates_cleanly/etc. activity into the SAIF and
    # corrupt the single-variable sparsity comparison. cocotb's variable name
    # for this changed across versions (TESTCASE historically, COCOTB_TESTCASE
    # in newer releases), so both are passed - whichever the installed cocotb
    # honors takes effect; check the sim log to confirm only one test ran.
    #
    # POWER=1 selects tb/power.do, which issues `power add -r /*` BEFORE
    # `run -all` in this same live session, then `power report -bsaif
    # $env(SAIF_OUT)` right after. This is NOT equivalent to running with
    # WAVES=1 and converting the .wlf afterward: Questa's power tracking must
    # be enabled during the live run - reopening a finished .wlf via `dataset
    # open` has no elaborated design for `power add` to attach to ("No
    # Design!"), and the resulting `power report` finds no data ("Power
    # checking is not enabled"). That two-stage approach was tried and
    # produced an empty/nonexistent SAIF silently (no hard error) - confirmed
    # by the downstream Vivado run falling back to its uniform50 default and
    # naming its reports accordingly instead of erroring.
    saif="$OUT_DIR/activity_${tag}.saif"
    pushd "$TB_DIR" >/dev/null
    make clean >/dev/null
    SAIF_OUT="$saif" SIM_CMD_PREFIX="$TB_DIR/power_vsim_wrapper.sh" make SIM="$SIM" TEST=array_sparse POWER=1 \
         TESTCASE=test_power_stimulus COCOTB_TESTCASE=test_power_stimulus \
         TB_SPARSITY="$s" TB_POWER_TILES="${TB_POWER_TILES:-16}"
    popd >/dev/null

    [[ -s "$saif" ]] || { echo "ERROR: $saif was not written (or is empty) - check the sim log above for a 'Power checking is not enabled' message"; exit 1; }
    echo "INFO: SAIF written to $saif ($(wc -l < "$saif") lines)"

    # --- 3. Annotated power for both gating configurations ------------------
    for cfg in sparse_nogate full; do
        echo "INFO: Vivado power run  config=$cfg  sparsity=$s"
        vivado -mode batch -nojournal -nolog \
               -source "$HERE/synth_sweep.tcl" \
               -tclargs "$cfg" "$PART" "$CLK_NS" "$saif" \
            | tee "$OUT_DIR/vivado_${cfg}_${tag}.log"

        cp "$HERE/reports/power_${cfg}_saif.rpt" \
           "$OUT_DIR/power_${cfg}_${tag}.rpt" 2>/dev/null || true
    done
done

echo ""
echo "=============================================================="
echo " Sweep complete. Extract the curve with:"
echo "   grep -H 'Dynamic (W)' $OUT_DIR/power_*_sp*.rpt"
echo ""
echo " Report as: dynamic power vs sparsity, sparse_nogate vs full."
echo " Remember: on FPGA this is operand isolation + clock ENABLE,"
echo " not true ICG. True ICG power comes from the Yosys/FreePDK45 flow."
echo "=============================================================="
