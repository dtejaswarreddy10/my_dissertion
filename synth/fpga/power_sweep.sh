#!/bin/bash
# ----------------------------------------------------------------------------
# power_sweep.sh - end-to-end power-vs-sparsity characterisation (M3.6)
#
# Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
# Stage   : Month 3 Week 12 - "4-point power vs sparsity curve (paper Figure 3)"
#
# Pipeline, repeated at 0 / 25 / 50 / 75% activation sparsity:
#
#   1. cocotb power stimulus  -> VCD   (test_array_sparse.py::test_power_stimulus)
#   2. VCD                    -> SAIF  (vcd2saif, or Vivado's read_saif path)
#   3. SAIF + routed netlist  -> annotated dynamic power (synth_sweep.tcl)
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

    # --- 1. Simulate with waves to capture switching activity ---------------
    pushd "$TB_DIR" >/dev/null
    make clean >/dev/null
    make SIM="$SIM" TEST=array_sparse WAVES=1 \
         TB_SPARSITY="$s" TB_POWER_TILES="${TB_POWER_TILES:-16}"

    # Simulator-dependent dump name; take whichever exists.
    dump=""
    for cand in dump.vcd systolic_array_sparse.vcd vsim.wlf; do
        [[ -f "$cand" ]] && dump="$cand" && break
    done
    [[ -n "$dump" ]] || { echo "ERROR: no waveform dump produced"; exit 1; }

    # --- 2. Convert to SAIF -------------------------------------------------
    saif="$OUT_DIR/activity_${tag}.saif"
    if [[ "$dump" == *.wlf ]]; then
        # Questa native path: power add / power report writes SAIF directly.
        vsim -c -do "dataset open $dump; power add -r /*; power report -all -bsaif $saif; quit -f" \
            || { echo "ERROR: Questa SAIF export failed"; exit 1; }
    elif command -v vcd2saif >/dev/null 2>&1; then
        vcd2saif -input "$dump" -output "$saif"
    else
        echo "ERROR: need vcd2saif (Synopsys/Cadence) or a Questa .wlf to make SAIF"
        exit 1
    fi
    popd >/dev/null

    echo "INFO: SAIF written to $saif"

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
