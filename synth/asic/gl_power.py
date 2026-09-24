#!/usr/bin/env python3
"""Gate-level dynamic-power estimate from a SAIF plus a Liberty library.

Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
Stage   : Month 3 / Section 3.4 step 3 - the ICG power number

This is the fallback for step 3 of the ASIC power flow when OpenSTA's
``report_power`` is unavailable or the Liberty lacks usable power arcs. It
implements the switching-power model directly:

    P_dyn = sum over nets of  0.5 * C_net * Vdd^2 * TC_net / T

where
    C_net  = sum of the input-pin capacitances the net drives (from Liberty)
    TC_net = toggle count on that net (from the SAIF)
    T      = SAIF observation window, converted to seconds
    Vdd    = Liberty nom_voltage

Only load capacitance is modelled - wire capacitance is unknown pre-layout, and
internal cell (short-circuit) power is not included. The absolute number is
therefore an underestimate. That is acceptable because the dissertation claim
is a RATIO: the same model applied to the ICG-off and ICG-on netlists under the
same stimulus isolates the clock-gating contribution, and the systematic
underestimate largely cancels. Report it as such, never as absolute silicon
power.

Usage:
    ./gl_power.py --liberty $FREEPDK45_LIB \
                  --netlist reports/pe_mixed_gated.v \
                  --saif    reports/activity_gated_sp50.saif \
                  --json    reports/power_gated_sp50.json

    ./gl_power.py --compare reports/power_nogate_sp50.json \
                            reports/power_gated_sp50.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Liberty capacitive_load_unit -> farads
CAP_UNIT_TO_F = {"ff": 1e-15, "pf": 1e-12, "nf": 1e-9}
# SAIF timescale -> seconds
TIME_UNIT_TO_S = {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


# ---------------------------------------------------------------------------
# Liberty
# ---------------------------------------------------------------------------

def parse_liberty(path):
    """Extract {cell: {pin: (direction, capacitance_F)}}, Vdd and the cap unit.

    Regex/brace-depth based rather than a full Liberty parser - sufficient for
    the Nangate/FreePDK45 style used here, and it fails loudly if the shape is
    unexpected instead of silently returning zeros. Output pins are recorded
    too, so a load lookup can tell "driver, no load" apart from "cell missing
    from the library" - only the latter corrupts the power number.
    """
    text = Path(path).read_text(errors="ignore")

    m = re.search(r"capacitive_load_unit\s*\(\s*([\d.]+)\s*,\s*(\w+)\s*\)", text)
    if m:
        cap_scale = float(m.group(1)) * CAP_UNIT_TO_F.get(m.group(2).lower(), 1e-15)
    else:
        cap_scale = 1e-15
        print("WARN: capacitive_load_unit not found; assuming 1 fF", file=sys.stderr)

    m = re.search(r"nom_voltage\s*:\s*([\d.]+)", text)
    vdd = float(m.group(1)) if m else 1.1
    if not m:
        print("WARN: nom_voltage not found; assuming 1.1 V", file=sys.stderr)

    cells = {}
    cur_cell = None
    cur_pin = None
    pin_dir = None
    pin_cap = None
    depth = 0
    cell_depth = pin_depth = -1

    token_re = re.compile(
        r"cell\s*\(\s*([\w\\/\[\]\.]+)\s*\)|"      # 1 cell name
        r"pin\s*\(\s*([\w\\\[\]\.]+)\s*\)|"        # 2 pin name
        r"direction\s*:\s*(\w+)|"                  # 3 direction
        r"capacitance\s*:\s*([\d.eE+-]+)|"         # 4 capacitance
        r"([{}])"                                  # 5 brace
    )

    for m in token_re.finditer(text):
        cell_name, pin_name, direction, cap, brace = m.groups()

        if cell_name:
            cur_cell = cell_name
            cells.setdefault(cur_cell, {})
            cell_depth = depth
        elif pin_name and cur_cell:
            cur_pin, pin_dir, pin_cap = pin_name, None, None
            pin_depth = depth
        elif direction and cur_pin:
            pin_dir = direction
        elif cap and cur_pin and pin_cap is None:
            pin_cap = float(cap)
        elif brace == "{":
            depth += 1
        elif brace == "}":
            depth -= 1
            if cur_pin is not None and depth <= pin_depth:
                if cur_cell:
                    cells[cur_cell][cur_pin] = (
                        pin_dir or "unknown",
                        (pin_cap or 0.0) * cap_scale,
                    )
                cur_pin = pin_dir = pin_cap = None
            if cur_cell is not None and depth <= cell_depth:
                cur_cell = None

    if not cells:
        raise RuntimeError(f"no cells parsed from Liberty {path}")
    return cells, vdd, cap_scale


# ---------------------------------------------------------------------------
# Gate-level netlist
# ---------------------------------------------------------------------------

INSTANCE_RE = re.compile(
    r"(?P<cell>[A-Za-z_]\w*)\s+(?P<inst>[\\\w\.\[\]$]+)\s*\((?P<conns>.*?)\)\s*;",
    re.DOTALL,
)
CONN_RE = re.compile(r"\.(?P<pin>\w+)\s*\(\s*(?P<net>[^)]*?)\s*\)")
SKIP_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg",
    "assign", "parameter", "localparam", "always", "initial", "generate",
}


def normalise_net(raw):
    """Return a comparable net name, or None for constants/concatenations."""
    net = raw.strip()
    if not net or net.startswith("{"):
        return None            # concatenation - skip, cannot attribute activity
    if re.fullmatch(r"\d*'[bdhoBDHO][0-9a-fA-FxzXZ_]+", net):
        return None            # constant tie
    if net.startswith("\\"):
        net = net[1:]
    return net.strip()


def parse_netlist(path):
    """Return {net: capacitance_F_placeholder_list} as {net: [(cell, pin)]}."""
    text = Path(path).read_text(errors="ignore")
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    loads = {}
    n_inst = 0
    for m in INSTANCE_RE.finditer(text):
        cell = m.group("cell")
        if cell in SKIP_KEYWORDS:
            continue
        conns = m.group("conns")
        if "." not in conns:
            continue
        n_inst += 1
        for c in CONN_RE.finditer(conns):
            net = normalise_net(c.group("net"))
            if net is None:
                continue
            loads.setdefault(net, []).append((cell, c.group("pin")))

    if n_inst == 0:
        raise RuntimeError(f"no cell instances parsed from netlist {path}")
    return loads, n_inst


# ---------------------------------------------------------------------------
# SAIF
# ---------------------------------------------------------------------------

def parse_saif(path):
    """Return ({net: toggle_count}, duration_seconds)."""
    text = Path(path).read_text(errors="ignore")

    m = re.search(r"\(\s*TIMESCALE\s+([\d.]*)\s*(\w+)\s*\)", text, re.IGNORECASE)
    if m:
        scale = float(m.group(1) or 1) * TIME_UNIT_TO_S.get(m.group(2).lower(), 1e-12)
    else:
        scale = 1e-12
        print("WARN: TIMESCALE not found; assuming ps", file=sys.stderr)

    m = re.search(r"\(\s*DURATION\s+([\d.eE+-]+)\s*\)", text, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"no DURATION found in SAIF {path}")
    duration_s = float(m.group(1)) * scale
    if duration_s <= 0:
        raise RuntimeError(f"non-positive SAIF duration in {path}")

    # Net records look like:  (net_name (T0 n) (T1 n) (TX n) (TC n) (IG n))
    toggles = {}
    net_re = re.compile(
        r"\(\s*([\w\\\[\]\.\$]+)\s*\(\s*T0\b.*?\(\s*TC\s+(\d+)\s*\)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in net_re.finditer(text):
        name = m.group(1)
        if name.startswith("\\"):
            name = name[1:]
        # Leaf name: the PE netlist is flattened, so hierarchy is not needed.
        leaf = name.split("/")[-1]
        toggles[leaf] = toggles.get(leaf, 0) + int(m.group(2))

    if not toggles:
        raise RuntimeError(f"no net toggle records parsed from SAIF {path}")
    return toggles, duration_s


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def estimate(liberty, netlist, saif):
    cells, vdd, _ = parse_liberty(liberty)
    loads, n_inst = parse_netlist(netlist)
    toggles, duration_s = parse_saif(saif)

    total_p = 0.0
    matched = unmatched_net = unknown_pin = 0
    clock_p = 0.0
    per_net = []

    for net, pins in loads.items():
        tc = toggles.get(net)
        if tc is None:
            unmatched_net += 1
            continue
        matched += 1

        c_net = 0.0
        for cell, pin in pins:
            entry = cells.get(cell, {}).get(pin)
            if entry is None:
                # Cell or pin absent from the Liberty: its load is genuinely
                # unknown, so the estimate is short by that amount.
                unknown_pin += 1
                continue
            direction, cap = entry
            if direction == "input":
                c_net += cap

        if c_net == 0.0:
            continue

        p = 0.5 * c_net * (vdd ** 2) * tc / duration_s
        total_p += p
        per_net.append((net, c_net, tc, p))
        # Gated-clock nets are the ones the ICG claim rests on.
        if "clk" in net.lower() or "gclk" in net.lower():
            clock_p += p

    per_net.sort(key=lambda r: -r[3])

    return {
        "netlist": str(netlist),
        "saif": str(saif),
        "vdd_v": vdd,
        "duration_s": duration_s,
        "instances": n_inst,
        "nets_in_netlist": len(loads),
        "nets_matched_in_saif": matched,
        "nets_unmatched": unmatched_net,
        "unknown_pin_caps": unknown_pin,
        "dynamic_power_w": total_p,
        "clock_net_power_w": clock_p,
        "top_nets": [
            {"net": n, "cap_f": c, "toggles": t, "power_w": p}
            for n, c, t, p in per_net[:15]
        ],
    }


def print_report(res):
    print("=" * 62)
    print(f"Gate-level dynamic power : {Path(res['netlist']).name}")
    print("=" * 62)
    print(f"  SAIF                  : {Path(res['saif']).name}")
    print(f"  Vdd                   : {res['vdd_v']:.3f} V")
    print(f"  Observation window    : {res['duration_s'] * 1e9:.3f} ns")
    print(f"  Cell instances        : {res['instances']}")
    print(f"  Nets matched in SAIF  : {res['nets_matched_in_saif']}"
          f" / {res['nets_in_netlist']}")
    if res["nets_unmatched"]:
        print(f"  WARN unmatched nets   : {res['nets_unmatched']}"
              " (activity missing - power underestimated)")
    if res["unknown_pin_caps"]:
        print(f"  WARN unknown pin caps : {res['unknown_pin_caps']}")
    print("-" * 62)
    print(f"  DYNAMIC POWER         : {res['dynamic_power_w'] * 1e3:.6f} mW")
    print(f"    of which clock nets : {res['clock_net_power_w'] * 1e3:.6f} mW")
    print("-" * 62)
    print("  Highest-power nets:")
    for e in res["top_nets"][:8]:
        print(f"    {e['net']:<28} {e['toggles']:>8} tog "
              f"{e['cap_f'] * 1e15:>8.2f} fF  {e['power_w'] * 1e3:>10.6f} mW")
    print("=" * 62)


def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    pa, pb = a["dynamic_power_w"], b["dynamic_power_w"]
    ca, cb = a["clock_net_power_w"], b["clock_net_power_w"]

    print("=" * 62)
    print("ICG power delta (gate level, same stimulus)")
    print("=" * 62)
    print(f"  reference : {Path(a['netlist']).name:<28} {pa * 1e3:10.6f} mW")
    print(f"  measured  : {Path(b['netlist']).name:<28} {pb * 1e3:10.6f} mW")
    if pa > 0:
        print(f"  reduction : {100.0 * (pa - pb) / pa:.2f} %")
    print(f"  clock-net : {ca * 1e3:.6f} -> {cb * 1e3:.6f} mW", end="")
    if ca > 0:
        print(f"   ({100.0 * (ca - cb) / ca:.2f} % less)")
    else:
        print()
    print("=" * 62)
    print("  Report as a RATIO. Load capacitance only: no wire or internal")
    print("  cell power, so absolute values underestimate silicon.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liberty")
    ap.add_argument("--netlist")
    ap.add_argument("--saif")
    ap.add_argument("--json", help="write the result as JSON")
    ap.add_argument("--compare", nargs=2, metavar=("REF_JSON", "MEAS_JSON"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0

    if not (args.liberty and args.netlist and args.saif):
        ap.error("--liberty, --netlist and --saif are required "
                 "(or use --compare)")

    res = estimate(args.liberty, args.netlist, args.saif)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"INFO: JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
