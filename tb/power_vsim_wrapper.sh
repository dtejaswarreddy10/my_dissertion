#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SAIF_OUT:-}" ]]; then
    echo "ERROR: SAIF_OUT is not set" >&2
    exit 1
fi

do_file="sim_build/runsim.do"

if [[ ! -f "$do_file" ]]; then
    echo "ERROR: $do_file was not generated" >&2
    exit 1
fi

python - "$do_file" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text()

# Keep Questa alive after Cocotb's $finish so Tcl can export SAIF.
content = content.replace("-onfinish exit", "-onfinish stop")

export_line = "power report -all -bsaif $env(SAIF_OUT)"
if export_line not in content:
    marker = "\nquit\n"
    position = content.rfind(marker)
    if position < 0:
        raise SystemExit("ERROR: Could not find final quit in runsim.do")
    content = content[:position] + "\n" + export_line + content[position:]

path.write_text(content)

print("INFO: Updated runsim.do:")
for line in content.splitlines():
    if "-onfinish" in line or "power report" in line or line.strip() == "run -all":
        print("  " + line)
PY

exec "$@"
