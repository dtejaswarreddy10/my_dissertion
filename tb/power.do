;# Questa power-tracking dump for the SAIF-based power sweep (synth/fpga/power_sweep.sh).
;#
;# power add MUST be issued before `run`, in the SAME live session that runs
;# the simulation - it cannot be retrofitted onto an already-finished .wlf by
;# reopening it afterward (confirmed empirically: `dataset open <wlf>` has no
;# live elaborated design, so a later `power add -r /*` fails with "No
;# Design!", and the subsequent `power report` fails with "Power checking is
;# not enabled" - no toggle data exists because tracking was never active
;# during the actual run). Keeping this as a separate .do from wave.do (rather
;# than adding `power add` unconditionally there) so ordinary WAVES=1 debug
;# runs don't pay the power-tracking overhead.
;#
;# SAIF_OUT must be set as an environment variable by the caller (see
;# synth/fpga/power_sweep.sh) - $env(...) reads it from the shell environment
;# make exported into this vsim invocation, the same mechanism that already
;# carries TB_SPARSITY/COCOTB_TESTCASE through successfully.
add wave -r /*
power add -r /*
run -all
power report -all -bsaif $env(SAIF_OUT)
