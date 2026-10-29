;# Questa waveform dump - loaded via `make waves SIM=questa` (WAVES=1 -do wave.do).
;# Recursively adds every signal under the toplevel so nothing needs listing
;# by name here as the DUT changes across TEST=pe/array/pe_mixed/... configs.
add wave -r /*
run -all
