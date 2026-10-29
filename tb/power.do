
;# Questa power tracking for Cocotb SAIF export
;# Cocotb controls the simulation lifecycle.
;# Do not use run -all in this file.

if {![info exists env(SAIF_OUT)]} {
    error "SAIF_OUT environment variable is missing"
}

set saif_path $env(SAIF_OUT)

puts "INFO: SAIF output path: $saif_path"

;# Enable power tracking before Cocotb stimulus runs.
power add -r /*
power on

puts "INFO: Questa power tracking enabled"
