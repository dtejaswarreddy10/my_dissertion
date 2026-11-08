# Load Liberty timing and power information
read_liberty $::env(FREEPDK45_LIB)

# Load combined technology and standard-cell LEF
read_lef $::env(FREEPDK45_LEF)

# Load design
read_verilog $::env(NETLIST)
link_design pe_mixed

# Create clock
create_clock -name clk -period 4.0 [get_ports clk]

# Apply temporary default activity
set_power_activity -global -activity 0.1 -duty 0.5

puts "===== OPENROAD POWER TRIAL ====="

report_power -digits 8

exit
