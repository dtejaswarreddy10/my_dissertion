puts "===== OPENROAD POWER COMMAND CHECK ====="

foreach cmd {read_saif read_vcd report_power set_power_activity} {
    if {[llength [info commands $cmd]] > 0} {
        puts "$cmd : AVAILABLE"
    } else {
        puts "$cmd : NOT AVAILABLE"
    }
}

puts "===== END CHECK ====="
exit
