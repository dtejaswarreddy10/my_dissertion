#-----------------------------------------------------------------------------
# cocotb Makefile - baseline INT8 systolic array verification
#
#   make              # PE test        (TOPLEVEL=pe_int8)
#   make TEST=array   # array GEMM test(TOPLEVEL=systolic_array)
#   make TEST=array TB_MATRICES=1000   # full random regression
#   make SIM=questa   # override simulator (icarus | verilator | questa | vcs | xcelium)
#   make waves        # run with VCD dump enabled
#   make clean
#-----------------------------------------------------------------------------

SIM           ?= icarus
TOPLEVEL_LANG ?= verilog
TEST          ?= pe

TB_DIR := $(shell pwd)
RTL    := $(TB_DIR)/../rtl

VERILOG_SOURCES := $(RTL)/pe_int8.sv $(RTL)/systolic_array.sv

ifeq ($(TEST),pe)
  TOPLEVEL := pe_int8
  MODULE   := test_pe_int8
else ifeq ($(TEST),array)
  TOPLEVEL := systolic_array
  MODULE   := test_systolic_array
else
  $(error TEST must be 'pe' or 'array', got '$(TEST)')
endif

# SystemVerilog (always_ff, logic, unpacked arrays) needs the 2012 standard.
ifeq ($(SIM),icarus)
  COMPILE_ARGS += -g2012
endif
ifeq ($(SIM),verilator)
  EXTRA_ARGS += --trace --trace-structs
endif
ifneq ($(filter $(SIM),questa modelsim),)
  # vlog infers SV from the .sv extension; -sv makes it explicit.
  COMPILE_ARGS += -sv
  # cocotb drives the sim from Python, so run vsim headless in 64-bit mode.
  SIM_ARGS += -64 -c
  # Uncomment if your Questa install defaults to 32-bit binaries:
  # COMPILE_ARGS += -64
endif

# Test knobs consumed by the Python testbenches.
export TB_SEED     ?= 1
export TB_N        ?= 8
export TB_MATRICES ?= 50

COCOTB_RESULTS_FILE := results_$(TEST).xml

include $(shell cocotb-config --makefiles)/Makefile.sim

# Defined after the include so cocotb's 'sim' stays the default goal.
waves:
	$(MAKE) WAVES=1 SIM=$(SIM) TEST=$(TEST)

.PHONY: waves
