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

VERILOG_SOURCES := $(RTL)/pe_int8.sv $(RTL)/systolic_array.sv \
                   $(RTL)/pe_mixed.sv $(RTL)/systolic_array_mixed.sv \
                   $(RTL)/icg_cell.sv $(RTL)/sparsity_decoder.sv \
                   $(RTL)/sparse_feeder.sv $(RTL)/systolic_array_sparse.sv

ifeq ($(TEST),pe)
  TOPLEVEL := pe_int8
  MODULE   := test_pe_int8
else ifeq ($(TEST),array)
  TOPLEVEL := systolic_array
  MODULE   := test_systolic_array
else ifeq ($(TEST),pe_mixed)
  TOPLEVEL := pe_mixed
  MODULE   := test_pe_mixed
else ifeq ($(TEST),array_mixed)
  TOPLEVEL := systolic_array_mixed
  MODULE   := test_array_mixed
else ifeq ($(TEST),decoder)
  TOPLEVEL := sparsity_decoder
  MODULE   := test_sparsity_decoder
else ifeq ($(TEST),array_sparse)
  TOPLEVEL := systolic_array_sparse
  MODULE   := test_array_sparse
else ifeq ($(TEST),pe_mixed_gl)
  # Gate-level: run the RTL testbench against the synthesised netlist. The
  # netlist keeps pe_mixed's port list, so a pass is also an equivalence check.
  TOPLEVEL := pe_mixed
  MODULE   := test_pe_mixed
  ifndef GL_NETLIST
    $(error TEST=pe_mixed_gl needs GL_NETLIST=<synth/asic/reports/pe_mixed_*.v>)
  endif
  ifndef GL_LIB_V
    $(error TEST=pe_mixed_gl needs GL_LIB_V=<standard-cell simulation models .v>)
  endif
  VERILOG_SOURCES := $(GL_LIB_V) $(GL_NETLIST)
else
  $(error TEST must be one of: pe | array | pe_mixed | array_mixed | decoder | array_sparse | pe_mixed_gl (got '$(TEST)'))
endif

# ENABLE_GATING is a module parameter, overridable per run for the PPA sweep.
ifdef GATING
  ifeq ($(SIM),icarus)
    COMPILE_ARGS += -PENABLE_GATING=$(GATING)
  else
    COMPILE_ARGS += -gENABLE_GATING=$(GATING)
  endif
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

  # COVERAGE=1 pulls in the bind-based covergroups (Questa only - Icarus
  # cannot compile covergroups, so never add this file for other sims).
  ifeq ($(COVERAGE),1)
    VERILOG_SOURCES += $(TB_DIR)/cov_pe_int8.sv
    COMPILE_ARGS    += +cover=bcesxf
    SIM_ARGS        += -coverage
  endif
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
