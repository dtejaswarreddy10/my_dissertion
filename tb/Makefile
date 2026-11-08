#-----------------------------------------------------------------------------
# cocotb Makefile - Mixed-Precision Sparsity-Aware Systolic Array Verification
#
# Supported simulators:
#   questa | modelsim | icarus | verilator | vcs | xcelium
#
# Usage:
#   make SIM=questa TEST=pe
#   make SIM=questa TEST=array
#   make SIM=questa TEST=pe_mixed
#   make SIM=questa TEST=array_mixed
#   make SIM=questa TEST=decoder
#   make SIM=questa TEST=array_sparse
#   make SIM=questa TEST=pe WAVES=1
#   make SIM=questa TEST=pe COVERAGE=1
#-----------------------------------------------------------------------------

SIM           ?= questa
TOPLEVEL_LANG ?= verilog
TEST          ?= pe

TB_DIR := $(shell pwd)
RTL    := $(TB_DIR)/../src

#----------------------------------------------------------------------------
# Select DUT, Cocotb module and RTL sources
#----------------------------------------------------------------------------

ifeq ($(TEST),pe)

TOPLEVEL := pe_int8
COCOTB_TOPLEVEL := pe_int8
MODULE := test_pe_int8
COCOTB_TEST_MODULES := test_pe_int8

VERILOG_SOURCES := \
    $(RTL)/pe_int8.sv

else ifeq ($(TEST),array)

TOPLEVEL := systolic_array
COCOTB_TOPLEVEL := systolic_array
MODULE := test_systolic_array
COCOTB_TEST_MODULES := test_systolic_array

VERILOG_SOURCES := \
    $(RTL)/pe_int8.sv \
    $(RTL)/systolic_array.sv

else ifeq ($(TEST),pe_mixed)

TOPLEVEL := pe_mixed
COCOTB_TOPLEVEL := pe_mixed
MODULE := test_pe_mixed
COCOTB_TEST_MODULES := test_pe_mixed

VERILOG_SOURCES := \
    $(RTL)/pe_mixed.sv

else ifeq ($(TEST),pe_mixed_250)

TOPLEVEL := pe_mixed
COCOTB_TOPLEVEL := pe_mixed
MODULE := test_pe_mixed_250mhz
COCOTB_TEST_MODULES := test_pe_mixed_250mhz

VERILOG_SOURCES := \
    $(RTL)/pe_mixed.sv

else ifeq ($(TEST),array_mixed)

TOPLEVEL := systolic_array_mixed
COCOTB_TOPLEVEL := systolic_array_mixed
MODULE := test_array_mixed
COCOTB_TEST_MODULES := test_array_mixed

VERILOG_SOURCES := \
    $(RTL)/pe_mixed.sv \
    $(RTL)/systolic_array_mixed.sv

else ifeq ($(TEST),decoder)

TOPLEVEL := sparsity_decoder
COCOTB_TOPLEVEL := sparsity_decoder
MODULE := test_sparsity_decoder
COCOTB_TEST_MODULES := test_sparsity_decoder

VERILOG_SOURCES := \
    $(RTL)/sparsity_decoder.sv

else ifeq ($(TEST),array_sparse)

TOPLEVEL := systolic_array_sparse
COCOTB_TOPLEVEL := systolic_array_sparse
MODULE := test_array_sparse
COCOTB_TEST_MODULES := test_array_sparse

VERILOG_SOURCES := \
    $(RTL)/pe_mixed.sv \
    $(RTL)/icg_cell.sv \
    $(RTL)/sparsity_decoder.sv \
    $(RTL)/sparse_feeder.sv \
    $(RTL)/systolic_array_sparse.sv

else ifeq ($(TEST),pe_mixed_gl)

TOPLEVEL := pe_mixed
COCOTB_TOPLEVEL := pe_mixed
MODULE := test_pe_mixed
COCOTB_TEST_MODULES := test_pe_mixed

ifndef GL_NETLIST
$(error TEST=pe_mixed_gl requires GL_NETLIST=<netlist.v>)
endif

ifndef GL_LIB_V
$(error TEST=pe_mixed_gl requires GL_LIB_V=<stdcell_lib.v>)
endif

VERILOG_SOURCES := \
    $(GL_LIB_V) \
    $(GL_NETLIST)

else ifeq ($(TEST),pe_mixed_250_gl)

TOPLEVEL := pe_mixed
COCOTB_TOPLEVEL := pe_mixed
MODULE := test_pe_mixed_250mhz
COCOTB_TEST_MODULES := test_pe_mixed_250mhz

ifndef GL_NETLIST
$(error TEST=pe_mixed_250_gl requires GL_NETLIST=<netlist.v>)
endif

ifndef GL_LIB_V
$(error TEST=pe_mixed_250_gl requires GL_LIB_V=<stdcell_lib.v>)
endif

VERILOG_SOURCES := \
    $(GL_LIB_V) \
    $(GL_NETLIST)

else ifeq ($(TEST),pe_int8_gl)

TOPLEVEL := pe_int8
COCOTB_TOPLEVEL := pe_int8
MODULE := test_pe_int8
COCOTB_TEST_MODULES := test_pe_int8

ifndef GL_NETLIST
$(error TEST=pe_int8_gl requires GL_NETLIST=<netlist.v>)
endif

ifndef GL_LIB_V
$(error TEST=pe_int8_gl requires GL_LIB_V=<stdcell_lib.v>)
endif

VERILOG_SOURCES := \
    $(GL_LIB_V) \
    $(GL_NETLIST)

else
$(error Invalid TEST=$(TEST))
endif

#----------------------------------------------------------------------------
# Parameter override
#----------------------------------------------------------------------------

ifdef GATING
ifeq ($(SIM),icarus)
COMPILE_ARGS += -PENABLE_GATING=$(GATING)
else
COMPILE_ARGS += -gENABLE_GATING=$(GATING)
endif
endif

#----------------------------------------------------------------------------
# Simulator specific options
#----------------------------------------------------------------------------

ifeq ($(SIM),icarus)
COMPILE_ARGS += -g2012
endif

ifeq ($(SIM),verilator)
EXTRA_ARGS += --trace --trace-structs
endif

ifneq ($(filter $(SIM),questa modelsim),)

COMPILE_ARGS += -sv

SIM_ARGS += -64 -c
SIM_ARGS += -no_autoacc
SIM_ARGS += -wlf sim_build/vsim.wlf

# Zero-delay gate-level runs: cocotb restarts the clock each test, which trips
# the std-cell $width min-pulse checks (reported as "Errors", not mismatches).
ifneq ($(filter $(TEST),pe_mixed_gl pe_int8_gl),)
SIM_ARGS += +notimingchecks
endif

ifeq ($(POWER),1)
SIM_ARGS += -do "$(TB_DIR)/power.do"
else ifeq ($(WAVES),1)
SIM_ARGS += -do "$(TB_DIR)/wave.do"
endif

ifeq ($(COVERAGE),1)
VERILOG_SOURCES += $(TB_DIR)/cov_pe_int8.sv
COMPILE_ARGS += +cover=bcesxf
SIM_ARGS += -coverage
endif

endif

#----------------------------------------------------------------------------
# Python environment
#----------------------------------------------------------------------------

export PYTHONPATH := $(TB_DIR):$(TB_DIR)/..:$(PYTHONPATH)

export TB_SEED ?= 1
export TB_N ?= 8
export TB_MATRICES ?= 50

COCOTB_RESULTS_FILE := results_$(TEST).xml

#----------------------------------------------------------------------------
# cocotb makefiles
#----------------------------------------------------------------------------

PYTHON ?= python3

include $(shell $(PYTHON) -m cocotb_tools.config --makefiles)/Makefile.sim

#----------------------------------------------------------------------------
# Utility targets
#----------------------------------------------------------------------------

waves:
	$(MAKE) SIM=$(SIM) TEST=$(TEST) WAVES=1

.PHONY: waves
