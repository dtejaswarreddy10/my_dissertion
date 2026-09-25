#-----------------------------------------------------------------------------
# cocotb Makefile - Mixed-Precision Sparsity-Aware Systolic Array Verification
#
#   make                                # PE test        (TOPLEVEL=pe_int8)
#   make TEST=array                     # baseline array GEMM test
#   make TEST=pe_mixed                  # precision-adaptive PE
#   make TEST=array_mixed                # precision-adaptive array
#   make TEST=decoder                   # 2:4 sparsity metadata decoder
#   make TEST=array_sparse               # full design (mixed + sparsity + ICG)
#   make TEST=array_sparse GATING=1      # ENABLE_GATING override for the PPA sweep
#   make TEST=pe_mixed_gl GL_NETLIST=... GL_LIB_V=...   # gate-level equivalence
#   make TEST=array TB_MATRICES=1000    # full random regression
#   make SIM=icarus TEST=pe             # override simulator (default: questa)
#                                        # (icarus | verilator | questa | vcs | xcelium)
#   make waves                          # run with waveform dump enabled
#   make clean
#
# Default SIM is questa, not icarus: this farm environment does not provision
# iverilog/cocotb-compatible Icarus by default, but Questa is reachable via
# `module load questa/...`. Override with SIM=icarus if you have it installed.
#-----------------------------------------------------------------------------

SIM           ?= questa
TOPLEVEL_LANG ?= verilog
TEST          ?= pe

TB_DIR := $(shell pwd)
RTL    := $(TB_DIR)/../src

VERILOG_SOURCES := $(RTL)/pe_int8.sv $(RTL)/systolic_array.sv \
                   $(RTL)/pe_mixed.sv $(RTL)/systolic_array_mixed.sv \
                   $(RTL)/icg_cell.sv $(RTL)/sparsity_decoder.sv \
                   $(RTL)/sparse_feeder.sv $(RTL)/systolic_array_sparse.sv

#-----------------------------------------------------------------------------
# Select DUT and Python test based on TEST. COCOTB_TOPLEVEL/COCOTB_TEST_MODULES
# are set alongside TOPLEVEL/MODULE because some cocotb versions on this farm
# read the COCOTB_-prefixed names instead of the bare ones; the reassignment
# below covers whichever your installed cocotb-config actually expects.
#-----------------------------------------------------------------------------

ifeq ($(TEST),pe)
  TOPLEVEL := pe_int8
  COCOTB_TOPLEVEL := pe_int8
  MODULE   := test_pe_int8
  COCOTB_TEST_MODULES := test_pe_int8
else ifeq ($(TEST),array)
  TOPLEVEL := systolic_array
  COCOTB_TOPLEVEL := systolic_array
  MODULE   := test_systolic_array
  COCOTB_TEST_MODULES := test_systolic_array
else ifeq ($(TEST),pe_mixed)
  TOPLEVEL := pe_mixed
  COCOTB_TOPLEVEL := pe_mixed
  MODULE   := test_pe_mixed
  COCOTB_TEST_MODULES := test_pe_mixed
else ifeq ($(TEST),array_mixed)
  TOPLEVEL := systolic_array_mixed
  COCOTB_TOPLEVEL := systolic_array_mixed
  MODULE   := test_array_mixed
  COCOTB_TEST_MODULES := test_array_mixed
else ifeq ($(TEST),decoder)
  TOPLEVEL := sparsity_decoder
  COCOTB_TOPLEVEL := sparsity_decoder
  MODULE   := test_sparsity_decoder
  COCOTB_TEST_MODULES := test_sparsity_decoder
else ifeq ($(TEST),array_sparse)
  TOPLEVEL := systolic_array_sparse
  COCOTB_TOPLEVEL := systolic_array_sparse
  MODULE   := test_array_sparse
  COCOTB_TEST_MODULES := test_array_sparse
else ifeq ($(TEST),pe_mixed_gl)
  # Gate-level: run the RTL testbench against the synthesised netlist. The
  # netlist keeps pe_mixed's port list, so a pass is also an equivalence check.
  TOPLEVEL := pe_mixed
  COCOTB_TOPLEVEL := pe_mixed
  MODULE   := test_pe_mixed
  COCOTB_TEST_MODULES := test_pe_mixed
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

# Backward compatibility with cocotb versions that expect the COCOTB_-prefixed
# variable names rather than the bare ones set above.
TOPLEVEL := $(COCOTB_TOPLEVEL)
MODULE   := $(COCOTB_TEST_MODULES)

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
  # Avoid deprecated automatic +acc behaviour (newer Questa warns/errors
  # without an explicit choice here).
  SIM_ARGS += -no_autoacc
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

# Allow importing project Python modules (e.g., golden_model.py) even if a
# test script's own sys.path.insert() isn't reached first.
export PYTHONPATH := $(TB_DIR):$(TB_DIR)/..:$(PYTHONPATH)

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

# Named .wlf + wave.do dump for Questa, defined after the include so it layers
# on top of (rather than replaces) the SIM_ARGS set above.
ifneq ($(filter $(SIM),questa modelsim),)
  SIM_ARGS += -wlf sim_build/vsim.wlf
  ifeq ($(WAVES),1)
    SIM_ARGS += -do $(TB_DIR)/wave.do
  endif
endif
