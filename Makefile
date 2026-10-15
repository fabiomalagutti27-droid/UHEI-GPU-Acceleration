# ==============================================================================
# Makefile for CUDA C++ UHEI Environmental Risk Acceleration
# Course: Sistemi di Elaborazione Accelerata - Università di Bologna
# Author: Fabio Malagutti (Student ID: 0001247400)
# ==============================================================================

# CUDA Compiler and Flags
NVCC        ?= nvcc
CXXFLAGS    ?= -O3 -std=c++17
ARCH        ?= -arch=sm_75

# Directories
SRC_DIR     := src
BIN_DIR     := bin
BUILD_DIR   := build

# Default block dimension (override via make BLOCK_DIM=16)
BLOCK_DIM   ?= 32
DEFINES     ?= -D BLOCK_DIM=$(BLOCK_DIM)

NVCC_FLAGS  := $(CXXFLAGS) $(ARCH) $(DEFINES)

# Primary Targets
TARGETS     := benchmark risk_map

.PHONY: all clean benchmark risk_map optimized_test tuning_extreme profiling_harness help

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 1. Benchmark Suite Target
benchmark: $(SRC_DIR)/benchmark_suite.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BIN_DIR)/$@

# 2. Risk Map Generator Target
risk_map: $(SRC_DIR)/risk_map_generator.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BIN_DIR)/$@

# 3. Supplemental Optimization Targets
optimized_test: $(SRC_DIR)/gpu_optimized_test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) --use_fast_math $< -o $(BIN_DIR)/$@

tuning_extreme: $(SRC_DIR)/kernel_tuning_extreme.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) --use_fast_math $< -o $(BIN_DIR)/$@

profiling_harness: $(SRC_DIR)/profiling_harness.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $(BIN_DIR)/$@

# Clean Target
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR) *.o *.obj *.out output_risk_map*.bin reports/*.ncu-rep

# Help Target
help:
	@echo "UHEI GPU Acceleration Build System"
	@echo "Usage: make [target] [ARCH=-arch=sm_XX] [BLOCK_DIM=XX]"
	@echo ""
	@echo "Available targets:"
	@echo "  all                Build benchmark and risk_map (default)"
	@echo "  benchmark          Compile benchmark suite ($(SRC_DIR)/benchmark_suite.cu)"
	@echo "  risk_map           Compile risk map generator ($(SRC_DIR)/risk_map_generator.cu)"
	@echo "  optimized_test     Compile pinned-memory optimization harness"
	@echo "  tuning_extreme     Compile instruction-level micro-benchmarks"
	@echo "  profiling_harness  Compile Nsight Compute profiling wrapper"
	@echo "  clean              Remove compiled binaries and build artifacts"
