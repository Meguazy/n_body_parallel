# Makefile for N-Body Simulation (src/ directory)

CC = mpicc
CFLAGS = -O3 -Wall -lm
PYTHON = python3

# Source and executable names
SRC = n_body_parallel_param.c
EXEC = n_body_parallel_param

# Directories
SCRIPTS_DIR = ../scripts
RESULTS_DIR = ../results

# Default target
all: compile

# Compile the program
compile:
	@echo "Compiling N-Body simulation..."
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC)
	@echo "Compilation successful!"

# Run all experiments (calls script from scripts directory)
run: compile
	@echo "Starting experimental runs..."
	@mkdir -p $(RESULTS_DIR)
	@chmod +x $(SCRIPTS_DIR)/run_experiments.sh
	cd .. && ./scripts/run_experiments.sh

# Analyze results (calls Python script)
analyze:
	@echo "Analyzing results..."
	@mkdir -p $(RESULTS_DIR)
	cd .. && $(PYTHON) scripts/analyze_results.py

# Clean generated files
clean:
	@echo "Cleaning up..."
	rm -f $(EXEC)
	rm -rf $(RESULTS_DIR)
	@echo "Cleanup complete!"

# Quick test with small parameters
test: compile
	@echo "Running quick test..."
	@mkdir -p $(RESULTS_DIR)
	mpirun -np 1 ./$(EXEC) 100 10
	mpirun -np 4 ./$(EXEC) 100 10

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Compile the program (default)"
	@echo "  compile  - Compile the N-Body simulation"
	@echo "  run      - Run all experiments"
	@echo "  analyze  - Analyze results and generate plots"
	@echo "  test     - Quick test run"
	@echo "  clean    - Remove generated files"
	@echo "  help     - Show this help"
	@echo ""
	@echo "For full workflow, use: make compile && make run && make analyze"

.PHONY: all compile run analyze clean test help