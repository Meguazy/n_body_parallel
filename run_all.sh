#!/bin/bash

# Main execution script for N-Body Simulation Experiments
# Location: project_root/run_all.sh

echo "=============================================="
echo "N-Body Simulation Complete Workflow"
echo "=============================================="
echo ""

# Check directory structure
echo "Checking directory structure..."
if [ ! -d "src" ] || [ ! -d "scripts" ]; then
    echo "Error: Missing required directories!"
    echo "Expected structure:"
    echo "  src/ (containing C source code)"
    echo "  scripts/ (containing shell and Python scripts)"
    echo "  results/ (will be created)"
    exit 1
fi

# Create results directory
mkdir -p results

echo "Directory structure OK"
echo ""

# Step 1: Compile
echo "Step 1: Compiling N-Body simulation..."
cd src
make compile
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
cd ..
echo "Compilation successful"
echo ""

# Step 2: Run experiments
echo "Step 2: Running experiments..."
echo ""
echo "Each configuration will be tested with 2-16 processors."
echo ""

read -p "Continue with experiments? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiments cancelled."
    exit 0
fi

echo "Starting experiments..."
./scripts/run_experiments.sh
if [ $? -ne 0 ]; then
    echo "Experiments failed!"
    exit 1
fi
echo "Experiments completed"
echo ""

# Step 3: Analyze results
echo "Step 3: Analyzing results..."
python3 scripts/analyze_results.py
if [ $? -ne 0 ]; then
    echo "Analysis failed!"
    exit 1
fi
echo "Analysis completed"
echo ""

# Summary
echo "=============================================="
echo "Workflow completed successfully!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  results/performance_results.csv"
echo "  results/performance_report.txt"
echo "  results/plots/ (5 visualization files)"
echo ""
echo "Quick preview of results:"
if [ -f "results/performance_results.csv" ]; then
    echo "Total experimental runs: $(tail -n +2 results/performance_results.csv | wc -l)"
    echo ""
    echo "Sample data:"
    head -5 results/performance_results.csv
fi
echo ""
echo "To view detailed analysis:"
echo "  cat results/performance_report.txt"
echo "  ls results/plots/"
