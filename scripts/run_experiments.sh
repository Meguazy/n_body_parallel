#!/bin/bash

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -d "scripts" ]; then
    echo "Error: This script must be run from the project root directory"
    exit 1
fi

# Check if executable exists
if [ ! -f "src/n_body_parallel_param" ]; then
    echo "Error: Executable not found. Please compile first:"
    echo "  cd src && make compile"
    exit 1
fi

# Create results directory
mkdir -p results

# Add these to your combinations array for stress testing:
# declare -a combinations=(
#     "100 1000"    # Communication bound
#     "200 800"     # Time step overhead  
#     "500 500"     # Balanced baseline
#     "1000 300"    # Medium computation
#     "2000 200"    # Computation bound
#     "5000 100"    # Highly parallel
#     "3000 400"    # High computation
#     "4000 500"    # Very high computation
#     "10000 200"   # EXTREME: 20B operations (~60-100 seconds)
#     "8000 400"    # EXTREME: 25.6B operations (~120-180 seconds)
#     "15000 100"   # EXTREME: 22.5B operations (~100-150 seconds)
#     "6000 600"    # EXTREME: 21.6B operations (~100-150 seconds)
#     "25000 100"   # EXTREME: 30B operations (~150-200 seconds)
#     "30000 100"   # EXTREME: 36B operations (~200-300 seconds)
# )

declare -a combinations=(
    "100 1000"    # Communication bound - High iterations, minimal computation
    "1000 300"    # Balanced medium - Good baseline for comparison  
    "5000 100"    # Computation bound - High computation, low communication overhead
    "3000 400"    # High both - Substantial computation + significant iterations
    "6000 600"    # Extreme iterations - Tests time step overhead at scale
    "15000 100"   # Extreme computation - Maximum bodies, tests memory/cache limits
    "25000 100"   # Extreme computation - Maximum bodies, tests memory/cache limits
)

# Initialize CSV file with header
echo "exec_id,num_bodies,num_steps,processors_number,elapsed_time,speed_up,efficiency" > results/performance_results.csv

echo "Starting experiments..."

# Loop through each combination
for combo in "${combinations[@]}"; do
    read -a params <<< "$combo"
    NUM_BODIES=${params[0]}
    NUM_STEPS=${params[1]}
    EXEC_ID="${NUM_BODIES}-${NUM_STEPS}"
    
    echo "Running experiments for configuration: $EXEC_ID"
    
    # Establish baseline
    echo "  Establishing baseline with 1 processor..."
    cd src
    mpirun --oversubscribe -np 1 ./n_body_parallel_param $NUM_BODIES $NUM_STEPS > /dev/null
    cd ..
    
    # Move baseline file
    if [ -f "src/baseline_${NUM_BODIES}_${NUM_STEPS}.txt" ]; then
        mv "src/baseline_${NUM_BODIES}_${NUM_STEPS}.txt" "results/"
    fi
    
    # Run experiments from 2 to 16 processors
    for np in {2..16}; do
        echo "  Running with $np processors..."
        cd src
        result=$(mpirun --oversubscribe -np $np ./n_body_parallel_param $NUM_BODIES $NUM_STEPS 2>/dev/null | tail -1)
        cd ..
        
        if [ ! -z "$result" ]; then
            IFS=',' read -ra ADDR <<< "$result"
            echo "$EXEC_ID,${ADDR[0]},${ADDR[1]},${ADDR[2]},${ADDR[3]},${ADDR[4]},${ADDR[5]}" >> results/performance_results.csv
        fi
    done
    
    echo "  Completed configuration $EXEC_ID"
done

echo "All experiments completed!"