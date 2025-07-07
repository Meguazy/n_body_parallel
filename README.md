# N-Body Parallel Simulation

A parallel implementation of gravitational N-body simulation using MPI with comprehensive performance analysis.

## 📁 Directory Structure

```
n_body_parallel/
├── src/
│   ├── n_body_parallel_param.c    # Main simulation code
│   └── Makefile                   # Compilation rules
├── scripts/
│   ├── run_experiments.sh         # Experiment runner
│   └── analyze_results.py         # Performance analysis
├── results/                       # Generated results (created automatically)
│   ├── performance_results.csv    # Raw experimental data
│   ├── performance_report.txt     # Analysis summary
│   ├── baseline_*.txt            # Baseline timing files
│   └── plots/                    # Visualization files
├── run_all.sh                    # Main execution script
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- MPI implementation (OpenMPI, MPICH, etc.)
- GCC compiler
- Python 3 with matplotlib, pandas, numpy (for analysis)

### Run Complete Workflow
```bash
# Make executable and run everything
chmod +x run_all.sh
./run_all.sh
```

### Step-by-Step Execution
```bash
# 1. Compile the simulation
cd src
make compile
cd ..

# 2. Run experiments (2-4 hours)
./scripts/run_experiments.sh

# 3. Analyze results
python3 scripts/analyze_results.py
```

## 🧪 Experimental Configurations

The framework tests 5 different problem sizes:

| Config ID | Bodies | Steps | Operations | Problem Size |
|-----------|--------|-------|------------|--------------|
| 500-50    | 500    | 50    | 12.5M      | Small        |
| 1000-100  | 1000   | 100   | 100M       | Medium-small |
| 2000-150  | 2000   | 150   | 600M       | Medium       |
| 5000-200  | 5000   | 200   | 5B         | Large        |
| 8000-250  | 8000   | 250   | 16B        | Very large   |

Each configuration is tested with 2-16 MPI processes.

## 📊 Results and Analysis

### Generated Files

1. **performance_results.csv** - Raw experimental data with columns:
   - `exec_id`: Configuration identifier (e.g., "1000-100")
   - `num_bodies`: Number of bodies in simulation
   - `num_steps`: Number of time steps
   - `processors_number`: Number of MPI processes used
   - `elapsed_time`: Wall-clock time in seconds
   - `speed_up`: Speedup compared to single-process baseline
   - `efficiency`: Parallel efficiency (speedup/processors)

2. **performance_report.txt** - Comprehensive analysis including:
   - Best speedup and efficiency across all configurations
   - Scalability analysis for each problem size
   - Performance highlights and bottlenecks

3. **Visualization Files** in `results/plots/`:
   - `speedup_comparison.png` - Speedup vs processors for all configs
   - `efficiency_comparison.png` - Efficiency analysis
   - `execution_time_comparison.png` - Runtime comparison
   - `efficiency_heatmap.png` - Efficiency across all combinations
   - `individual_configurations.png` - Detailed per-config analysis

## 🔧 Manual Usage

### Compile and Test
```bash
cd src
make compile
make test  # Quick test with small parameters
```

### Run Single Configuration
```bash
cd src
# Run with 4 processes, 1000 bodies, 100 steps
mpirun -np 4 ./n_body_parallel_param 1000 100
```

### Custom Experiments
```bash
# Modify the combinations array in scripts/run_experiments.sh
declare -a combinations=(
    "100 20"     # Your custom configuration
    "200 30"     # Another configuration
)
```

## 🏗️ Implementation Details

### Algorithm
- **Physics**: Gravitational N-body simulation using Newton's law
- **Numerical Integration**: 4th-order Runge-Kutta method
- **Parallelization**: Master-slave architecture with MPI
- **Communication**: Broadcast-scatter-gather pattern

### Key Features
- **Command-line parameters**: Configurable NUM_BODIES and NUM_STEPS
- **Load balancing**: Optimal work distribution with remainder handling
- **Custom MPI datatype**: Efficient communication of Body structures
- **Performance measurement**: Automatic speedup and efficiency calculation

### Computational Complexity
- **Sequential**: O(N² × T) where N = bodies, T = time steps
- **Parallel**: O(N²T/P + communication overhead)
- **Memory**: O(N × P) due to global data dependency

## 📈 Expected Performance Characteristics

Based on the N-body algorithm's properties:

- **Good scaling**: 2-8 processes typically show near-linear speedup
- **Communication bound**: Efficiency degrades beyond 10-12 processes
- **Super-linear effects**: Possible due to cache improvements at low process counts
- **Problem size dependency**: Larger problems scale better due to computation/communication ratio

## 🛠️ Troubleshooting

### Common Issues

1. **Compilation Errors**
   ```bash
   # Ensure MPI is installed
   which mpicc
   # Install if missing (Ubuntu/Debian)
   sudo apt-get install libopenmpi-dev
   ```

2. **Python Dependencies**
   ```bash
   # Install required packages
   pip3 install matplotlib pandas numpy seaborn
   ```

3. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x run_all.sh scripts/run_experiments.sh
   ```

4. **Long Runtime**
   - Large configurations (5000+ bodies) can take hours
   - Consider reducing problem sizes for quick testing
   - Monitor with `top` or `htop` to ensure processes are running

### Performance Tips

- **CPU affinity**: Use process binding for consistent results
  ```bash
  mpirun --bind-to core -np 4 ./n_body_parallel_param 1000 100
  ```

- **Node allocation**: For clusters, ensure processes are on same node for small tests
  ```bash
  mpirun --map-by node -np 4 ./n_body_parallel_param 1000 100
  ```

## 📚 Technical Background

### Physics
The simulation implements Newton's law of universal gravitation:
```
F = G × m₁ × m₂ / (r² + ε)
```
Where ε is a softening parameter to prevent singularities.

### Parallel Algorithm
1. **Initialization**: Master generates random body distribution
2. **Broadcast**: All processes receive complete system state
3. **Scatter**: Work distributed among processes with load balancing
4. **Compute**: Each process calculates forces for assigned bodies
5. **Gather**: Updated bodies collected by master
6. **Repeat**: Steps 2-5 for each time step

### Architecture
- **Master Process (rank 0)**: Coordination, I/O, performance measurement
- **Worker Processes**: Force calculation and numerical integration
- **Communication**: Global broadcast required due to O(N²) force interactions

## 🤝 Contributing

To extend or modify the simulation:

1. **Add new configurations**: Edit `combinations` array in `run_experiments.sh`
2. **Modify physics**: Update force calculations in `compute_acceleration()`
3. **Change integration**: Modify `runge_kutta_step()` function
4. **Extend analysis**: Add visualizations in `analyze_results.py`

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🏆 Performance Benchmarks

Expected results on typical hardware:

| Configuration | 4 Processes | 8 Processes | 16 Processes |
|---------------|-------------|-------------|--------------|
| 500-50        | ~3.2x       | ~5.8x       | ~8.1x        |
| 1000-100      | ~3.5x       | ~6.2x       | ~9.3x        |
| 2000-150      | ~3.7x       | ~6.8x       | ~10.1x       |
| 5000-200      | ~3.8x       | ~7.1x       | ~11.2x       |
| 8000-250      | ~3.9x       | ~7.3x       | ~12.1x       |

*Actual results depend on hardware, network, and system load.*