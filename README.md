# N-Body Parallel Simulation

A high-performance parallel implementation of gravitational N-body simulation using MPI, with comprehensive performance analysis and visualization tools.

## Overview

This project simulates the gravitational interactions of N bodies in 3D space using Newton's law of universal gravitation. The simulation is parallelized using MPI (Message Passing Interface) and employs a master-slave architecture for efficient load distribution across multiple processors. The implementation is configurable for different problem sizes and includes automated performance benchmarking with detailed analysis and visualization capabilities. The numerical integration uses a 4th-order Runge-Kutta method for accurate trajectory calculations.

## Mathematical Foundation

### Gravitational Force

The simulation implements Newton's law of universal gravitation between every pair of bodies:

**F = G × m₁ × m₂ / (r² + ε)**

The gravitational constant G is set to 6.67430×10⁻¹¹ m³/(kg·s²), and ε represents a softening parameter (10⁻¹⁰) introduced to prevent singularities when bodies come very close together. For each body, the net acceleration is computed by summing the gravitational effects from all other bodies in the system.

### Numerical Integration

The system uses the **4th-order Runge-Kutta method** (RK4) for time integration. This method provides excellent accuracy while maintaining reasonable computational cost. At each time step, RK4 evaluates the system's derivative at four intermediate points, then combines these evaluations with specific weights (1:2:2:1) to advance both position and velocity. This approach has a local truncation error of O(Δt⁵) and a global error of O(Δt⁴), making it significantly more accurate than simpler methods like Euler integration.

### Computational Complexity

The algorithm has inherent O(N²) complexity for force calculations at each time step, since every body interacts with every other body. The total sequential complexity is O(N² × T), where N is the number of bodies and T is the number of time steps. Parallelization reduces this to approximately O(N²T/P + C), where P is the number of processors and C represents communication overhead from broadcasting and gathering body data.

## Parallel Architecture

The implementation uses a master-slave MPI architecture. The master process (rank 0) initializes the system with random body distributions and coordinates all operations. At each time step, the complete system state is broadcast to all processes. Work is then distributed using MPI_Scatterv, with load balancing that handles remainders when the number of bodies doesn't divide evenly. Each worker process computes forces and updates positions for its assigned bodies, and results are gathered back to the master using MPI_Gatherv. A custom MPI datatype is created for efficient Body struct communication, avoiding the overhead of packing and unpacking individual fields.

## Project Structure

The repository is organized with source code in the `src/` directory, automation scripts in `scripts/`, and experimental results generated in `results/`. The main C implementation is `n_body_parallel_param.c`, which accepts command-line parameters for the number of bodies and time steps. The `run_experiments.sh` script automates running multiple configurations with different processor counts, while `analyze_results.py` processes the raw timing data to generate performance metrics and visualizations.

## Getting Started

### Prerequisites

You need an MPI implementation (OpenMPI or MPICH), a GCC compiler, and Python 3 with matplotlib, pandas, and numpy for the analysis scripts. On Ubuntu or Debian systems, you can install OpenMPI with `sudo apt-get install libopenmpi-dev`. Python dependencies can be installed using `pip3 install matplotlib pandas numpy seaborn`.

### Quick Start

To run the complete workflow, make the main script executable and execute it:

```bash
chmod +x run_all.sh
./run_all.sh
```

This will compile the simulation, run all configured experiments (which may take 2-4 hours depending on your hardware), and generate comprehensive analysis with visualizations.

### Manual Execution

If you prefer step-by-step control, first compile the simulation:

```bash
cd src
make compile
cd ..
```

To run a single configuration manually:

```bash
cd src
mpirun -np 4 ./n_body_parallel_param 1000 100
cd ..
```

This executes the simulation with 4 processes, 1000 bodies, and 100 time steps. For a complete experimental run:

```bash
./scripts/run_experiments.sh
python3 scripts/analyze_results.py
```

## Experimental Configurations

The default experiment suite tests seven different problem configurations, ranging from small communication-bound scenarios (100 bodies, 1000 steps) to large computation-bound problems (25,000 bodies, 100 steps). Each configuration is executed with processor counts from 2 to 16, allowing analysis of scalability characteristics across different problem sizes. The baseline single-processor timing is established first, then used to calculate speedup and parallel efficiency for each multi-processor run.

## Performance Analysis

The analysis script generates several outputs. Raw experimental data is saved in CSV format with columns for configuration ID, number of bodies, time steps, processor count, elapsed time, speedup, and efficiency. A detailed performance report summarizes key metrics including best speedup, efficiency trends, and scalability characteristics for each configuration. Five visualization plots are generated: speedup comparison showing how each configuration scales with processors, efficiency comparison revealing parallel efficiency degradation, execution time trends on a logarithmic scale, an efficiency heatmap across all combinations, and individual configuration analyses with dual-axis plots of speedup and efficiency.

## Expected Performance

Performance characteristics depend strongly on problem size. Smaller problems tend to be communication-bound, showing modest speedup due to the overhead of broadcasting the full system state at each time step. Larger problems achieve better speedup as the computation-to-communication ratio improves. Typical results show near-linear speedup up to 8 processors, with efficiency around 0.95-1.0 for large problems. Beyond 8 processors on typical hardware, oversubscription effects appear as multiple MPI processes compete for physical cores, reducing efficiency to approximately 50%. The largest configurations (15,000-25,000 bodies) demonstrate the best scalability, sometimes even exhibiting super-linear speedup at low processor counts due to improved cache utilization.

## Implementation Notes

The simulation initializes bodies with random masses between 10²⁰ and 10³⁰ kg, positions within ±10¹³ meters, and velocities within ±5×10⁴ m/s. These values are chosen to represent astronomical-scale scenarios while maintaining numerical stability. The time step (DT) is set to 10⁴ seconds, appropriate for the spatial and velocity scales involved. The master process uses a fixed random seed (10) to ensure reproducible results across runs. When running experiments, baseline timing files are saved for each configuration, allowing subsequent multi-processor runs to calculate accurate speedup metrics.

## Troubleshooting

If you encounter compilation errors, verify that MPI is installed correctly by running `which mpicc`. For Python dependency issues, ensure all required packages are installed with `pip3 install matplotlib pandas numpy seaborn`. Long runtime is expected for large configurations; a full experimental suite with seven configurations and 15 processor counts can take several hours. You can monitor progress using `top` or `htop` to verify that MPI processes are actively running. For consistent performance measurements, consider using process binding with `mpirun --bind-to core` to reduce variability from process migration.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for complete terms and conditions.
