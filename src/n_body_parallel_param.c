#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>

#define G 6.67430e-11

// Global variables that can be set via command line
int NUM_BODIES = 1000;  // Default value
int NUM_STEPS = 100;    // Default value
double DT = 1e4;        // Time step

// Define the Body struct
typedef struct {
    double mass;
    double position[3];
    double velocity[3];
} Body;

// Create MPI datatype for Body struct
MPI_Datatype create_body_datatype() {
    MPI_Datatype body_type;
    int blocklengths[] = {1, 3, 3};
    MPI_Aint offsets[3];
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    
    offsets[0] = offsetof(Body, mass);
    offsets[1] = offsetof(Body, position);
    offsets[2] = offsetof(Body, velocity);
    
    MPI_Type_create_struct(3, blocklengths, offsets, types, &body_type);
    MPI_Type_commit(&body_type);
    
    return body_type;
}

// Compute the acceleration between two bodies
void compute_acceleration(Body *body1, Body *body2, double acceleration[3]) {
    double dx = body2->position[0] - body1->position[0];
    double dy = body2->position[1] - body1->position[1];
    double dz = body2->position[2] - body1->position[2];
    
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    double dist_cubed = pow(distance, 3) + 1e-10;
    
    double magnitude = G * body2->mass / dist_cubed;
    
    acceleration[0] = magnitude * dx;
    acceleration[1] = magnitude * dy;
    acceleration[2] = magnitude * dz;
}

// Perform a single step of the Runge-Kutta integration
void runge_kutta_step(Body *body, double net_acceleration[3]) {
    double k1v[3], k1x[3];
    double k2v[3], k2x[3];
    double k3v[3], k3x[3];
    double k4v[3], k4x[3];
    
    for (int i = 0; i < 3; i++) {
        k1v[i] = net_acceleration[i] * DT;
        k1x[i] = body->velocity[i] * DT;
        k2v[i] = net_acceleration[i] * DT;
        k2x[i] = (body->velocity[i] + 0.5 * k1v[i]) * DT;
        k3v[i] = net_acceleration[i] * DT;
        k3x[i] = (body->velocity[i] + 0.5 * k2v[i]) * DT;
        k4v[i] = net_acceleration[i] * DT;
        k4x[i] = (body->velocity[i] + k3v[i]) * DT;
    }
    
    for (int i = 0; i < 3; i++) {
        body->velocity[i] += (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) / 6;
        body->position[i] += (k1x[i] + 2*k2x[i] + 2*k3x[i] + k4x[i]) / 6;
    }
}

// Update the positions of all bodies in the simulation
void update_body_positions(Body *local_bodies, int local_count, Body *all_bodies, int rank) {
    for (int i = 0; i < local_count; i++) {
        double net_acceleration[3] = {0};
        for (int j = 0; j < NUM_BODIES; j++) {
            if (&local_bodies[i] != &all_bodies[j]) {
                double acceleration[3];
                compute_acceleration(&local_bodies[i], &all_bodies[j], acceleration);
                for (int k = 0; k < 3; k++) {
                    net_acceleration[k] += acceleration[k];
                }
            }
        }
        runge_kutta_step(&local_bodies[i], net_acceleration);
    }
}

// Generate random double between min and max
double random_double(double min, double max) {
    return min + (rand() / (double)RAND_MAX) * (max - min);
}

// Initialize bodies randomly
void initialize_random_bodies(Body *bodies, int num_bodies, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < num_bodies; i++) {
        bodies[i].mass = random_double(1e20, 1e30);
        bodies[i].position[0] = random_double(-1e13, 1e13);
        bodies[i].position[1] = random_double(-1e13, 1e13);
        bodies[i].position[2] = random_double(-1e13, 1e13);
        bodies[i].velocity[0] = random_double(-5e4, 5e4);
        bodies[i].velocity[1] = random_double(-5e4, 5e4);
        bodies[i].velocity[2] = random_double(-5e4, 5e4);
    }
}

int main(int argc, char **argv) {
    // Parse command line arguments
    if (argc >= 3) {
        NUM_BODIES = atoi(argv[1]);
        NUM_STEPS = atoi(argv[2]);
    } else {
        printf("Usage: %s <NUM_BODIES> <NUM_STEPS>\n", argv[0]);
        printf("Using default values: NUM_BODIES=%d, NUM_STEPS=%d\n", NUM_BODIES, NUM_STEPS);
    }

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create MPI datatype for Body struct
    MPI_Datatype body_type = create_body_datatype();
    int *sendcounts = (int*) malloc(size * sizeof(int));
    int *displs = (int*) malloc(size * sizeof(int));
    Body *all_bodies = malloc(NUM_BODIES * sizeof(Body));
    int local_count;
    
    // Create filename for baseline - save in results directory when run from scripts
    char baseline_filename[100];
    snprintf(baseline_filename, sizeof(baseline_filename), "../results/baseline_%d_%d.txt", NUM_BODIES, NUM_STEPS);
    
    if (rank == 0) {   
        // Initialize the system with random bodies
        initialize_random_bodies(all_bodies, NUM_BODIES, 10);
        
        // Calculate distribution
        int base_count = NUM_BODIES / size;
        int remainder = NUM_BODIES % size;
        
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base_count + (i < remainder ? 1 : 0);
            if (i > 0) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
        }
    }
    
    // Scatter sendcounts to all processes
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate local array
    Body *local_bodies = malloc(local_count * sizeof(Body));

    double start = MPI_Wtime();
    
    // Main simulation loop
    for (int step = 0; step < NUM_STEPS; step++) {
        MPI_Bcast(all_bodies, NUM_BODIES, body_type, 0, MPI_COMM_WORLD);
        MPI_Scatterv(all_bodies, sendcounts, displs, body_type,
                     local_bodies, local_count, body_type,
                     0, MPI_COMM_WORLD);
        
        update_body_positions(local_bodies, local_count, all_bodies, rank);

        MPI_Gatherv(local_bodies, local_count, body_type,
                    all_bodies, sendcounts, displs, body_type,
                    0, MPI_COMM_WORLD);
    }
    
    double end = MPI_Wtime();
    
    if (rank == 0) {
        double elapsed = end - start;
        
        if (size == 1) {
            // Save baseline for this configuration
            FILE *file = fopen(baseline_filename, "w");
            if (file) {
                fprintf(file, "%.6f\n", elapsed);
                fclose(file);
            }
            printf("%d,%d,%d,%.6f,0.000000,0.000000\n", NUM_BODIES, NUM_STEPS, size, elapsed);
        } else {
            // Calculate speedup and efficiency
            FILE *file = fopen(baseline_filename, "r");
            double baseline = elapsed; // Default to current time if no baseline
            double speedup = 1.0;
            double efficiency = 1.0 / size;
            
            if (file) {
                fscanf(file, "%lf", &baseline);
                fclose(file);
                speedup = baseline / elapsed;
                efficiency = speedup / size;
            }
            
            // Output in CSV format: num_bodies,num_steps,processors,elapsed_time,speedup,efficiency
            printf("%d,%d,%d,%.6f,%.6f,%.6f\n", NUM_BODIES, NUM_STEPS, size, elapsed, speedup, efficiency);
        }
    }
    
    // Cleanup
    free(all_bodies);
    free(local_bodies);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
    
    MPI_Type_free(&body_type);
    MPI_Finalize();
    return 0;
}
