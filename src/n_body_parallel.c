#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>  // For usleep
#include <time.h>

#define G 6.67430e-11
#define DT 1e4
#define NUM_BODIES 1000
#define NUM_STEPS 300

// Define the Body struct
// This struct represents a body with mass, position, and velocity
typedef struct {
    double mass;
    double position[3];
    double velocity[3];
} Body;

// Create MPI datatype for Body struct
MPI_Datatype create_body_datatype() {
    // Create MPI datatype for Body struct
    MPI_Datatype body_type;
    // Define the blocklengths, types, and offsets for the struct members
    int blocklengths[] = {1, 3, 3};
    MPI_Aint offsets[3];
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    
    // The three offsets are the addresses of the mass, position, and velocity members
    offsets[0] = offsetof(Body, mass);
    offsets[1] = offsetof(Body, position);
    offsets[2] = offsetof(Body, velocity);
    
    // Create the struct type
    MPI_Type_create_struct(3, blocklengths, offsets, types, &body_type);
    MPI_Type_commit(&body_type);
    
    return body_type;
}

// Compute the acceleration between two bodies
void compute_acceleration(Body *body1, Body *body2, double acceleration[3]) {
    // Compute the distance between the two bodies across each dimension
    double dx = body2->position[0] - body1->position[0];
    double dy = body2->position[1] - body1->position[1];
    double dz = body2->position[2] - body1->position[2];
    
    // Compute the distance between the two bodies using the euclidean distance formula
    double distance = sqrt(dx*dx + dy*dy + dz*dz);

    // Compute the magnitude of the gravitational force between the two bodies
    // The formula is G * m2 / (r^2 + ε), where G is the gravitational constant,
    double magnitude = (G * body2->mass) / (distance * distance + 1e-10);

    // Compute the acceleration components using the formula a = (magnitude * d[x, y, z]) / (r + ε)
    acceleration[0] = magnitude * dx / (distance + 1e-10);
    acceleration[1] = magnitude * dy / (distance + 1e-10);
    acceleration[2] = magnitude * dz / (distance + 1e-10);
}

// Perform a single step of the Runge-Kutta integration
void runge_kutta_step(Body *body, double net_acceleration[3]) {
    // Compute the four Runge-Kutta steps for velocity and position
    double k1v[3], k1x[3];
    double k2v[3], k2x[3];
    double k3v[3], k3x[3];
    double k4v[3], k4x[3];
    
    // Compute the four Runge-Kutta steps for velocity and position
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
    
    // Update the velocity and position using the Runge-Kutta steps
    for (int i = 0; i < 3; i++) {
        body->velocity[i] += (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) / 6;
        body->position[i] += (k1x[i] + 2*k2x[i] + 2*k3x[i] + k4x[i]) / 6;
    }
}

// Update the positions of all bodies in the simulation
void update_body_positions(Body *local_bodies, int local_count, Body *all_bodies, int rank) {
    // Compute the net acceleration on each body
    // We use the local bodies to update the positions with all other bodies
    for (int i = 0; i < local_count; i++) {
        double net_acceleration[3] = {0};
        for (int j = 0; j < NUM_BODIES; j++) {
            double acceleration[3];
            compute_acceleration(&local_bodies[i], &all_bodies[j], acceleration);
            for (int k = 0; k < 3; k++) {
                net_acceleration[k] += acceleration[k];
            }
        }
        // Update the position of the body using the Runge-Kutta integration
        runge_kutta_step(&local_bodies[i], net_acceleration);
    }
}

// Define a function to generate a random double between min and max
double random_double(double min, double max) {
    return min + (rand() / (double)RAND_MAX) * (max - min);
}

// Function to initialize a given number of bodies randomly
void initialize_random_bodies(Body *bodies, int num_bodies, unsigned int seed) {
    // Seed the random number generator with the provided seed
    srand(seed);

    for (int i = 0; i < num_bodies; i++) {
        // Generate a random mass between 1e20 and 1e30 kg
        bodies[i].mass = random_double(1e20, 1e30);

        // Generate random positions within a cubic region of size 1e13 meters
        bodies[i].position[0] = random_double(-1e13, 1e13);
        bodies[i].position[1] = random_double(-1e13, 1e13);
        bodies[i].position[2] = random_double(-1e13, 1e13);

        // Generate random velocities within a range of ±5e4 meters per second
        bodies[i].velocity[0] = random_double(-5e4, 5e4);
        bodies[i].velocity[1] = random_double(-5e4, 5e4);
        bodies[i].velocity[2] = random_double(-5e4, 5e4);
    }
}

int main(int argc, char **argv) {
    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create MPI datatype for Body struct
    MPI_Datatype body_type = create_body_datatype();
    // Create arrays to store sendcounts and displacements for scatterv and gatherv
    int *sendcounts = (int*) malloc(size * sizeof(int));
    int *displs = (int*) malloc(size * sizeof(int));
    // Initializes bodies at random
    Body *all_bodies = malloc(NUM_BODIES * sizeof(Body));  // Allocate for all processes
    int local_count;
    
    if (rank == 0) {   
        // Initialize the system with random bodies
        Body initial_bodies[NUM_BODIES];
        initialize_random_bodies(initial_bodies, NUM_BODIES, 10);

        // Copy the initial bodies to the all_bodies array
        memcpy(all_bodies, initial_bodies, NUM_BODIES * sizeof(Body));
        
        // Calculate distribution
        int base_count = NUM_BODIES / size;
        int remainder = NUM_BODIES % size;
        
        // The distribution is calculated as follows:
        // - Each process will handle base_count bodies
        // - The first remainder processes will handle one additional body
        // - The displacements are calculated based on the sendcounts
        // - The displacements are the starting index of the bodies for each process
        // - The sendcounts are the number of bodies each process will handle
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base_count + (i < remainder ? 1 : 0);
            if (i > 0) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
            printf("Process %d will handle %d bodies starting at index %d\n", 
                   i, sendcounts[i], displs[i]);
        }
    }
    
    // Broadcast sendcounts to all processes
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate local array used by each process to store bodies
    Body *local_bodies = malloc(local_count * sizeof(Body));

    double start = MPI_Wtime();
    
    // Main simulation loop
    for (int step = 0; step < NUM_STEPS; step++) {
        // Broadcast the updated positions to all processes for the next iteration
        MPI_Bcast(all_bodies, NUM_BODIES, body_type, 0, MPI_COMM_WORLD);

        // Scatter bodies to processes
        // Each process will receive a subset of the bodies based on the sendcounts and displacements
        MPI_Scatterv(all_bodies, sendcounts, displs, body_type,
                     local_bodies, local_count, body_type,
                     0, MPI_COMM_WORLD);
        
        // Update positions using runge kutta integration to approximate new positions
        update_body_positions(local_bodies, local_count, all_bodies, rank);

        // Gather updated bodies from all processes inside of all_bodies
        MPI_Gatherv(local_bodies, local_count, body_type,
                    all_bodies, sendcounts, displs, body_type,
                    0, MPI_COMM_WORLD);
    }
    
    double end = MPI_Wtime();
    
    if (rank == 0) {
        double elapsed = end - start;
        printf("---------------------N° of processors used: %d---------------------\n", size);
        printf("Elapsed time: %.6f seconds\n", elapsed);
        
        if (size == 1) {
            FILE *file = fopen("baseline_time.txt", "w");
            if (file) {
                fprintf(file, "%.6f\n", elapsed);
                fclose(file);
                printf("Baseline time saved.\n");
            }
        } else {
            FILE *file = fopen("baseline_time.txt", "r");
            if (file) {
                double baseline;
                fscanf(file, "%lf", &baseline);
                fclose(file);
                
                double speedup = baseline / elapsed;
                double efficiency = speedup / size;
                printf("Speedup: %.6f\n", speedup);
                printf("Efficiency: %.6f\n", efficiency);
            }
        }
        printf("-------------------------------------------------------------------\n");
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