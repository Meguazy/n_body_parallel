#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define G 6.67430e-11 // Gravitational constant
#define DT 1e4 // Time step
#define NUM_BODIES 28 // Number of celestial bodies
#define NUM_STEPS 1000000 // Number of time steps

// Structure to store the position and velocity of a celestial body
typedef struct 
{
    double mass;
    double position[3];
    double velocity[3];
} Body;

// Function to compute the gravitational acceleration on a body due to another body
void compute_acceleration(Body *body1, Body *body2, double acceleration[3]) {
    // Computing the x differential webetwwen the two different bodies
    double dx = body2->position[0] - body1->position[0];
    // Computing the y differential webetwwen the two different bodies
    double dy = body2->position[1] - body1->position[1];
    // Computing the z differential webetwwen the two different bodies
    double dz = body2->position[2] - body1->position[2];
    // Computing the distance between the two different bodies
    double distance = sqrt(dx * dx + dy * dy + dz * dz);
    // Computing the magnitude of the gravitational force between the two different bodies
    double magnitude = (G * body2->mass) / (distance * distance + 1e-10);

    acceleration[0] = magnitude * dx / distance;
    acceleration[1] = magnitude * dy / distance;
    acceleration[2] = magnitude * dz / distance;
}

// Runge-Kutta 4th order integration for position and velocity
void runge_kutta_step(Body *body, double net_acceleration[3]) {
    // Declaring the variables for the Runge-Kutta 4th order integration
    double k1v[3], k1x[3];
    double k2v[3], k2x[3];
    double k3v[3], k3x[3];
    double k4v[3], k4x[3];

    // Computing the Runge-Kutta 4th order integration
    // Computing the k1v and k1x (1st order)
    for (int i = 0; i < 3; i++) {
        k1v[i] = net_acceleration[i] * DT;
        k1x[i] = body->velocity[i] * DT;
    }

    // Computing the k2v and k2x (2nd order)
    for (int i = 0; i < 3; i++) {
        k2v[i] = net_acceleration[i] * DT;
        k2x[i] = (body->velocity[i] + 0.5 * k1v[i]) * DT;
    }

    // Computing the k3v and k3x (3rd order)
    for (int i = 0; i < 3; i++) {
        k3v[i] = net_acceleration[i] * DT;
        k3x[i] = (body->velocity[i] + 0.5 * k2v[i]) * DT;
    }

    // Computing the k4v and k4x (4th order)
    for (int i = 0; i < 3; i++) {
        k4v[i] = net_acceleration[i] * DT;
        k4x[i] = (body->velocity[i] + k3v[i]) * DT;
    }

    // Updating the position and velocity of the body
    for (int i = 0; i < 3; i++) {
        body->velocity[i] += (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6;
        body->position[i] += (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) / 6;
    }
}

// Serial implementation of the n-body simulation
void update_bodies_serial(Body bodies[NUM_BODIES]) {
    // Declaring the net acceleration for each body by initializing it to 0
    double net_acceleration[NUM_BODIES][3] = {0};

    // Compute net acceleration for each body
    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = 0; j < NUM_BODIES; j++) {
            if (i != j) {
                double acceleration[3];
                compute_acceleration(&bodies[i], &bodies[j], acceleration);
                for (int k = 0; k < 3; k++) {
                    net_acceleration[i][k] += acceleration[k];
                }
            }
        }
    }

    // Apply Runge-Kutta integration for each body
    for (int i = 0; i < NUM_BODIES; i++) {
        runge_kutta_step(&bodies[i], net_acceleration[i]);
    }
}

// Print bodies
void print_bodies(Body bodies[NUM_BODIES], int step) {
    printf("Step %d:\n", step);
    for (int i = 0; i < NUM_BODIES; i++) {
        printf("Body %d: Position = (%.2e, %.2e, %.2e), Velocity = (%.2e, %.2e, %.2e)\n",
               i,
               bodies[i].position[0], bodies[i].position[1], bodies[i].position[2],
               bodies[i].velocity[0], bodies[i].velocity[1], bodies[i].velocity[2]);
    }
    printf("\n");
}

int main() {
    // Initialize bodies (example: Sun, Earth, Moon)
    Body bodies[NUM_BODIES] = {
        {1.989e30, {0, 0, 0}, {0, 0, 0}},           // Sun
        {5.972e24, {1.496e11, 0, 0}, {0, 29780, 0}}, // Earth
        {7.348e22, {1.496e11 + 3.844e8, 0, 0}, {0, 29780 + 1022, 0}} // Moon
    };

    // Measure start time
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Run simulation
    for (int step = 0; step < NUM_STEPS; step++) {
        update_bodies_serial(bodies);
        //print_bodies(bodies, step);
    }

    // Measure end time
    gettimeofday(&end, NULL);

    // Calculate elapsed time in seconds
    double time_taken = (end.tv_sec - start.tv_sec) + 
                        (end.tv_usec - start.tv_usec) / 1e6;

    printf("Simulation completed in %.6f seconds.\n", time_taken);

    return 0;
}
