#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define G 6.67430e-11 // Gravitational constant
#define DT 1e4 // Time step
#define NUM_BODIES 10 // Number of celestial bodies
#define NUM_STEPS 1000000 // Number of time steps

typedef struct {
    double mass;
    double position[3];
    double velocity[3];
} Body;

void compute_acceleration(Body *body1, Body *body2, double acceleration[3]) {
    double dx = body2->position[0] - body1->position[0];
    double dy = body2->position[1] - body1->position[1];
    double dz = body2->position[2] - body1->position[2];
    double distance = sqrt(dx * dx + dy * dy + dz * dz);
    double magnitude = (G * body2->mass) / (distance * distance + 1e-10);

    acceleration[0] = magnitude * dx / distance;
    acceleration[1] = magnitude * dy / distance;
    acceleration[2] = magnitude * dz / distance;
}

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
        body->velocity[i] += (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6;
        body->position[i] += (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) / 6;
    }
}

void update_bodies(Body bodies[NUM_BODIES]) {
    double net_acceleration[NUM_BODIES][3] = {0};

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

    for (int i = 0; i < NUM_BODIES; i++) {
        runge_kutta_step(&bodies[i], net_acceleration[i]);
    }
}

void plot_bodies(FILE *gnuplot, Body bodies[NUM_BODIES]) {
    fprintf(gnuplot, "plot '-' with points pt 7 ps 1 title 'Bodies'\n");
    for (int i = 0; i < NUM_BODIES; i++) {
        fprintf(gnuplot, "%.2e %.2e\n", bodies[i].position[0], bodies[i].position[1]);
    }
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);
}

int main() {
    Body bodies[NUM_BODIES] = {
        {1.989e30, {0, 0, 0}, {0, 0, 0}},            // Sun
        {3.301e23, {5.791e10, 0, 0}, {0, 47400, 0}}, // Mercury
        {4.867e24, {1.082e11, 0, 0}, {0, 35020, 0}}, // Venus
        {5.972e24, {1.496e11, 0, 0}, {0, 29780, 0}}, // Earth
        {6.417e23, {2.279e11, 0, 0}, {0, 24100, 0}}, // Mars
        {1.899e27, {7.785e11, 0, 0}, {0, 13070, 0}}, // Jupiter
        {5.685e26, {1.433e12, 0, 0}, {0, 9690, 0}},  // Saturn
        {8.682e25, {2.872e12, 0, 0}, {0, 6810, 0}},  // Uranus
        {1.024e26, {4.495e12, 0, 0}, {0, 5430, 0}},  // Neptune
        {1.461e22, {5.906e12, 0, 0}, {0, 4740, 0}}   // Pluto
    };

    struct timeval start, end;
    gettimeofday(&start, NULL);

    FILE *gnuplot = popen("gnuplot -persistent", "w");
    if (!gnuplot) {
        fprintf(stderr, "Could not open gnuplot\n");
        return 1;
    }

    fprintf(gnuplot, "set xrange [-6e12:6e12]\n");
    fprintf(gnuplot, "set yrange [-6e12:6e12]\n");
    fprintf(gnuplot, "set size square\n");

    for (int step = 0; step < NUM_STEPS; step++) {
        update_bodies(bodies);
        if (step % 50 == 0) {
            plot_bodies(gnuplot, bodies);
            
        }
    }

    pclose(gnuplot);

    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) + 
                        (end.tv_usec - start.tv_usec) / 1e6;

    printf("Simulation completed in %.6f seconds.\n", time_taken);

    return 0;
}
