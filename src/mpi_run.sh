#!/bin/bash

for np in {1..8}
do
    mpirun -np $np ./n_body_parallel
done
