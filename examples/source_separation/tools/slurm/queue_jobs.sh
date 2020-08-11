#!/usr/bin/env bash

for num_speakers in 3 2; do
    sbatch tools/slurm/launch_job.sh "${num_speakers}"
done
