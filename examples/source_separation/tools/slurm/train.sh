#!/bin/bash

num_speakers="$1"

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="${this_dir}/../../"
save_dir="/checkpoint/${USER}/jobs/source_separation/${SLURM_JOB_ID}"
dataset_dir="${HOME}/dataset/wsj0-mix/${num_speakers}speakers/wav8k/min"


if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    protocol="file:///checkpoint/${USER}/jobs/source_separation/${SLURM_JOB_ID}/sync"
else
    protocol="env://"
fi

mkdir -p "${save_dir}"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate "${root_dir}/env"
if [ "${SLURM_PROCID}" = "0" ]; then
    nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv 1>&2
    pip freeze | grep torch 1>&2
fi

python -u \
  "${root_dir}/train.py" \
  --worker-id "${SLURM_PROCID}" \
  --num-workers "${SLURM_NTASKS}" \
  --device-id "${SLURM_LOCALID}" \
  --sync-protocol "${protocol}" \
  -- \
  --num-speakers "${num_speakers}" \
  --sample-rate 8000 \
  --batch-size $((16 / SLURM_NTASKS)) \
  --dataset-dir "${dataset_dir}" \
  --save-dir "${save_dir}" \
  --epochs 200
