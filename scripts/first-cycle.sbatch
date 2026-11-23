#!/bin/bash
#SBATCH --job-name=visforge-cycle
#SBATCH --account=jiahengwei_account
#SBATCH --partition=i64m1tga40u
#SBATCH --qos=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%j.out

set -eu
cd "$HOME/RSI/visforge"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export PYTHONUNBUFFERED=1
python="$HOME/miniconda3/envs/h3/bin/python"
echo "job=$SLURM_JOB_ID node=$(hostname) started=$(date -Is)"
"$python" -c 'import torch; print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); assert 1 <= torch.cuda.device_count() <= 2'
args=(--manifest data/raw/candidates.csv --out runs/pilot-same-four --model-path "$HOME/models/Qwen3.8-27B"
      --mix chart=2,document=1,photo=1 --reuse-development-for-comparison)
exec "$python" -m visforge.cli cycle "${args[@]}" "$@"
