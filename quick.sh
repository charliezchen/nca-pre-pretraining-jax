#!/usr/bin/env bash
set -euo pipefail

rm -rf checkpoints/nca
./scripts/train_nca.sh
./scripts/train_lm.sh --nca_ckpt checkpoints/nca/final --wandb_name nca_lm
