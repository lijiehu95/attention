export CUDA_VISIBLE_DEVICES=2
source activate xai
export PYTHONPATH=/home/yila22/prj

wandb sweep sweep_imdb.yaml
wandb agent

