

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='7'
exp_name="debug-adversarial"


for dataset in sst imdb hate emotion;do
python train_and_run_experiments_bc.py --dataset ${dataset} --data_dir . --output_dir test_adv_outputs/ \
  --encoder lstm --adversarial --lmbda ${2} --gold_label_dir ${3} --n_iters 80 \
    --exp_name $exp_name
done;
