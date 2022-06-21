

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='7'
exp_name="debug-adversarial"



pgd_radius="0.2"
pgd_step="10"
pgd_step_size="0.04"
lambda_1="1e-2"
lambda_2="1e-2"
K="4"
topk_prox_metric="l1"

x_pgd_radius="0.2"
x_pgd_step="10"
x_pgd_step_size="0.04"
#for dataset in sst imdb hate emotion;do

for dataset in hate emotion;do
python train_and_run_experiments_bc.py --dataset ${dataset} --data_dir . --output_dir test_adv_outputs/ \
  --encoder lstm --adversarial --lmbda ${2} --gold_label_dir ${3} --n_iters 80 \
    --exp_name $exp_name --pgd_radius $pgd_radius
done;
