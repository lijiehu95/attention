

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='7'
exp_name="debug-ours"



#pgd_radius="0.2"
#pgd_step="10"
#pgd_step_size="0.04"
#lambda_1="1e-2"
#lambda_2="1e-2"
#K="4"
#topk_prox_metric="l1"

#x_pgd_radius="0.2"
#x_pgd_step="10"
#x_pgd_step_size="0.04"
#for dataset in sst imdb hate emotion;do

golddir=("/home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Mon_Jun__6_13:13:30_2022/" )

for dataset in sst;do
python train.py --dataset ${dataset} --data_dir . --output_dir test_adv_outputs/ \
  --encoder lstm --ours --gold_label_dir ${golddir[0]} --n_iters 80 \
    --exp_name $exp_name
done;
