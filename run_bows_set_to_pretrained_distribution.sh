

source activate xai
export PYTHONPATH=/home/yila22/prj
export CUDA_VISIBLE_DEVICES='4'
exp_name="debug-freeze-bows-with-train-att"


for dataset in sst imdb hate emotion;do
  python train_and_run_experiments_bc.py --dataset dataset --data_dir . --output_dir outputs/ \
    --encoder average --attention pre-loaded --gold_label_dir ${1} \
      --exp_name $exp_name
done;


