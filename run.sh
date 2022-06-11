#export CUDA_VISIBLE_DEVICES=4
source activate xai
export PYTHONPATH=/home/yila22/prj
exp_name="debug-topk-v3"
i=0 # gpu pointer
gpu=(0 2 3 4 5 6 7 9)
gpunum=8
for data in sst; do
  for lambda_1 in 0;do
    for lambda_2 in  1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
      for topk_prox_metric in l1 l2 kl jsd;do
        for K in 3 5 7;do
          i=`expr $i % $gpunum`
          export CUDA_VISIBLE_DEVICES=${gpu[$i]}
          echo "use gpu id is ${gpu[$i]}"

          com="python train.py --exp_name $exp_name --dataset $data \
            --data_dir . --output_dir test_adv_outputs/ \
            --encoder lstm --ours --lambda_1 $lambda_1 --lambda_2 $lambda_2 --topk_prox_metric $topk_prox_metric --K $K\
            --gold_label_dir /home/yila22/prj/attention/test_outputs/sst/lstm+tanh/Mon_Jun__6_13:13:30_2022/ --n_iters 80"
#          $com
          if [ $i == 7 ]
          then
            $com
          else
            nohup $com > ./logs/$data-${RANDOM}.txt 2>&1 &
          fi
          i=`expr $i + 1`
done;done;done;done;done

for data in imdb; do
  for lambda_1 in 0;do
    for lambda_2 in  1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
      for topk_prox_metric in l1 l2 kl jsd;do
        for K in 3 5 7;do
          i=`expr $i % $gpunum`
          export CUDA_VISIBLE_DEVICES=${gpu[$i]}
          echo "use gpu id is ${gpu[$i]}"
          com="python train.py --exp_name $exp_name --dataset $data \
            --data_dir . --output_dir test_adv_outputs/ \
            --encoder lstm --ours --lambda_1 $lambda_1 --lambda_2 $lambda_2 --topk_prox_metric $topk_prox_metric --K $K\
            --gold_label_dir /home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Thu_Jun__9_07:14:57_2022  --n_iters 80"
#          $com
          if [ $i == 7 ]
          then
            $com
          else
            nohup $com > ./logs/$data-${RANDOM}.txt 2>&1 &
          fi
          i=`expr $i + 1`
done;done;done;done;done



#for data in imdb; do
#  for lambda_1 in 1 1e-4 0;do
#    for lambda_2 in  1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7;do
#      for topk_prox_metric in l1 l2 kl jsd;do
#        for K in 2 3 4 5;do
#          python train.py --exp_name $exp_name --dataset $data \
#            --data_dir . --output_dir test_adv_outputs/ \
#            --encoder lstm --ours --lambda_1 $lambda_1 --lambda_2 $lambda_2 --topk_prox_metric $topk_prox_metric --K $K\
#            --gold_label_dir /home/yila22/prj/attention/test_outputs/imdb/lstm+tanh/Thu_Jun__9_07:14:57_2022 --n_iters 80
#done;done;done;done;done
