exp_name="seed-v1"

n_iters=40
for model in lstm simple-rnn; do
for dataset in hate rotten_tomatoes  imdb SetFit/sst5 emoji  \
                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
                stance_hillary;do
for seed in 50 257 500231 2; do
 com="python train.py --dataset $dataset --data_dir . --output_dir seed_output/ --attention tanh --encoder $model --seed $seed \
  --exp_name $exp_name --n_iters $n_iters"
  $com
done;done;done