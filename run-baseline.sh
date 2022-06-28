exp_name="baseline"
for model in simple-rnn lstm; do
for dataset in hate rotten_tomatoes  imdb SetFit/sst5 emoji  \
                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
                stance_hillary ; do
python train.py --dataset $dataset --data_dir . --output_dir test_outputs/ --attention tanh \
    --encoder $model --exp_name $exp_name --baseline
done;done;