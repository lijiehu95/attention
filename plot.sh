for dataset in hate rotten_tomatoes  imdb sst emoji  \
                sentiment  stance_abortion  stance_atheism  stance_climate  stance_feminist  \
                stance_hillary;do
for att in tanh;do
for encoder in lstm simple-rnn; do
python seed_graphs.py --dataset $dataset --model_type $encoder+$att
done; done; done;
