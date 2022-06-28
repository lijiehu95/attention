# REAL Attention 

We've based our repository on the [code](https://github.com/sarahwie/attention) provided by Sarthak Jain & Byron Wallace for their paper *[Attention is not not Explanation](https://arxiv.org/abs/1908.04626)*.

# Envrioment Setup
1. Ensure you have installed python environment in your computer. (Anaconda is recommended)
2. Use the following command to install the dependencies:
```
pip install -r requirements.txt
```
3. Preprocess the dataset using the following command:
```

```

Dependencies
--------------
- Python==3.6
- cuda_11.2

Data Preprocessing
--------------
Please perform the preprocessing instructions provided by Jain & Wallace [here](https://github.com/successar/AttentionExplanation/tree/master/preprocess). We replicated these instructions for the `Diabetes`, `Anemia`, `SST`, `IMDb`, `AgNews`, and `20News` datasets.

Running Baselines
--------------
We replicate the reported baselines in Jain & Wallace's paper (as reported in our paper in Table 2) by running the following commands:
- `./run_baselines.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`

Freezing the Attention Distribution (Section 3.1)
--------------
- `./run_frozen_attention.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`

Running Random Seeds Experiments (Section 3.2)
--------------
- `parallel ./run_seeds.sh :::: seeds.txt ::: sst AgNews imdb 20News_sports Diabetes Anemia`
- Code for constructing the violin plots in Figure 3 can be found in `seed_graphs.py` and `Seed_graphs.ipynb`.

Running BOWs Experiments (Section 3.3)
--------------
- To run the Bag of Words model with trained (MLP) attention weights: `./run_bows_baselines.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`
- To run the Bag of Words model with uniform attention weights: `./run_bows_frozen_attn.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst]`
- To run the Bag of Words model with frozen attention weights from another model: `./run_bows_set_to_pretrained_distribution.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst] [path/to/saved/model/with/attention/weights]`

Running Adversarial Model Experiments (Section 4)
--------------
- `./run_adversarial.sh [Diabetes, Anemia, AgNews, 20News_sports, imdb, sst] [lambda_value] [path/to/saved/model/with/gold/attentions/and/predictions]`
