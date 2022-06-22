from datasets import load_dataset,DatasetDict
for dataname in ['offensive', 'irony']:
    dataname = "offensive"
    load_data = ('tweet_eval',dataname)
    dataset = load_dataset(*load_data)
    df_trian = dataset['train'].to_pandas()
    df_trian = df_trian[df_trian['label'].apply(lambda x: x in [0,1])]
    df_trian['exp_split'] = "train"
    df_test = dataset['test'].to_pandas()
    df_test = df_test[df_test['label'].apply(lambda x: x in [0,1])]
    df_test['exp_split'] = "test"
    df = df_trian.append(df_test)
    df = df.dropna(axis=0,how='any')
    import os
    df_path = f'./preprocess/{dataname}'
    try:
        os.mkdir(df_path)
    except:
        pass
    df_file = f"./preprocess/{dataname}/{dataname}.csv"
    df.to_csv(df_file)

    import argparse
    parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
    parser.add_argument('--data_file', type=str,default='./data/emotion/emotion.csv')
    parser.add_argument("--output_file", type=str,default="./data/emotion/vec.p")
    parser.add_argument('--word_vectors_type', type=str, choices=['fasttext.simple.300d'], default="fasttext.simple.300d")
    parser.add_argument('--min_df', type=int,default=1)
    parser.add_argument(
            '-f',
            '--file',
            help='Path for input file. First line should contain number of lines to search in'
        )
    args, extras = parser.parse_known_args()
    args.extras = extras

    args.data_file = df_file
    args.output_file = os.path.join(df_path,'vec.p')

    from attention.preprocess import vectorizer
    vec = vectorizer.Vectorizer(min_df=args.min_df)

    assert 'text' in df.columns, "No Text Field"
    assert 'label' in df.columns, "No Label Field"
    assert 'exp_split' in df.columns, "No Experimental splits defined"

    texts = list(df[df.exp_split == 'train']['text'])
    # assert np.nan not in texts
    # for i in texts:
    #     if i == np.nan:
    #         print(i)
    vec.fit(texts)

    print("Vocabulary size : ", vec.vocab_size)

    vec.seq_text = {}
    vec.label = {}
    vec.raw_text = {}

    splits = df.exp_split.unique()
    for k in splits :
        split_texts = list(df[df.exp_split == k]['text'])
        vec.raw_text[k] = split_texts
        vec.seq_text[k] = vec.get_seq_for_docs(split_texts)
        vec.label[k] = list(df[df.exp_split == k]['label'])

    if args.word_vectors_type in ['fasttext.simple.300d'] :
        vec.extract_embeddings_from_torchtext(args.word_vectors_type,cache="./attention/preprocess/.vector_cache")
    else :
        vec.embeddings = None

    import pickle, os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    pickle.dump(vec, open(args.output_file, 'wb'))

    #
    #
    #
    # from datasets import load_dataset
    # dataname = ('tweet_eval','emotion')
    # dataset = load_dataset(*dataname)
    # df_trian = dataset['train'].to_pandas()
    # df_trian = df_trian[df_trian['label'].apply(lambda x: x in [0,1])]
    # df_trian['exp_split'] = "train"
    # df_test = dataset['test'].to_pandas()
    # df_test = df_test[df_test['label'].apply(lambda x: x in [0,1])]
    # df_test['exp_split'] = "test"
    # df = df_trian.append(df_test)
    # import os
    # dataname = "emotion"
    # df_path = f'./preprocess/{dataname}'
    # try:
    #     os.mkdir(df_path)
    # except:
    #     pass
    # df_file = f"./preprocess/{dataname}/{dataname}.csv"
    # df.to_csv(df_file)
    # import argparse
    # parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
    # parser.add_argument('--data_file', type=str,default='./data/emotion/emotion.csv')
    # parser.add_argument("--output_file", type=str,default="./data/emotion/vec.p")
    # parser.add_argument('--word_vectors_type', type=str, choices=['fasttext.simple.300d'], default="fasttext.simple.300d")
    # parser.add_argument('--min_df', type=int,default=1)
    # parser.add_argument(
    #         '-f',
    #         '--file',
    #         help='Path for input file. First line should contain number of lines to search in'
    #     )
    # args, extras = parser.parse_known_args()
    # args.extras = extras
    #
    # args.data_file = df_file
    # args.output_file = os.path.join(df_path,'vec.p')
    #
    # from attention.preprocess import vectorizer
    # vec = vectorizer.Vectorizer(min_df=args.min_df)
    #
    # import pandas as pd
    #
    # df = pd.read_csv(args.data_file) if args.data_file.endswith('.csv') else pd.read_msgpack(args.data_file)
    # assert 'text' in df.columns, "No Text Field"
    # assert 'label' in df.columns, "No Label Field"
    # assert 'exp_split' in df.columns, "No Experimental splits defined"
    #
    # texts = list(df[df.exp_split == 'train']['text'])
    # vec.fit(texts)
    #
    # print("Vocabulary size : ", vec.vocab_size)
    #
    # vec.seq_text = {}
    # vec.label = {}
    # vec.raw_text = {}
    #
    # splits = df.exp_split.unique()
    # for k in splits :
    #     split_texts = list(df[df.exp_split == k]['text'])
    #     vec.raw_text[k] = split_texts
    #     vec.seq_text[k] = vec.get_seq_for_docs(split_texts)
    #     vec.label[k] = list(df[df.exp_split == k]['label'])
    #
    # if args.word_vectors_type in ['fasttext.simple.300d', 'glove.840B.300d'] :
    #     vec.extract_embeddings_from_torchtext(args.word_vectors_type,cache="./attention/preprocess/.vector_cache")
    # else :
    #     vec.embeddings = None
    #
    # import pickle, os
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # pickle.dump(vec, open(args.output_file, 'wb'))