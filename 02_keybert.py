# %%
from keybert import KeyBERT
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
import os
from tqdm import tqdm, trange
from random import randint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# %%
tqdm.pandas()

# %%
def morphs(text, noun = True, verb = False, adjective = False, adverb = False):
    tokens = word_tokenize(text)
    poses = pos_tag(tokens, tagset = 'universal')
    filters = []

    if noun:
        filters.append('NOUN')
    if verb:
        filters.append('VERB')
    if adjective:
        filters.append('ADJ')
    if adverb:
        filters.append('ADV')

    return [pos[0] for pos in poses if pos[1] in filters]

# %%
models = [
    'all-mpnet-base-v2',
    'multi-qa-mpnet-base-dot-v1',
    'all-distilroberta-v1',
    'all-MiniLM-L12-v2',
    'multi-qa-distilbert-cos-v1'
]

# %%
for model_name in models:
    print(f'TRAINING ON {model_name}')

    model = KeyBERT(model_name)
    
    df1 = pd.read_csv('./datasets/original/roblox1.csv', index_col = 0, low_memory = False)
    df2 = pd.read_csv('./datasets/original/roblox2.csv', index_col = 0, low_memory = False)
    df3 = pd.read_csv('./datasets/original/roblox3.csv', index_col = 0, low_memory = False)
    df4 = pd.read_csv('./datasets/original/roblox4.csv', index_col = 0, low_memory = False)
    df5 = pd.read_csv('./datasets/original/roblox5.csv', index_col = 0, low_memory = False)
    df6 = pd.read_csv('./datasets/original/zepeto.csv', index_col = 0, low_memory = False)

    df1['keybert_keywords'] = df1['content'].progress_apply(lambda x : model.extract_keywords(x, top_n = 10))
    df2['keybert_keywords'] = df2['content'].progress_apply(lambda x : model.extract_keywords(x, top_n = 10))
    df3['keybert_keywords'] = df3['content'].progress_apply(lambda x : model.extract_keywords(x, top_n = 10))
    df4['keybert_keywords'] = df4['content'].progress_apply(lambda x : model.extract_keywords(x, top_n = 10))
    df5['keybert_keywords'] = df5['content'].progress_apply(lambda x : model.extract_keywords(x, top_n = 10))
    df6['keybert_keywords'] = df6['content'].progress_apply(lambda x : model.extract_keywords(x, top_n = 10))

    os.makedirs(f'./datasets/keybert-{model_name}')

    df1.reset_index(drop = True).to_csv(f'./datasets/keybert-{model_name}/roblox1.csv')
    df2.reset_index(drop = True).to_csv(f'./datasets/keybert-{model_name}/roblox2.csv')
    df3.reset_index(drop = True).to_csv(f'./datasets/keybert-{model_name}/roblox3.csv')
    df4.reset_index(drop = True).to_csv(f'./datasets/keybert-{model_name}/roblox4.csv')
    df5.reset_index(drop = True).to_csv(f'./datasets/keybert-{model_name}/roblox5.csv')
    df6.reset_index(drop = True).to_csv(f'./datasets/keybert-{model_name}/zepeto.csv')

# %%



