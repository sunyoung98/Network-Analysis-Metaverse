# %%
import pandas as pd
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# %%
tqdm.pandas()
lemmatizer = WordNetLemmatizer()

# %%
def tag2sym(tag):
    if tag == 'NOUN':
        return 'n'
    elif tag == 'VERB':
        return 'v'
    elif tag == 'ADJ':
        return 'a'
    elif tag == 'ADV':
        return 'r'
    else:
        return 'n'

# %%
for modelname in [path for path in os.listdir('./datasets/') if path[:7] == 'keybert']:
    print(f'PROCESSING {modelname}')

    df1 = pd.read_csv(f'./datasets/{modelname}/roblox1.csv', low_memory = False, index_col = 0)
    df2 = pd.read_csv(f'./datasets/{modelname}/roblox2.csv', low_memory = False, index_col = 0)
    df3 = pd.read_csv(f'./datasets/{modelname}/roblox3.csv', low_memory = False, index_col = 0)
    df4 = pd.read_csv(f'./datasets/{modelname}/roblox4.csv', low_memory = False, index_col = 0)
    df5 = pd.read_csv(f'./datasets/{modelname}/roblox5.csv', low_memory = False, index_col = 0)
    df6 = pd.read_csv(f'./datasets/{modelname}/zepeto.csv', low_memory = False, index_col = 0)

    df1['keybert_keywords_lemmatized'] = df1['keybert_keywords'].progress_apply(lambda array : [(lemmatizer.lemmatize(word, tag2sym(pos_tag([word]))), score) for word, score in eval(array)])
    df2['keybert_keywords_lemmatized'] = df2['keybert_keywords'].progress_apply(lambda array : [(lemmatizer.lemmatize(word, tag2sym(pos_tag([word]))), score) for word, score in eval(array)])
    df3['keybert_keywords_lemmatized'] = df3['keybert_keywords'].progress_apply(lambda array : [(lemmatizer.lemmatize(word, tag2sym(pos_tag([word]))), score) for word, score in eval(array)])
    df4['keybert_keywords_lemmatized'] = df4['keybert_keywords'].progress_apply(lambda array : [(lemmatizer.lemmatize(word, tag2sym(pos_tag([word]))), score) for word, score in eval(array)])
    df5['keybert_keywords_lemmatized'] = df5['keybert_keywords'].progress_apply(lambda array : [(lemmatizer.lemmatize(word, tag2sym(pos_tag([word]))), score) for word, score in eval(array)])
    df6['keybert_keywords_lemmatized'] = df6['keybert_keywords'].progress_apply(lambda array : [(lemmatizer.lemmatize(word, tag2sym(pos_tag([word]))), score) for word, score in eval(array)])

    os.makedirs(f'./datasets/{modelname}-post-tokenized/')

    df1.to_csv(f'./datasets/{modelname}-post-tokenized/roblox1.csv')
    df2.to_csv(f'./datasets/{modelname}-post-tokenized/roblox2.csv')
    df3.to_csv(f'./datasets/{modelname}-post-tokenized/roblox3.csv')
    df4.to_csv(f'./datasets/{modelname}-post-tokenized/roblox4.csv')
    df5.to_csv(f'./datasets/{modelname}-post-tokenized/roblox5.csv')
    df6.to_csv(f'./datasets/{modelname}-post-tokenized/zepeto.csv')

# %%



