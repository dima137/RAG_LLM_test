#from pypdf import PdfReader
from PyPDF2 import PdfReader
import pandas as pd
import bm25s
import numpy as np


def pdf2text(pdf_fn, PdfReader=PdfReader):
    reader = PdfReader(pdf_fn)
    pages = [page.extract_text() for page in reader.pages]
    text = '\n'.join(pages)
    return text


def get_words(text):
    '''
        simple tokenizer that returns words
    '''
    words = text.split()
    words = [word.strip('.,!?();') for word in words]
    words = [word.lower() for word in words]
    return words

def tokens2freq_df(tokens):
    token_freq = {}
    for token in tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    # convert to pandas dataframe with tokens as index
    df = pd.DataFrame(list(token_freq.items()), columns=['token', 'frequency']).set_index('token')
    return df


def token_freq_df_embed_documents(tokenizer=get_words):
    def embedding(texts):
        return [tokens2freq_df(tokenizer(text)) for text in texts]
    return embedding


def freq_weighted(q_df, chunk_df, word_freq_df, full_out=False):
    '''
        Get similarity between two documents based on word frequencies.
    '''
    index_intersection = q_df.index.intersection(chunk_df.index)
    index_intersection = index_intersection.intersection(word_freq_df.index)
    if len(index_intersection) == 0:
        return 0.0
    else:
        q_vector = q_df.loc[index_intersection, 'frequency'].values
        chunk_vector = chunk_df.loc[index_intersection, 'frequency'].values
        token_vector = word_freq_df.loc[index_intersection, 'frequency'].values
        score = np.sum(q_vector * chunk_vector / token_vector)
        if full_out:
            return score, index_intersection
        else:
            return score

def token_similarity_weighted(word_freq_df):
    def similarity(q_dfs, chunk_dfs):
        return [freq_weighted(q_dfs[0], c, word_freq_df) for c in chunk_dfs]
    return similarity

def add_dfs(dfs, col_name='frequency'):
    '''
    add dataframes along one column (ignore the others)
    with the union of indices
    '''
    if len(dfs) == 1:
        return dfs[0]
    else:
        # determine the union of indices
        index = dfs[0].index.copy()
        for df in dfs[1:]:
            index = index.union(df.index)
        # initialize the output dataframe with defaule values = 0
        res_df = pd.DataFrame({col_name: 0.}, index=index)
    # add values to the output dataframe
    for df in dfs:
        res_df.loc[df.index, col_name] += df.loc[df.index, col_name]
    return res_df


def bm25_embedding(docs):
    return bm25s.tokenize(docs, stopwords="en")

def bm25_metric(bm25qs, bm25chunks):
    retriever = bm25s.BM25()
    retriever.index(bm25chunks)
    scores = retriever.get_scores(list(bm25qs.vocab.keys()))
    return scores
