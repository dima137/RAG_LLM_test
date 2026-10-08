#from pypdf import PdfReader
from PyPDF2 import PdfReader
import pandas as pd
import bm25s
import numpy as np
#from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

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




def list2lines(lst, current=None):
    res = ''
    for x in lst:
        if x == current:
            res += f'\t- {x} (current)\n'
        else:
            res += f'\t- {x}\n'
    return res

def dict2lines(dct):
    res = ''
    for key, value in dct.items():
        res += f'\n{key}:'
        if type(value) is list:
            res += '\n' + list2lines(value)
        else:
            res += f' {value}'
    return res

commands_dict = {
    '/help': 'print the list of available commands etc.',
    '/LLM': 'print current LLM',
    '/LLMs': 'print available LLMs',
    '/RAG': 'print current RAG',
    '/RAGs': 'print available RAGs',
    '/LLM=new_LLM_model': 'change LLM specified by "new_LLM_model" \
        (partical match is supported: the first instance is returned)',
    '/RAG=new_RAG_model': 'change RAG specified by "new_RAG_model" \
        (partical match is supported: the first instance is returned)',
    '/vars': 'print the current values of all variables',
    '@RAG query': 'address the query to RAG, response is "ktop" top chunks',
    '@LLM query': 'address the query to LLM',
    '@LLM+RAG query or @RAG+LLM query or query': 'Use LLM with RAG context using "ktop" \
        top chunks (if ktop > 0)',
    'info': 'in general "/var_name" prints the current value of the variable, \
        and "/var_name=new_value" changes the value of the variable to "new_value"',
}




def handle_special_commands(command, var_dict={}):
    if command not in var_dict['special_commands']:
        result = f'Command {command} is not found.\n'
        result += handle_special_commands('help', var_dict=var_dict)
        return result
    if command == '/help':
        result = 'Available commands:'
        result += dict2lines(commands_dict)
        return result
    elif command == '/LLMs':
        result = 'Available LLMs:\n'
        result += list2lines(var_dict['LLMs'], current=var_dict['LLM'])
        return result
    elif command == '/RAGs':
        result = 'Available RAGs:\n'
        result += list2lines(var_dict['RAGs'], current=var_dict['RAG'])
        return result
    elif command == '/vars':
        result = 'Current variable values:\n'
        for var in var_dict['vars']:
            result += f'{var} = {var_dict[var]}\n'
        return result

def get_model_index(model, model_list):
    for ind, mdl in enumerate(model_list):
        if mdl.find(model) > -1:
            return ind
    return -1

def find_new_value(model, model_list):
    ind = get_model_index(model, model_list)
    if ind == -1:
        return None
    else:
        return model_list[ind]

def handle_model_commands(message, var_dict={}):
    if not message.startswith('/'):
        return None
    command = message.split(' ')[0]
    
    if command in var_dict['special_commands']:
        return handle_special_commands(command, var_dict=var_dict)
    elif message.find('=') > -1:
        var_name = message.split('=')[0]
    else:
        var_name = message.split(' ')[0]
    var_name = var_name.replace('/', '').strip()

    if var_name not in var_dict['vars']:
        out_message = f'{command} is not found.\n'
        out_message += handle_special_commands('/help', var_dict=var_dict)
        return out_message
    
    if message.find('=') > -1:
        new_value = message.split('=')[1].strip()
        old_value = var_dict[var_name]
        
        if var_name == 'LLM':
            if get_model_index(new_value, var_dict['LLMs']) == -1:
                out_message = f'Model "{new_value}" not found.\n'
                out_message += handle_special_commands('/LLMs', var_dict=var_dict)
                return out_message
            else:
                new_value = find_new_value(new_value, var_dict['LLMs'])
        elif var_name == 'RAG':
            if get_model_index(new_value, var_dict['RAGs']) == -1:
                out_message = f'Model "{new_value}" not found.\n'
                out_message += handle_special_commands('/RAGs', var_dict=var_dict)
                return out_message
            else:
                new_value = find_new_value(new_value, var_dict['RAGs'])
        if new_value.isnumeric():
            new_value = int(new_value)

        var_dict[var_name] = new_value

        out_message = f'Old {var_name} = {old_value}, new {var_name} = {new_value}'
    else:
        out_message = f'Current {var_name} = {var_dict[var_name]}'

    return out_message


def handle_help(message, var_dict={}):
    return 'Print information about commands and other options'
    
def execute(message, var_dict={}):
    if not message.startswith('/'):
        return None
    res = handle_model_commands(message, var_dict=var_dict)
    if res: return res
    res = handle_help(message, var_dict=var_dict)
    if res: return res
    return None


