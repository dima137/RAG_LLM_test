

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from ordered_set import OrderedSet

import RAG_lib as rlib



# Variables
var_dict = {}
# ~ 1000 words or tokens (~ 2 pages) is a good chunk size for scientific texts
# ~ 500 words (~ 1 page) may be OK too
var_dict['chunk_size'] = 1024 # 512
# overlap between chunks - useful if context is at the boundary of a chunk
var_dict['chunk_overlap'] = 200
var_dict['chunk_overlap_BM25'] = 0 # overlap for BM25 RAG
#chunk_overlap_dict = {'BM25':0} # custom overlap for different RAGs
var_dict['ktop'] =  5 # if 0, then RAG is not used

# BM25, vector
#RAG_representation = 'tokens' # 'tokens', 'vectors'
var_dict['RAG'] = 'words'
#RAG_metric = 'overlap_weighted'
RAG_options = {}
RAG_options['tokens'] = {}
RAG_options['vectors'] = {}

# use a dictionary to query metrics for calculating similarity of embeddings
metric_dict = {'cosine': lambda xs, ys: cosine_similarity(xs, ys)[0]}

# an example of a vector model
RAG_options['vectors']['ollama:nomic-embed-text_cos']  = {}
RAG_options['vectors']['ollama:nomic-embed-text_cos']['embedding']  = \
    OllamaEmbeddings(model="nomic-embed-text").embed_documents
RAG_options['vectors']['ollama:nomic-embed-text_cos']['metric'] = 'cosine'
    

# an example of a token model: words
RAG_options['tokens']['words'] = {}
RAG_options['tokens']['words']['embedding'] = \
    rlib.token_freq_df_embed_documents(tokenizer=rlib.get_words)
RAG_options['tokens']['words']['metric'] = 'token_similarity_weighted'

# an example of a token model: words + BM25
RAG_options['tokens']['BM25'] = {}
RAG_options['tokens']['BM25']['embedding'] = rlib.bm25_embedding
RAG_options['tokens']['BM25']['metric'] = 'BM25_default'

metric_dict['BM25_default'] = rlib.bm25_metric

RAG_dict = {'vectors':['ollama:nomic-embed-text_cos'],
             'tokens':['words', 'BM25']
             }

var_dict['RAG_dict'] = RAG_dict

var_dict['RAGs'] = OrderedSet()
for RAGs in RAG_dict.values():
    var_dict['RAGs'] = var_dict['RAGs'].union(RAGs)

# LLM options
#LLMs = ['gpt-5-mini', 'gpt-5-nano', 'llama3.2', "claude-sonnet-4-5", "gemini-2.5-flash"]
LLM_dict = {'openai':['gpt-5-mini', 'gpt-5-nano'],
             'anthropic':["claude-sonnet-4-5"],
             'google':["gemini-2.5-flash"],
             'hf':[]}

LLM_dict['hf'] = ["meta-llama/Llama-3.2-1B-Instruct:featherless-ai", # 0
                    "meta-llama/Llama-3.2-3B-Instruct:featherless-ai", # 1
                    "Qwen/Qwen3-0.6B:featherless-ai", # 2
                    "google/gemma-3-4b-it:deepinfra", # 3
                    "Qwen/Qwen3.5-9B:together", # 4
                    "openai/gpt-oss-20b:groq", # 5
                    ]

var_dict['LLM_dict'] = LLM_dict

var_dict['LLMs'] = OrderedSet()
for LLMs in LLM_dict.values():
    var_dict['LLMs'] = var_dict['LLMs'].union(LLMs)

#hf_models = ["meta-llama/Llama-3.2-3B:auto", "openai/gpt-oss-20b:auto", "Qwen/Qwen3.5-9B:auto"]

var_dict['RAG'] = 'words'
var_dict['LLM'] = 'gpt-5-mini'


# load environment variables from .env
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
hf_api_key = os.getenv('HF_TOKEN')

if 1:
    # test if api keys exist
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")
        
    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set (and this is optional)")
        
    if google_api_key:
        print(f"Google API Key exists and begins {google_api_key[:2]}")
    else:
        print("Google API Key not set (and this is optional)")

    if hf_api_key:
        print(f"HF API Key exists and begins {hf_api_key[:2]}")
    else:
        print("HF API Key not set (and this is optional)")





def openai_bot(var_dict):
    model = var_dict['LLM']
    full_response = var_dict.get('full_response', 0)
    bot = OpenAI()
    def chatbot(messages, full_response=full_response):
        response = bot.chat.completions.create(model=model, messages=messages)
        response_dict = {'text': response.choices[0].message.content}
        usage = response.usage
        response_dict['usage'] = {'prompt_tokens': usage.prompt_tokens,
                                  'completion_tokens': usage.completion_tokens,
                                  'reasoning_tokens': 0}
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
            response_dict['usage']['reasoning_tokens'] = reasoning_tokens
        if full_response:
            return response
        else:
            return response_dict
    return chatbot

def google_bot(var_dict):
    model = var_dict['LLM']
    full_response = var_dict.get('full_response', 0)
    bot = OpenAI(api_key=google_api_key, 
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    def chatbot(messages, full_response=full_response):
        response = bot.chat.completions.create(model=model, messages=messages)
        response_dict = {'text': response.choices[0].message.content}
        usage = response.usage
        response_dict['usage'] = {'prompt_tokens': usage.prompt_tokens,
                                  'completion_tokens': usage.completion_tokens,
                                  'reasoning_tokens': 0}
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
            response_dict['usage']['reasoning_tokens'] = reasoning_tokens
        return response_dict
    return chatbot


def anthropic_bot(var_dict):
    model = var_dict['LLM']
    full_response = var_dict.get('full_response', 0)
    bot = Anthropic()
    def chatbot(messages, full_response=full_response):
        response = bot.messages.create(model=model, messages=messages, max_tokens=1000)
        response_dict = {'text': response.content[0].text}
        usage = response.usage
        # placeholder for usage for anthropic
        response_dict['usage'] = {'prompt_tokens': usage.input_tokens,
                                  'completion_tokens': usage.output_tokens,
                                  'reasoning_tokens': 0}
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details.thinking_tokens:
            reasoning_tokens = usage.output_tokens_details.thinking_tokens
            response_dict['usage']['reasoning_tokens'] = reasoning_tokens
        return response_dict
    return chatbot



def hf_bot(var_dict):
    model = var_dict['LLM']
    full_response = var_dict.get('full_response', 0)
    bot = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_api_key
        )

    def chatbot(messages, full_response=full_response):
        response = bot.chat.completions.create(model=model, messages=messages)
        response = bot.chat.completions.create(
                model=model,  # ":any other providers" is required
                messages=messages)
        response_text = str(response.choices[0].message.content)
        response_dict = {'text': response_text}
        usage = response.usage
        response_dict['usage'] = {'prompt_tokens': usage.prompt_tokens,
                                  'completion_tokens': usage.completion_tokens,
                                  'reasoning_tokens': 0}
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
            response_dict['usage']['reasoning_tokens'] = reasoning_tokens
        return response_dict
    return chatbot


def get_bot(var_dict):
    model = var_dict['LLM']
    LLM_dict = var_dict['LLM_dict']
    if model in LLM_dict.get('openai', []):
        return openai_bot(var_dict)
    elif model in LLM_dict.get('anthropic', []):
        return anthropic_bot(var_dict)
    elif model in LLM_dict.get('google', []):
        return google_bot(var_dict)
    elif model in LLM_dict.get('hf', []):
        return hf_bot(var_dict)
    else:
        return None



## RAG preprocessing
# converte pdf to text and save (if not yet converted)
# generalize later for on-demand additions to RAG
fn0 = 'LAT_FBs-1407.7905v1' # Fermi LAT paper about the Fermi bubbles
pdf_fn = 'RAG_data/%s.pdf' % fn0
txt_fn = 'RAG_data/%s.txt' % fn0
rewrite_txt = False
if not os.path.exists(txt_fn) or rewrite_txt:
    text = rlib.pdf2text(pdf_fn)
    with open(txt_fn, "w", encoding="utf-8") as f:
        f.write(text)
else:
    text = open(txt_fn, "r", encoding="utf-8").read()
var_dict['text'] = text

# calculate the word frequency in the corpus
text_dfs = RAG_options['tokens']['words']['embedding']([text])
word_freq_df = rlib.add_dfs(text_dfs, col_name='frequency')
    

# use the word frequencies as inverse weights
metric_dict['token_similarity_weighted'] = \
    rlib.token_similarity_weighted(word_freq_df)



## RAGs setup
def get_type(model, type_dict):
    for key, models in type_dict.items():
        if model in models:
            return key
    return None


def RAG_response(query, var_dict):
    RAG = var_dict['RAG']
    RAG_dict = var_dict['RAG_dict']
    ktop = var_dict['ktop']
    chunk_overlap = var_dict['chunk_overlap']
    chunk_size = var_dict['chunk_size']
    text = var_dict['text']
    
    if RAG == 'BM25':
        chunk_overlap = var_dict['chunk_overlap_BM25']

    RAG_type = get_type(RAG, RAG_dict)
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    embedding = RAG_options[RAG_type][RAG]['embedding']
    metric_name = RAG_options[RAG_type][RAG]['metric']
    metric = metric_dict[metric_name]
    emb_chunks = embedding(chunks)
    emb_qs = embedding([query])

    scores = metric(emb_qs, emb_chunks)
    inds = np.argsort(scores)[::-1][:ktop]

    message = ''
    for i, ind in enumerate(inds):
        message += f'{i + 1}: {chunks[ind]}\n\n'
    return message



## Variables and commands for the chat interface
vars = ['LLM', 'RAG', 'chunk_size', 'chunk_overlap', 'chunk_overlap_BM25', 'ktop']
special_commands = ['/help', '/LLMs', '/RAGs', '/vars']
var_dict['vars'] = vars
var_dict['special_commands'] = special_commands


def chat(message, history, var_dict=var_dict):
    messages = history + [message]
    if message.startswith('/'):
        response = rlib.execute(message, var_dict=var_dict)
        return response
    if message.find('@') == -1:
        message = '@RAG+LLM: ' + message

    target = message.split(' ')[0]
    query = message.replace(target, '').strip()
    response = ''
    if var_dict['ktop'] > 0 and message.lower().find('rag') > -1:
        # determine RAG response (or context if LLM is used)
        response += f'RAG "{var_dict["RAG"]}" top {var_dict["ktop"]} chunks '
        answer = RAG_response(query, var_dict)
    if message.lower().find('llm') > -1:
        messages = [{"role": "user", "content": query}]
        if len(response) > 0:
            response += '+ '
            messages.append({"role": "user", "content": answer})
        
        
        bot = get_bot(var_dict)
        bot_response = bot(messages)
        answer = bot_response['text'] # if LLM is used: overwrite answer with LLM response

        response += f'LLM "{var_dict["LLM"]}" '
    response += f'response:\n\n{answer}'

    return response


if __name__ == "__main__":
    chat_box = gr.ChatInterface(chat, #type="messages",     
                            textbox=gr.Textbox(placeholder="type /help for available commands", 
                                               lines=1, max_lines=30))


    chat_box.launch()
