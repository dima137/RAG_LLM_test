Test RAG + LLM combinations.

This is a gradio app hosted at Hugging Face:<br>
https://dvmalyshev-rag-llm-test.hf.space

The main goal is to test how different RAG approaches (e.g., vector embedding and BM25) work in combination with various LLMs.<br>
We use https://arxiv.org/abs/1407.7905 paper to build the text data for RAG in this example. 

One can ask questions in the chat and get answers, which combine RAG context and LLM response (the Chatbot answers only questions about Fermi bubbles and Galactic center region of the Milky way galaxy).

We use slash commands in the app.<br>
For instance, /help prints available commands and the corresponding short info.

One can also switch between RAGs and LLMs from the chat window using slash commands.<br>
For example, "/LLM" prints out the current LLM, "/LLMs" prints out the list of available LLMs, "/LLM=new_LLM_name" changes the current LLM to the LLM specified by "new_LLM_name" (NB: new_LLM_name can be a part of the LLM name - the script finds the first match).<br>
"@LLM question" sends the question to the LLM, e.g., question can be "what is the distance to Fermi bubbles".<br>
Similarly "@RAG question" sends the question to RAG and outputs the top results specified by "ktop" variable (can be changed by /ktop=new_value).<br>
"@RAG+LLM question" uses the ktop results from RAG as additional input to the LLM ("@RAG+LLM question" is equivalent to simply writing "question" in the chat).<br>
"/vars" prints all available variables and their current values, e.g., here is a typical output of /vars:<br>
Current variable values:<br>
LLM = Qwen/Qwen3-0.6B:featherless-ai<br>
RAG = transformer:all-MiniLM-L6-v2<br>
chunk_size = 1024<br>
chunk_overlap = 200<br>
chunk_overlap_BM25 = 0<br>
ktop = 5<br>
print_costs = True

Currently available LLM options (an output of /LLMs):<br>
gpt-5-mini<br>
gpt-5-nano<br>
claude-sonnet-5-5<br>
gemini-2.5-flash<br>
meta-llama/Llama-3.2-1B-Instruct:featherless-ai<br>
meta-llama/Llama-3.2-3B-Instruct:featherless-ai<br>
Qwen/Qwen3-0.6B:featherless-ai (current)<br>
google/gemma-3-4b-it:deepinfra<br>
Qwen/Qwen3.5-9B:together<br>
openai/gpt-oss-20b:groq<br>

The open source LLMs (such as meta-llama/Llama-3.2-1B-Instruct:featherless-ai and below) are accessed via HF interface, the proprietary LLMs are accessed via the corresponding APIs.

Currently available RAG options (an output of /RAGs):<br>
transformer:all-MiniLM-L6-v2 (current)<br>
words<br>
BM25<br>

Here "words" is a custom embedding based on frequency of words in the text. Similarity between query and chunk is defined as "score = sum_w q_w c_w / a_w", where the sum is over all words "w" common in query and chunk, q_w is the frequency of w in query, c_w - in chunk, and a_w - in the corpus (the arxiv:1407.7905 paper in our case). This metric increases the score when the same word happens more often in query and chunk, while a_w in the denominator reduces the significance of common words (such as is, the, a etc.). The metric does not account for the length of chunks (i.e., it would promote long chunks). Consequently, chunks of about the same size should be used.
