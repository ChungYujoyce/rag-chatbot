import os, re
import chainlit as cl
import pickle
import chromadb
from utils import clean_text

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever, RouterRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever

import argparse
parser = argparse.ArgumentParser(description='Parse command-line arguments')
parser.add_argument('--port', type=int, help='Port number')
parser.add_argument('-w', help='w')
parser.add_argument('run', help='run')
args = parser.parse_args()

LLM = OpenAILike(
        model="llama3-8b-instruct", 
        api_base="http://vllm:8000/v1", 
        api_key="fake",
        temperature=0.0,
        max_tokens=512,
    )

SYS_PROMPT = PromptTemplate(
    """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You're a Supermicro product expert, and you will answer questions based on the provided documents. 
To answer questions accurately, you may need to extract information in tables, figures, diagrams.  Do not use any other information to answer the question and provide a detailed answer to the question, highlighting Supermicro and its products' power savings and green computing benefits.
Firstly, highlight GPU characteristics and GPU products. Secondly, mention power-related features, such as fan cooling or liquid cooling, power consumption, etc. Thirdly, try to mention green computing benefits.
Please remember:
1. Supermicro does not have CPU or GPU products.
2. X12 and X13 use Intel CPU, while H12 and H13 use AMD CPU.
3. Server is different from Workstation.

If you don't know the answer, please say "Sorry, I haven't been trained with that data yet."

Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query as concise as possible..<|eot_id|>
{query_str}\
"""
)
CONVERSATION_PROMPT = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
RESPONSE_PROMPT = "<|start_header_id|>{role}<|end_header_id|>\n\n"
EMBEDDING = InstructorEmbedding(model_name="hkunlp/instructor-xl")
DB_PATH = "./chroma_db_v1"
DB_PATH_DS = "./chroma_db_v1_ds"
chroma_client = chromadb.PersistentClient(DB_PATH)
chroma_collection = chroma_client.get_collection("test_v1")
VECTOR_STORE = ChromaVectorStore(chroma_collection=chroma_collection)

chroma_client2 = chromadb.PersistentClient(DB_PATH_DS)
chroma_collection2 = chroma_client2.get_collection("test_v1")
VECTOR_STORE_DS = ChromaVectorStore(chroma_collection=chroma_collection2)

Settings.llm = LLM
Settings.embed_model = EMBEDDING
Settings.num_output = 512
Settings.context_window = 8192

index = VectorStoreIndex.from_vector_store(
    vector_store=VECTOR_STORE,
    embed_model=EMBEDDING,
)
index_ds = VectorStoreIndex.from_vector_store(
    vector_store=VECTOR_STORE_DS,
    embed_model=EMBEDDING,
)

with open(f"{DB_PATH}/nodes.pickle", 'rb') as f:
    global nodes
    nodes = pickle.load(f)

with open(f"{DB_PATH_DS}/nodes.pickle", 'rb') as f:
    global nodes_ds
    nodes_ds = pickle.load(f)


bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, tokenizer=clean_text, similarity_top_k=5)
bm25_retriever_ds = BM25Retriever.from_defaults(nodes=nodes_ds, tokenizer=clean_text, similarity_top_k=10)

def get_query(input_text, start_tag, end_tag):
    # Construct the regular expression pattern to match text between start_tag and end_tag
    pattern = re.compile(rf'{re.escape(start_tag)}(.*?){re.escape(end_tag)}', re.DOTALL)
    # Use re.findall() to find all occurrences of the pattern in the input text
    matches = re.findall(pattern, input_text)
    return matches


# Advanced - Hybrid Retriever + Re-Ranking
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


def query_engine_router(index, type_):
    # vector_retriever = index.as_retriever(similarity_top_k=7)
    # hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever4) # 10 docs
    #step_decompose_transform = StepDecomposeQueryTransform(llm=LLM, verbose=True)
    #rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=4)
    #reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")
    rankergpt = RankGPTRerank(
            llm=LLM,
            top_n=10,
            verbose=True,
        )

    query_engine = None
    if type_ == "hybrid": # hybrid
        vector_retriever = index.as_retriever(similarity_top_k=7)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever) # 15 docs

        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            #node_postprocessors=[reranker],
            node_postprocessors=[rankergpt],
            llm=LLM,
            streaming=True,
            text_qa_template=SYS_PROMPT,
        )
    elif type_ == "bm25_spec": # bm25
        query_engine = RetrieverQueryEngine.from_args(
            retriever=bm25_retriever_ds,
            llm=LLM,
            streaming=True,
            text_qa_template=SYS_PROMPT,
        )

    return query_engine


@cl.on_chat_start
async def start():
    cl.user_session.set("counter", 0)
    cl.user_session.set("query_engine_bm25_s", query_engine_router(index, "bm25_spec"))
    cl.user_session.set("query_engine_hybrid", query_engine_router(index, "hybrid"))

    message_history = []
    cl.user_session.set("message_history", message_history)

    await cl.Message(
        author="SuperGPT", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


async def set_sources(response, response_message):
    label_list = []
    count = 1

    above_zero_nodes = []
    for sr in response.source_nodes:
        # print(sr.node.text)
        print(sr.get_score())
        if sr.get_score() > 0.03:
            above_zero_nodes.append(sr)

    pdfs = set()
    elements = []
    end = len(nodes) if len(nodes) < 4 else 4
    for sr in above_zero_nodes[:end]:
        node = chroma_collection.get(ids = sr.id_)
        file_name = str(node["metadatas"][0]["file_name"])
        pdfs.add(file_name)
        print("-" * 10, sr)
        elements.append(
            cl.Text(
                name=file_name.split('.')[0] + ".chunk_" + str(count),
                content=f"{sr.node.text}",
                display="side",
                size="small",
            )
        )
        label_list.append(file_name.split('.')[0] + ".chunk_" + str(count))
        count += 1

    count = 1
    
    for pdf in pdfs:
        elements.append(cl.Pdf(name=pdf, display="page", path=f"./SOURCE_DOCUMENTS/{pdf}"))
        count += 1
    response_message.elements = elements

    if len(above_zero_nodes) > 0:
        response_message.content += "\n\nSources: " + ", ".join(label_list)
        response_message.content += "\nSource pdfs: " + ", ".join(pdfs)
    await response_message.update()


@cl.on_message
async def main(user_message: cl.Message):
    n_history_messages = 15
    counter = cl.user_session.get("counter")

    message_history = cl.user_session.get("message_history")
    prompt_template = ""

    for past_message in message_history:
        prompt_template += CONVERSATION_PROMPT.format(
            role=past_message['author'],
            content=past_message['content'],
        )

    prompt_template += CONVERSATION_PROMPT.format(
        role="user",
        content=user_message.content,
    ) 
    
    prompt_template += RESPONSE_PROMPT.format(
        role='SuperGPT',
    )

    user_query = get_query(prompt_template, '<|end_header_id|>', '<|eot_id|>')
    user_query = user_query[counter][2:]
    user_query = user_query.lower()
    print(user_query)
    query_words = user_query.split(' ')
    product_prefix = ['ars-', 'as-', 'asg-', 'ssg-', 'sys-']
    skus = ['4u', '1u', '2u', '5u', '8u', '10u']

    query_engine = None
    if any(prefix in user_query for prefix in product_prefix) or any(sku in query_words for sku in skus):
        print("bm25s")
        query_engine = cl.user_session.get("query_engine_bm25_s") # 4 docs
    else:
        print("hybrid")
        query_engine = cl.user_session.get("query_engine_hybrid") # 10 docs

    response = await cl.make_async(query_engine.query)(prompt_template)
    
    assistant_message = cl.Message(content="", author="SuperGPT")
    for token in response.response_gen:
        await assistant_message.stream_token(token)
    if response.response_txt:
        assistant_message.content = response.response_txt
    await assistant_message.send()

    message_history.append({"author": "user", "content": user_message.content})
    message_history.append({"author": "SuperGPT", "content": assistant_message.content})
    message_history = message_history[-n_history_messages:]
    cl.user_session.set("message_history", message_history)

    if response.source_nodes:
        await set_sources(response, assistant_message)

    counter = counter+1 if counter != 15 else 0
    cl.user_session.set("counter", counter)