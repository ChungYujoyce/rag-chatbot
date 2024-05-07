import os, re
import chainlit as cl
import pickle
from typing import List

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
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

import argparse

parser = argparse.ArgumentParser(description='Parse command-line arguments')
parser.add_argument('--port', type=int, help='Port number')
parser.add_argument('-w', help='w')
parser.add_argument('run', help='run')
args = parser.parse_args()

from chromadb.config import Settings
import chromadb
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


LLM = OpenAILike(
        model="llama3-8b-instruct", 
        api_base="http://vllm:8000/v1", 
        api_key="fake",
        temperature=0.0,
        max_tokens=256,
    )


SYS_PROMPT = PromptTemplate(
    """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You're a Supermicro product expert, and you will answer questions based on the provided documents. 
To answer questions accurately, you may need to extract information in tables, figures, diagrams.  Do not use any other information to answer the question and provide a detailed answer to the question, trying to highlight Supermicro products' power savings and green computing benefits.
Firstly, highlight GPU characteristics and GPU products. Secondly, mention power-related features, such as fan cooling or liquid cooling, power consumption, etc. Thirdly, try to mention green computing benefits.
If you don't know the answer, please say "Sorry, I haven't been trained with that data yet."  Please do not provide any fabricated information.

Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.<|eot_id|>
{query_str}\
"""
)
CONVERSATION_PROMPT = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
RESPONSE_PROMPT = "<|start_header_id|>{role}<|end_header_id|>\n\n"
EMBEDDING = InstructorEmbedding(model_name="hkunlp/instructor-xl")
DB_PATH = "./chroma_db_v1"
chroma_client = chromadb.PersistentClient(DB_PATH)
chroma_collection = chroma_client.get_collection("test_v1")
VECTOR_STORE = ChromaVectorStore(chroma_collection=chroma_collection)

with open(f"./{DB_PATH}/nodes.pickle", 'rb') as f:
    global nodes
    nodes = pickle.load(f)

def clean_text(text: str) -> List[str]:
    text = text.lower()
    
    # Remove stopwords from text using regex
    stopwords_list = set(nltk.corpus.stopwords.words('english'))
    stopwords_pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
    text = re.sub(stopwords_pattern, '', text)

    # Replace punctuation, newline, tab with space
    text = re.sub(r'[,.!?|]|[\n\t\'\\]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # remove # from markdown
    text = re.sub(r'#+', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    # remove leading and trailing non-alphabet characters
    text = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', text)

    # remove from user query
    text = re.sub(r'supergpt', '', text)
    text = re.sub(r'user', '', text)
    
    text = text.strip().split(" ")
    text = [t for t in text if t != "-" and t != ""]
    return list(set(text))


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


def hybrid_retriver_reranking_query_engine(index):
    vector_retriever = index.as_retriever(similarity_top_k=5)

    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, 
                                                tokenizer=clean_text,
                                                similarity_top_k=10)
    bm25_retriever5 = BM25Retriever.from_defaults(nodes=nodes,
                                                tokenizer=clean_text,
                                                similarity_top_k=5)
    bm25_retriever4 = BM25Retriever.from_defaults(nodes=nodes,
                                                tokenizer=clean_text,
                                                similarity_top_k=4)

    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever5) # 10 docs
    step_decompose_transform = StepDecomposeQueryTransform(llm=LLM, verbose=True)
    reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")
    rankergpt = RankGPTRerank(
            llm=LLM,
            top_n=4,
            verbose=True,
        )

    query_engine = None
    if args.port == 4100: # hybrid
        print("hybrid")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[reranker],
            llm=LLM,
            streaming=True,
            #query_transform=step_decompose_transform,
            text_qa_template=SYS_PROMPT,
        )
    elif args.port == 4101: # hybrid
        print("bm25")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[rankergpt],
            llm=LLM,
            streaming=True,
            #query_transform=step_decompose_transform,
            text_qa_template=SYS_PROMPT,
        )
    elif args.port == 4102: # bm25
        query_engine = RetrieverQueryEngine.from_args(
            retriever=bm25_retriever4,
            #node_postprocessors=[reranker],
            llm=LLM,
            streaming=True,
            #query_transform=step_decompose_transform,
            text_qa_template=SYS_PROMPT,
        )

    return query_engine


@cl.cache
def load_context():
    Settings.llm = LLM
    Settings.embed_model = EMBEDDING
    Settings.num_output = 4096
    Settings.context_window = 8192

    index = VectorStoreIndex.from_vector_store(
        vector_store=VECTOR_STORE,
        embed_model=EMBEDDING,
    )
    return index


@cl.on_chat_start
async def start():
    index = load_context()
    query_engine = hybrid_retriver_reranking_query_engine(index)

    cl.user_session.set("query_engine", query_engine)

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
        if sr.get_score() > 0.03:
            above_zero_nodes.append(sr)

    pdfs = set()
    elements = []
    for sr in above_zero_nodes:
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
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    response_message.content += "\nSource pdfs: " + ", ".join(pdfs)
    await response_message.update()


@cl.on_message
async def main(user_message: cl.Message):
    index = load_context()
    n_history_messages = 8
    query_engine = cl.user_session.get("query_engine")
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

    response = await cl.make_async(query_engine.query)(prompt_template)
    
    assistant_message = cl.Message(content="", author="SuperGPT")
    for token in response.response_gen:
        await assistant_message.stream_token(token)
    if response.response_txt:
        assistant_message.content = response.response_txt
    await assistant_message.send()

    message_history.append({"author": "user", "content": user_message.content})
    #message_history.append({"author": "SuperGPT", "content": assistant_message.content})
    message_history = message_history[-n_history_messages:]
    cl.user_session.set("message_history", message_history)

    if response.source_nodes:
        await set_sources(response, assistant_message)
        first_search = False