import os, re
import chainlit as cl
import pickle
from typing import List

from llama_index.core import Settings, VectorStoreIndex,SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

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

# openai.api_key = os.environ.get("OPENAI_API_KEY")
# pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# MODEL = os.getenv("MODEL", "gpt-4-0125-preview")
# EMBEDDING = os.getenv("EMBEDDING", "text-embedding-3-large")
SYS_PROMPT = PromptTemplate(
    """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

As a helpful assistant, you will utilize the provided document to answer user questions. 
Read the given document before providing answers and think step by step. 
The document has an order of paragraphs with a higher correlation to the questions from the top to the bottom. 
The answer may be hidden in the tables, so please find it as closely as possible. 
Do not use any other information to answer the user. Provide a etailed answer to the question.
Also, please provide the answer in the following order of priorities if applicable:
Firstly, emphasize GPU characteristics and GPU products.
Secondly, Give prominence to power-related specifications such as fan cooling or liquid cooling, power consumption, and so on.
Thirdly, If applicable, mention green computing.
Remember, please don't provide any fabricated information, ensuring that everything stated is accurate and true.

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
EMBEDDING = os.getenv("EMBEDDING", "hkunlp/instructor-xl")

with open("nodes.pickle", 'rb') as f:
    global nodes
    nodes = pickle.load(f)


def load_chroma_vector_store(path):
    chroma_client = chromadb.PersistentClient(path)
    chroma_collection = chroma_client.get_collection("test")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store


def clean_text(text: str) -> List[str]:
        
        # Convert text to lowercase
        text = text.lower()
        
        # Remove stopwords from text using regex
        stopwords_list = set(nltk.corpus.stopwords.words('english'))
        stopwords_pattern = r'\b(?:{})\b'.format('|'.join(stopwords_list))
        text = re.sub(stopwords_pattern, '', text)

        # Replace punctuation, newline, tab with space
        text = re.sub(r'[,.!?|]|[\n\t]', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # remove # from markdown
        text = re.sub(r'#+', '', text)
        text = re.sub(r'<[^>]*>', '', text)

        # remove from user query
        text = re.sub(r'assistant', '', text)
        text = re.sub(r'user', '', text)
        
        text = text.strip().split(" ")
        text = [t for t in text if t != "-" and t != ""]
        return text


def build_router_retriever_query_engine(index):
    vector_retriever = VectorIndexRetriever(index)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

    retriever_tools = [
        RetrieverTool.from_defaults(
            retriever=vector_retriever,
            description="Useful in most cases",
        ),
        RetrieverTool.from_defaults(
            retriever=bm25_retriever,
            description="Useful if searching about specific information",
        ),
    ]
    router_retriever = RouterRetriever.from_defaults(
        retriever_tools=retriever_tools,
        llm=LLM,
        select_multi=True,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=router_retriever,
        streaming=True,
        text_qa_template=SYS_PROMPT,
    )
    return query_engine


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
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, 
                                                tokenizer=clean_text,
                                                similarity_top_k=4)

    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    # step_decompose_transform = StepDecomposeQueryTransform(llm=LLM, verbose=True)
    reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

    query_engine = RetrieverQueryEngine.from_args(
        retriever=bm25_retriever,
        # node_postprocessors=[reranker],
        # llm=LLM,
        streaming=True,
        # query_transform=step_decompose_transform,
        text_qa_template=SYS_PROMPT,
    )

    return query_engine

@cl.cache
def load_context():
    Settings.llm = LLM
    Settings.embed_model = InstructorEmbedding(model_name="hkunlp/instructor-xl")
    Settings.num_output = 2048
    Settings.context_window = 8192
    
    vector_store = load_chroma_vector_store("./chroma_db")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    return index


@cl.on_chat_start
async def start():
    index = load_context()

    query_engine1 = build_router_retriever_query_engine(index)
    query_engine2 = hybrid_retriver_reranking_query_engine(index)
    
    cl.user_session.set("query_engine", query_engine2)

    message_history = []
    cl.user_session.set("message_history", message_history)

    await cl.Message(
        author="assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


async def set_sources(response, response_message):
    label_list = []
    count = 1

    above_zero_nodes = []
    for sr in response.source_nodes:
        if sr.get_score() > 0:
            above_zero_nodes.append(sr)

    for sr in above_zero_nodes:
        chroma_client = chromadb.PersistentClient("./chroma_db")
        chroma_collection = chroma_client.get_collection("test")
        node = chroma_collection.get(ids = sr.id_)
        print("-" * 10, sr)
        elements = [
            cl.Text(
                name=str(node["metadatas"][0]["file_name"]) + str(count),
                content=f"{sr.node.text}",
                display="side",
                size="small",
            )
        ]
        response_message.elements = elements
        label_list.append(str(node["metadatas"][0]["file_name"])+ str(count))
        await response_message.update()
        count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    await response_message.update()


@cl.on_message
async def main(user_message: cl.Message):
    n_history_messages = 4
    query_engine = cl.user_session.get("query_engine")
    message_history = cl.user_session.get("message_history")
    prompt_template = ""
    print(message_history)
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
        role='assistant',
    ) 

    response = await cl.make_async(query_engine.query)(prompt_template)
    
    assistant_message = cl.Message(content="", author="assistant")
    for token in response.response_gen:
        await assistant_message.stream_token(token)
    if response.response_txt:
        assistant_message.content = response.response_txt
    await assistant_message.send()

    message_history.append({"author": "user", "content": user_message.content})
    message_history.append({"author": "assistant", "content": assistant_message.content})
    message_history = message_history[-n_history_messages:]
    cl.user_session.set("message_history", message_history)

    if response.source_nodes:
        await set_sources(response, assistant_message)