import nest_asyncio
nest_asyncio.apply()

import os, re
import chainlit as cl
import pickle
import chromadb
from utils import clean_text

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
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

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n{message.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


LLM = OpenAILike(
        model="llama3-8b-instruct",
        api_base="http://vllm:8000/v1", 
        api_key="fake",
        temperature=0.0,
        max_tokens=512,
        is_chat_model=True,
        messages_to_prompt=messages_to_prompt,
)

CONTEXT_PROMPT = """\
You're a Supermicro product expert, and you will answer questions based on the provided documents. 
To answer questions accurately, you may need to extract information in tables, figures, diagrams.  Do not use any other information to answer the question and provide a detailed answer to the question, highlighting Supermicro and its products' power savings and green computing benefits.
Firstly, highlight GPU characteristics and GPU products. Secondly, mention power-related features, such as fan cooling or liquid cooling, power consumption, etc. Thirdly, try to mention green computing benefits.
If you don't know the answer, please say "Sorry, I haven't been trained with that data yet."

Context information is below.
---------------------
{context_str}
---------------------
Please provide url when applicable if you are answering the question from the "FAQ" document.
\
"""

EMBEDDING = InstructorEmbedding(
  model_name="hkunlp/instructor-xl",
  query_instruction='Represent the question for retrieving supporting documents: ',
  text_instruction='Represent the document for retrieval: ',
)
DB_PATH = "./chroma_db_v1"
DB_PATH_DS = "./chroma_db_v1_ds"
DB_PATH_FAQ = "./csvtest"
chroma_client = chromadb.PersistentClient(DB_PATH)
chroma_collection = chroma_client.get_collection("test_v1")
VECTOR_STORE = ChromaVectorStore(chroma_collection=chroma_collection)

chroma_client2 = chromadb.PersistentClient(DB_PATH_FAQ)
chroma_collection2 = chroma_client2.get_collection("csv")
VECTOR_STORE_FAQ = ChromaVectorStore(chroma_collection=chroma_collection2)

Settings.llm = LLM
Settings.embed_model = EMBEDDING
Settings.num_output = 512
Settings.context_window = 8192

# for vector search
index = VectorStoreIndex.from_vector_store(
    vector_store=VECTOR_STORE,
    embed_model=EMBEDDING,
)
index_faq = VectorStoreIndex.from_vector_store(
    vector_store=VECTOR_STORE_FAQ,
    embed_model=EMBEDDING,
)

# for bm25 search
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
    rankergpt = RankGPTRerank(
            llm=LLM,
            top_n=10,
            verbose=True,
        )
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=8192
    )
    query_engine = None
    if type_ == "hybrid":
        vector_retriever = index.as_retriever(similarity_top_k=7)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever) # 15 docs

        query_engine = ContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            #node_postprocessors=[reranker],
            node_postprocessors=[rankergpt],
            llm=LLM,
            memory=memory,
            context_template=CONTEXT_PROMPT,
        )
    elif type_ == "vector":
        vector_retriever = index.as_retriever(similarity_top_k=5)
        query_engine = ContextChatEngine.from_defaults(
            retriever=vector_retriever,
            node_postprocessors=[rankergpt],
            llm=LLM,
            memory=memory,
            context_template=CONTEXT_PROMPT,
        )
    elif type_ == "bm25_spec":
        query_engine = ContextChatEngine.from_defaults(
            retriever=bm25_retriever_ds,
            llm=LLM,
            memory=memory,
            context_template=CONTEXT_PROMPT,
        )
    else:
        query_engine = SimpleChatEngine.from_defaults(
            llm=LLM,
            memory=memory,
        )

    return query_engine

@cl.on_chat_start
async def start():
    cl.user_session.set("sources", [])
    cl.user_session.set("query_engine_bm25_s", query_engine_router(index, "bm25_spec"))
    cl.user_session.set("query_engine_hybrid", query_engine_router(index, "hybrid"))
    cl.user_session.set("query_engine_vector_faq", query_engine_router(index_faq, "vector"))
    cl.user_session.set("query_engine_simple", query_engine_router(index, None))
    cl.user_session.set("message_history", [])

    await cl.Message(
        author="SuperGPT", content="Hello! How may I help you?"
    ).send()


async def set_sources(source_nodes, response_message):
    label_list = []
    count = 1

    above_zero_nodes = []
    for sr in source_nodes:
        print(sr.get_score())
        #print(sr.node.text)
        if sr.get_score() > 0.03:
            above_zero_nodes.append(sr)

    pdfs = set()
    elements = []

    end = len(above_zero_nodes) if len(above_zero_nodes) < 4 else 4
    print(len(above_zero_nodes))
    if len(above_zero_nodes) > 0:
        for sr in above_zero_nodes[:end]:
            if chroma_collection.get(ids = sr.id_)['ids']:
                node = chroma_collection.get(ids = sr.id_)
            else:
                node = chroma_collection2.get(ids = sr.id_)
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
            if os.path.exists(f"./SOURCE_DOCUMENTS/{pdf}"):
                elements.append(cl.Pdf(name=pdf, display="page", path=f"./SOURCE_DOCUMENTS/{pdf}"))
            count += 1
        response_message.elements = elements

        response_message.content += "\n\nSources: " + ", ".join(label_list)
        response_message.content += "\nSource pdfs: " + ", ".join(pdfs)
    await response_message.update()


@cl.on_message
async def main(user_message: cl.Message):
    n_history_messages = 12
    message_history = cl.user_session.get("message_history")
    chat_history = [
        ChatMessage(
            role=past_message['author'],
            content=past_message['content'],
        )
        for past_message in message_history 
    ]
    query_engine = None
    if len(message_history) == 0:
        query_text = user_message.content.lower()
        print(query_text)
        query_words = query_text.split(' ')
        product_prefix = ['ars-', 'as-', 'asg-', 'ssg-', 'sys-']
        skus = ['4u', '1u', '2u', '5u', '8u', '10u']
        if any(prefix in query_text for prefix in product_prefix) or any(sku in query_words for sku in skus):
            print("bm25s")
            query_engine = cl.user_session.get("query_engine_bm25_s") # 10 docs
        else:
            retriever = index_faq.as_retriever(similarity_top_k=5)
            res = retriever.retrieve(query_text)
            if res[0].score > 0.7:
                print("faq")
                query_engine = cl.user_session.get("query_engine_vector_faq")
            else:
                print("hybrid")
                query_engine = cl.user_session.get("query_engine_hybrid") # 10 docs
    else:
        query_engine = cl.user_session.get("query_engine_simple")

    response = await cl.make_async(query_engine.stream_chat)(
        message=user_message.content,
        chat_history=chat_history,
    )

    assistant_message = cl.Message(content="", author="SuperGPT")
    for token in response.response_gen:
        await assistant_message.stream_token(token)
    await assistant_message.send()

    if hasattr(response, 'sources') and len(response.sources) > 0:
        message_history.append({"author": "system", "content": response.sources[0].content})
    message_history.append({"author": "user", "content": user_message.content})
    message_history.append({"author": "assistant", "content": assistant_message.content})
    if len(message_history) > n_history_messages:
        message_history = [message_history[0]] + message_history[-n_history_messages:]

    cl.user_session.set("message_history", message_history)

    sources = cl.user_session.get("sources")
    if len(sources) == 0 and hasattr(response, 'source_nodes'):
        await set_sources(response.source_nodes, assistant_message)
        cl.user_session.set("sources", response.source_nodes)
    else:
        await set_sources(sources, assistant_message)