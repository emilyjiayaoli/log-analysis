# from openai import OpenAI
import streamlit as st

import os.path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
# NOTE: Reads the documents from the "data" directory
from redisvl.schema import IndexSchema

from llama_index.core import VectorStoreIndex # index
from llama_index.core import SimpleDirectoryReader # data loader

# Output parser
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.llms.openai import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

import ast

############################################
## if creating a new container
# !docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
## # if starting an existing container
## !docker start -a redis-stack
############################################

openai_api_key = os.getenv('OPENAI_API_KEY')
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')

def set_up(clear_cache=False, data_path="data"):
    
    # Set up Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # e: Define redis schema
    custom_schema = IndexSchema.from_dict(
        {
            "index": {"name": "gdrive", "prefix": "doc"},
            # customize fields that are indexed
            "fields": [
                # required fields for llamaindex
                {"type": "tag", "name": "id"},
                {"type": "tag", "name": "doc_id"},
                {"type": "text", "name": "text"},
                # custom vector field for bge-small-en-v1.5 embeddings
                {
                    "type": "vector",
                    "name": "vector",
                    "attrs": {
                        "dims": 384,
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
            ],
        }
    )
    st.write("Connecting to Redis...", f"redis://{redis_host}:{redis_port}")
    # e: define vector store given schema
    vector_store = RedisVectorStore(
        schema=custom_schema,
        redis_url=f"redis://{redis_host}:{redis_port}",
    )
    # Optional: clear vector store if exists & clear_cache is True
    if vector_store.index_exists():
        if clear_cache:
            st.write("Clearing redis cache...")
            vector_store.delete_index()
    else:
        vector_store.create_index()

    # Set up the ingestion cache layer
    cache = IngestionCache(
        cache=RedisCache.from_host_and_port(redis_host, redis_port),
        collection="redis_cache",
    )
    docstore = RedisDocumentStore.from_host_and_port(
        redis_host, redis_port, namespace="document_store"
    )
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embed_model,
        ],
        docstore=docstore,
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    index = VectorStoreIndex.from_vector_store(
        pipeline.vector_store, embed_model=embed_model
    )
    loader = SimpleDirectoryReader(data_path)

    return pipeline, index, loader, embed_model, vector_store, cache, docstore


def load_data(loader):
    docs = loader.load_data()
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs

def get_output_parser():
    """
    Define the output parser for llm
    """
    # define output schema
    response_schemas = [
        ResponseSchema(
            name="Python code",
            description="Write python code that correspond to the query",
        ),
        ResponseSchema(
            name="Explanation",
            description="Describe what you've done",
        ),
    ]

    # define output parser
    lc_output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas
    )
    output_parser = LangchainOutputParser(lc_output_parser)
    return output_parser


# Set up the Streamlit app
st.title("Docs Assistant")

# # File uploader
# uploaded_files = st.file_uploader("Upload Documentation Files", type=["txt", "pdf", "docx", "py"], accept_multiple_files=True)

# # GitHub link input
# github_link = st.text_input("Enter GitHub Repository Link")

# toggle
clear_cache = st.checkbox("Clear cache?", value=False)
data_path = st.text_input("Path to local data source directory", value="data")
if not os.path.exists(data_path):
    st.error("Data directory does not exist. Please provide a valid path.")
    st.stop()

# Refresh button
if st.button("Sync data changes"):
    st.session_state.clear()

st.session_state.set_up_complete = False
with st.status("Initializing", expanded=True) as status:

    # Initialize the pipeline, index, loader, and query engine if not already in session state
    if not all(key in st.session_state for key in ["pipeline", "loader", "index", "query_engine"]) or clear_cache:
            st.write("Initializing pipeline, index, loader, and query engine...")
            pipeline, index, loader, embed_model, vector_store, cache, docstore = set_up(clear_cache=clear_cache, 
                                                                                        data_path=data_path)
            st.session_state.pipeline = pipeline
            st.session_state.index = index
            st.session_state.loader = loader
            st.session_state.embed_model = embed_model
            st.session_state.vector_store = vector_store
            st.session_state.cache = cache
            st.session_state.docstore = docstore

            output_parser = get_output_parser()
            llm = OpenAI(temperature=0.1, model="gpt-4o", output_parser= output_parser)
            query_engine = index.as_query_engine(llm=llm) #, streaming=True)

            st.session_state.query_engine = query_engine

    st.write("Loading data...")
    docs = load_data(loader=st.session_state.loader)
    st.write(f"--> Loaded {len(docs)} documents")
    st.write("Ingesting data...")
    nodes = st.session_state.pipeline.run(documents=docs)
    st.write(f"--> Ingested {len(nodes)} new nodes")

    status.update(label="Pipeline ready!")
    st.session_state.set_up_complete = True



with st.expander("Data Ingestion Details", expanded=False):
    st.write("Pipeline:", st.session_state.pipeline)
    st.write("Index:", st.session_state.index)
    st.write("Loader:", st.session_state.loader)
    st.write("Embedding Model:", st.session_state.embed_model)
    st.write("Vector Store:", st.session_state.pipeline.vector_store)
    st.write("Cache:", st.session_state.pipeline.cache)
    st.write("Docstore:", st.session_state.pipeline.docstore)
    st.write("Query Engine:", st.session_state.query_engine)

with st.expander("Preview Data", expanded=False):
    for doc in docs[:50]:
        st.write(f"**{doc.metadata['file_name']}**")
        # with st.expander(f"**{doc.metadata['file_name']}**", expanded=False):
        st.write(vars(doc)["text"])
        break


with st.expander("Prompt Editor", expanded=False):
    # Text area for inputting the prompt
    prompt = st.text_area("Enter your prompt here:")

    # Function to save the prompt to a text file
    def save_prompt(prompt_text, filename="prompt.txt"):
        with open(filename, "w") as file:
            file.write(prompt_text)

    # Save button
    if st.button("Save Prompt"):
        save_prompt(prompt)
        st.success("Prompt saved successfully!")

    # Provide the file for download
    if st.button("Download Prompt"):
        st.download_button(
            label="Download Prompt",
            data=prompt,
            file_name="prompt.txt",
            mime="text/plain"
        )

    print("Prompt:", prompt)

if prompt := st.chat_input("What is this documentation about? Explain in detail?"):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.session_state.query_engine.query(prompt)
        # st.markdown(response)

    # Assuming response_obj is your Response object
    response_str = response.response

    # Parse the string representation of the dictionary
    response_dict = ast.literal_eval(response_str)

    # Extract 'Python code' and 'Explanation'
    python_code = response_dict.get('Python code')
    explanation = response_dict.get('Explanation')

    st.markdown(f"**Python Code:**\n```python\n{python_code}\n```")
    st.markdown(f"**Explanation:**\n{explanation}")
    # print("Python Code:\n", python_code)
    # print("\nExplanation:\n", explanation)



    # response = ""

    # with st.chat_message("assistant"):
    #     response_placeholder = st.empty()
    #     context_placeholder = st.empty()
        
    #     response_gen = st.session_state.query_engine.query(prompt)
        
    #     for text in response_gen.response_gen:
    #         response += text
    #         response_placeholder.markdown(response.replace("\n", "  \n"))
        
    #     # Display retrieved context
    #     if response_gen.source_nodes:
    #         context_str = "\n\n".join([node.get_content() for node in response_gen.source_nodes]).replace("\n", "  \n")
    #         # context_str = response_gen.source_nodes[0].get_content().replace("\n", "  \n")
    #         context_placeholder.markdown(f"**Retrieved Context:**{context_str}")


    # st.session_state.messages.append({"role": "assistant", "content": response})
