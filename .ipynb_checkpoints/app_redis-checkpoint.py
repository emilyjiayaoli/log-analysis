# from openai import OpenAI
import streamlit as st

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
# from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms.openai import OpenAI

from llama_index.core import Settings

# NOTE: Reads the documents from the "data" directory

st.title("Documentation Assistant")

# File uploader
uploaded_files = st.file_uploader("Upload Documentation Files", type=["txt", "pdf", "docx", "py"], accept_multiple_files=True)

# GitHub link input
github_link = st.text_input("Enter GitHub Repository Link")

# only if index is not already created
if "index" not in st.session_state:
    # check if storage already exists
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        print("Creating new index...")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        print("Loading existing index")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    st.session_state.index = index


# Settings.llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=4096)
# # OpenAI(temperature=0.2, model="gpt-4")
Settings.llm = OpenAI(temperature=0.2, model="gpt-4o")
query_engine = st.session_state.index.as_query_engine(streaming=True)


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is this documentation about? Explain in detail?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    # with st.chat_message("assistant"):
    #     response_placeholder = st.empty()
    #     raw_response = query_engine.query(prompt)
        
    #     # print out the context retrieved from the index

    #     for text in raw_response.response_gen:
    #         response += text
    #         response_placeholder.markdown(response.replace("\n", "  \n"))


    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        context_placeholder = st.empty()
        
        response_gen = query_engine.query(prompt)
        
        for text in response_gen.response_gen:
            response += text
            response_placeholder.markdown(response.replace("\n", "  \n"))
        
        # Display retrieved context
        if response_gen.source_nodes:
            context_str = "\n\n".join([node.get_content() for node in response_gen.source_nodes]).replace("\n", "  \n")
            # context_str = response_gen.source_nodes[0].get_content().replace("\n", "  \n")
            context_placeholder.markdown(f"**Retrieved Context:**{context_str}")


    st.session_state.messages.append({"role": "assistant", "content": response})