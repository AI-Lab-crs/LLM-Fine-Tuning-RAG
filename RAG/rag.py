import streamlit as st 
from MyFunctions import MySpecialFunctions
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.storage import InMemoryByteStore

from htmlTemplates import css, bot_template, user_template
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# create an instance of my class
special_functions_instance = MySpecialFunctions()

store = LocalFileStore("./embeddings_cache")
# store = InMemoryByteStore()

def get_text(external_data ):
    return special_functions_instance.get_file_text(external_data )

def get_pdf_text(external_data ):
    return special_functions_instance.get_pdf_text(external_data )

def get_chunks(text):
    return special_functions_instance.get_text_chunks(text)

def get_HuggingFaceEmbeddings():
    return special_functions_instance.get_HuggingFaceEmbeddings()

def get_OpenAIEmbeddings():
    return special_functions_instance.get_OpenAIEmbeddings()

def get_open_embedding():
    return special_functions_instance.get_open_embedding()


def get_cached_embedder():
    # create an embedder
    embedder = special_functions_instance.get_OpenAIEmbeddings()
    cache_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedder,
        store,
        namespace = embedder.model
           )
    return cache_embedder

def get_faiss_vectorstore(text_chunks, embedding_method):
    return special_functions_instance.get_faiss_vectorstore(text_chunks, embedding_method)

def get_chroma_vectorstore(text_chunks, embedding_method):
    return special_functions_instance.get_chroma_vectorstore(text_chunks, embedding_method)

def get_weaviate_vectorstore(text_chunks, embedding_method):
    return special_functions_instance.get_weaviate_vectorstore(text_chunks, embedding_method)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # Initialization of conversation memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)  
    # Initialization of conversation -- converse with the vector store
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


# def get_generate(vectorstore):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     generate_chain = ConversationalRetrievalChain.from_llm(
#         llm = llm,
#         retriever= vectorstore.as_retriever(),
#         memory = memory
#     )
#     return generate_chain




    

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)









def main():
    load_dotenv()
    st.set_page_config(page_title="Generation of Telegram Ads in Amharic", page_icon= ":smile")
    
    with st.sidebar:
        #st.markdown("# Menu")
        pass

    st.markdown("## Ads Generation in Amharic")
    external_data = st.file_uploader(" Upload the generative text from fine-tuning", accept_multiple_files= True)
    if st.button("Retrieval"):
        with st.spinner("Processing"):
            # get files text
            #text = get_text(external_data )
            text = get_pdf_text(external_data )

            # get_text_chunks(text)
            text_chunks = get_chunks(text)

            # get embedding function or embedder
            openai_embedder = get_OpenAIEmbeddings() 
            cache_embedder = get_cached_embedder()
            open_embedder = get_open_embedding()         # Has an issue.    

            # get_faiss_vectorstore 
            #faiss_vectorstore_db = get_faiss_vectorstore(text_chunks, cache_embedder)
    
            # get_chroma_vectorstore
            #chroma_vectorstore_db = get_chroma_vectorstore(text_chunks, openai_embedder)
            chroma_vectorstore_db = get_chroma_vectorstore(text_chunks, cache_embedder)
            # c = chroma_vectorstore_db.similarity_search("What is the best sport?", k=4)
            # st.write(c)

            st.session_state.conversation= get_conversation_chain(chroma_vectorstore_db)
            

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat")
    user_question = st.text_input("")
    if user_question:
        handle_userinput(user_question)
            
           


if __name__ == "__main__":
    main()