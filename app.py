import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from msgTemplates import css, bot_template, user_template

def extract_pdf_text(pdf_documents):
    """
    Extract text from PDF documents.

    Args:
        pdf_documents (list): List with the uploaded files.

    Returns:
        str: Concatenated text extracted from all the PDF documents.
    """
    text = ""
    for document in pdf_documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_text_into_chunks(text):
    """
    Split text into smaller chunks.

    Args:
        text (str): Input text to be split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):
    """
    Create a vector store from text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        VectorStore: Vector store created from the text chunks.
    """
    embeddings_model = OpenAIEmbeddings()
    # embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings_model)
    return vector_store


def generate_conversation_chain(vector_store):
    """
    Generate a conversation chain using a vector store.

    Args:
        vector_store (VectorStore): Vector store containing embeddings of text chunks.

    Returns:
        ConversationalRetrievalChain: Chain generated using the vector store.
    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory( # Memory to store chat history
        memory_key='chat_history',
        return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm( # Chain to handle conversation
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def process_user_input(user_query):
    """
    Process user input and generate responses.

    Args:
        user_query (str): User input.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': user_query}) # Generate response
    st.session_state.chat_history = response['chat_history'] # Store chat history

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # Display user message
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Display bot message
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv() # Load environment variables

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")

    user_query = st.text_input("Ask a question about your PDFs:")
    if user_query:
        process_user_input(user_query)

    with st.sidebar:
        st.subheader("Your Documents")

        uploaded_pdfs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract PDF text
                raw_text = extract_pdf_text(uploaded_pdfs)

                # Split text into chunks
                text_chunks = split_text_into_chunks(raw_text)

                # Create vector store
                vector_store = create_vector_store(text_chunks)

                # Create conversation chain
                st.session_state.conversation = generate_conversation_chain(vector_store)

if __name__ == '__main__':
    main()