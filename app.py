import streamlit as st
import pyperclip  # For copying text to clipboard
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
import pandas as pd  # Required for handling CSV files

import psutil

# CPU usage information
cpu_percent = psutil.cpu_percent(interval=1)  # in percentage
cpu_count = psutil.cpu_count(logical=False)  # Physical cores
cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores

# Memory usage information
memory = psutil.virtual_memory()
total_memory = memory.total / (1024 ** 3)  # Convert to GB
used_memory = memory.used / (1024 ** 3)  # Convert to GB
free_memory = memory.available / (1024 ** 3)  # Convert to GB
memory_percent = memory.percent  # Percentage usage

# Disk usage information
disk = psutil.disk_usage('/')
total_disk = disk.total / (1024 ** 3)  # Convert to GB
used_disk = disk.used / (1024 ** 3)  # Convert to GB
free_disk = disk.free / (1024 ** 3)  # Convert to GB
disk_percent = disk.percent  # Percentage usage

# Network usage information
network = psutil.net_io_counters()
bytes_sent = network.bytes_sent / (1024 ** 2)  # Convert to MB
bytes_recv = network.bytes_recv / (1024 ** 2)  # Convert to MB

# Output the information
print("CPU Usage: {}%".format(cpu_percent))
print("Physical CPU Cores: {}".format(cpu_count))
print("Logical CPU Cores: {}".format(cpu_count_logical))
print("Total Memory: {:.2f} GB".format(total_memory))
print("Used Memory: {:.2f} GB".format(used_memory))
print("Free Memory: {:.2f} GB".format(free_memory))
print("Memory Usage: {}%".format(memory_percent))
print("Total Disk: {:.2f} GB".format(total_disk))
print("Used Disk: {:.2f} GB".format(used_disk))
print("Free Disk: {:.2f} GB".format(free_disk))
print("Disk Usage: {}%".format(disk_percent))
print("Network Sent: {:.2f} MB".format(bytes_sent))
print("Network Received: {:.2f} MB".format(bytes_recv))



# Load environment variables
load_dotenv()

# Retrieve API keys
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API keys are set
if not groq_api_key or not openai_api_key:
    st.error("API keys are not set. Please check your .env file.")
    st.stop()

# Initialize Streamlit page
st.set_page_config(page_title="EDA Assistant", layout="wide")

# Custom CSS for Sleek Design
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 14px;
        margin: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .response-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333333;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .follow-up-question {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #e3f2fd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .copy-button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
    }
    .copy-button:hover {
        background-color: #1565c0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Settings
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Select Model:", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768", "gemma2-9b-it"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 8000, 3000)
retrieve_mode = st.sidebar.selectbox("Retrieve Mode:", ["Text (Hybrid)", "Vector Only", "Text Only"])

# Document upload
st.header("EDA Assistant")
uploaded_files = st.file_uploader("Upload PDF(s) or CSV(s):", type=["pdf", "csv"], accept_multiple_files=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Document processing
vector_store = None
if uploaded_files:
    st.subheader("Processing Documents...")
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                # Process PDF
                pdf_reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in pdf_reader.pages])

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)

            elif uploaded_file.type == "text/csv":
                # Process CSV
                csv_data = pd.read_csv(uploaded_file)
                text = csv_data.to_string(index=False)  # Convert the CSV content to a string

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)

            # Embed chunks into vector store
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            if vector_store is None:
                vector_store = FAISS.from_texts(chunks, embeddings)
            else:
                temp_vector_store = FAISS.from_texts(chunks, embeddings)
                vector_store.merge_from(temp_vector_store)

            st.success(f"Processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Ask a question
st.header("Ask your Assistant")
predefined_questions = [
    "How do disposable and capital item sales vary by region for the top 15 products?",
    "Which regions have the highest correlation between disposable and capital item sales, and why?",
    "How do disposable and capital item sales vary by region for the top 15 products?",
]
question = st.radio("Choose a predefined question or type your own:", predefined_questions)
custom_question = st.text_input("Or type your custom question:")

if st.button("Submit"):
    if custom_question:
        question = custom_question

    if vector_store and question:
        # Retrieve context
        relevant_chunks = vector_store.similarity_search(question, k=3)
        context = " ".join([chunk.page_content for chunk in relevant_chunks])

        if len(context) > max_context_length:
            context = context[:max_context_length]

        # Generate response
        try:
            system_message = {
                "role": "system",
                "content": "You are an EDA assistant. Provide precise and concise answers based on the provided context. Ensure factual accuracy and include references where applicable."
            }
            user_message = {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely and include references with follow-up questions."
                )
            }
            llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
            response = llm.invoke([system_message, user_message], temperature=temperature)
            response_text = response.content

            # Display response
            st.markdown(f"<div class='response-box'><b>Response:</b><br>{response_text.strip()}</div>", unsafe_allow_html=True)

            # Log conversation
            st.session_state.conversation_history.append({"question": question, "response": response_text.strip()})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    else:
        st.warning("Please upload and process a document first.")

# Conversation history
if st.session_state.conversation_history:
    with st.expander("Conversation History"):
        for idx, entry in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Q{idx + 1}:** {entry['question']}")
            st.markdown(f"**A:** {entry['response']}")
