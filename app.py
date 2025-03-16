import os
import argparse

import boto3
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# Set up environment
def setup_aws_credentials():
    """Setup AWS credentials from environment variables, .env file, or AWS SSO profile"""
    load_dotenv()
    
    # Check if using AWS SSO profile
    if os.getenv("AWS_PROFILE"):
        try:
            # Create a Bedrock client to test permissions
            bedrock = boto3.client('bedrock')
            # Test if we can list models (this will fail if permissions are incorrect)
            bedrock.list_foundation_models()
            return
        except Exception as e:
            st.error(f"Failed to access Bedrock with AWS profile: {str(e)}")
            st.error("Please ensure your AWS SSO session is active (aws sso login) and you have the required Bedrock permissions")
            st.stop()
    
    # If not using SSO, check for standard AWS credentials
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required AWS credentials: {', '.join(missing_vars)}")
        st.error("Please set up AWS credentials using one of the methods described in the README")
        st.stop()
    
    # Test Bedrock access with standard credentials
    try:
        bedrock = boto3.client('bedrock')
        bedrock.list_foundation_models()
    except Exception as e:
        st.error(f"Failed to access Bedrock: {str(e)}")
        st.error("Please ensure you have the required Bedrock permissions")
        st.stop()

setup_aws_credentials()

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Add argument parsing
parser = argparse.ArgumentParser(description='Document Q&A Chatbot')
parser.add_argument('--material-dir', 
                   default='./material',
                   help='Directory containing technical test material (default: ./material)')
parser.add_argument('--domain', 
                   required=True,
                   help='The domain of documents')
args = parser.parse_args()

# Load and process data
@st.cache_resource
def load_and_process_data():
    # Define persistent directory for Chroma
    persist_directory = ".cache/chroma_db"
    
    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}
    )
    
    # File to track processed documents
    processed_docs_file = ".cache/processed_docs.txt"
    processed_docs = set()
    
    # Load list of previously processed documents if it exists
    if os.path.exists(processed_docs_file):
        with open(processed_docs_file, 'r') as f:
            processed_docs = set(line.strip() for line in f.readlines())
    
    # Check if database already exists on disk
    if os.path.exists(persist_directory):
        # Load existing database from disk
        st.info("Loading existing vector database from disk...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Check for new documents
        data_directory = args.material_dir
        
        # Check for new CSV files
        new_csv_data = []
        csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            full_path = os.path.join(data_directory, csv_file)
            if full_path not in processed_docs and os.path.exists(full_path):
                st.info(f"Processing new CSV file: {full_path}")
                csv_loader = CSVLoader(full_path)
                new_csv_data.extend(csv_loader.load())
                processed_docs.add(full_path)
        
        # Check for new PDF files
        new_pdf_docs = []
        pdf_files = [f for f in os.listdir(data_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            full_path = os.path.join(data_directory, pdf_file)
            if full_path not in processed_docs:
                st.info(f"Processing new PDF: {full_path}")
                loader = PyPDFLoader(full_path)
                new_pdf_docs.extend(loader.load())
                processed_docs.add(full_path)
        
        # If we have new documents, process and add them to the vectorstore
        new_docs = new_csv_data + new_pdf_docs
        
        if new_docs:
            st.info(f"Adding {len(new_docs)} new documents to the existing database...")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            new_splits = text_splitter.split_documents(new_docs)
            
            # Add to existing vectorstore
            vectorstore.add_documents(new_splits)
            vectorstore.persist()
            
            # Update the processed documents file
            with open(processed_docs_file, 'w') as f:
                for doc in processed_docs:
                    f.write(f"{doc}\n")
        
        return vectorstore
    
    # If no database exists, create a new one
    st.info("Creating new vector database (this may take a while)...")
    
    # Initialize set of processed documents
    processed_docs = set()
    
    # Process all documents from the directory
    data_directory = args.material_dir
    all_docs = []
    
    # Process CSV files
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    for csv_file in csv_files:
        full_path = os.path.join(data_directory, csv_file)
        csv_loader = CSVLoader(full_path)
        all_docs.extend(csv_loader.load())
        processed_docs.add(full_path)
        st.info(f"Processed CSV file: {full_path}")
    
    # Process PDF files
    pdf_files = [f for f in os.listdir(data_directory) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        full_path = os.path.join(data_directory, pdf_file)
        loader = PyPDFLoader(full_path)
        all_docs.extend(loader.load())
        processed_docs.add(full_path)
        st.info(f"Processed PDF file: {full_path}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)
    
    # Create and persist the vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Explicitly persist to disk
    vectorstore.persist()
    
    # Save list of processed documents
    with open(processed_docs_file, 'w') as f:
        for doc in processed_docs:
            f.write(f"{doc}\n")
    
    return vectorstore

# Set up the agent
def setup_agent(vectorstore):
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Define tools
    tools = [
        Tool(
            name=f"{args.domain} Database",
            func=retriever.get_relevant_documents,
            description=f"Useful for answering questions about {args.domain}. Use this tool first for any {args.domain.lower()} related queries."
        ),
        Tool(
            name="Web Search",
            func=search_tool.run,
            description="Useful for finding information that might not be in the database. Use this when you need additional context or when the database doesn't have the answer."
        )
    ]
    
    # Create Bedrock LLM
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name=os.getenv("AWS_DEFAULT_REGION"),
        model_kwargs={
            "temperature": 0,
            "max_tokens": 1000,
        }
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(f"""
    You are a helpful assistant that specializes in {args.domain.lower()}. 
    You have access to a database of {args.domain.lower()} and can search the web for additional information.
    
    When answering questions:
    1. Be precise and factual
    2. Cite your sources (database or web search)
    3. If comparing items, provide clear metrics
    4. If you don't know, say so rather than making up information

    You have access to the following tools:

    {{tools}}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{{tool_names}}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    {{chat_history}}
    Question: {{input}}
    {{agent_scratchpad}}
    """)
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# Streamlit UI
st.title(f"{args.domain.capitalize()} Q&A Chatbot")
st.markdown(f"""
This chatbot can answer questions about {args.domain.lower()} using:
- Planning application dataset
- Supporting PDF documents
- Web search for additional information
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load data and set up agent
vectorstore = load_and_process_data()
agent_executor = setup_agent(vectorstore)

# Chat input
if prompt := st.chat_input(f"Ask a question about {args.domain.lower()}"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})
            st.markdown(response["output"])
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["output"]}) 