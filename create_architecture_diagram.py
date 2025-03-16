from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.aws.storage import S3
from diagrams.aws.compute import Lambda
from diagrams.aws.network import APIGateway
from diagrams.custom import Custom
from diagrams.onprem.client import User
from typing import Any  # Add type hints to address linter errors

with Diagram("Document Q&A Chatbot Architecture", show=False, direction="TB"):
    user = User("User")
    
    with Cluster("Frontend"):
        ui = Custom("Streamlit UI", "./icons/ui.png")
    
    with Cluster("Data Sources"):
        documents = Custom("PDF/Text Documents", "./icons/pdf.png")
    
    with Cluster("Processing Pipeline"):
        loader = Custom("Document Loader", "./icons/loader.png")
        splitter = Custom("Text Splitter", "./icons/splitter.jpg")
        embeddings = Custom("HuggingFace Embeddings", "./icons/embedding.jpg")
        vector_db = Custom("Chroma Vector Store", "./icons/database.png")
        
        documents >> loader >> splitter >> embeddings >> vector_db
    
    with Cluster("Reasoning Layer"):
        llm = Custom("Claude 3.5 Sonnet (Bedrock)", "./icons/llm.jpg")
        retriever = Custom("ReAct Agent", "./icons/agent.png")
        
        vector_db >> retriever
        retriever >> llm
    
    with Cluster("Tools"):
        db_tool = Custom("Domain Database", "./icons/database.png")
        search_tool = Custom("Web Search", "./icons/search.png")
        
        db_tool >> retriever
        search_tool >> retriever
    
    with Cluster("Memory"):
        memory = Custom("Chat History", "./icons/memory.png")
        
        llm >> memory
        memory >> retriever
    
    # Connect the user flow
    user >> ui
    ui >> retriever
    llm >> ui 