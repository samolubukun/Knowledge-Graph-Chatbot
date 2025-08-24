import os
import io
import tempfile
from typing import List, Tuple, Optional
import gradio as gr
from dotenv import load_dotenv

# --- Web scraping ---
import requests
from bs4 import BeautifulSoup

# --- LangChain core ---
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# --- Loaders (files) ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

# --- Neo4j + Vector store + Graph ---
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain

# --- LLMs & Embeddings: Cohere or Gemini ---
# Cohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import Cohere

# Gemini (Google Generative AI API key)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Global state management
class AppState:
    def __init__(self):
        self.graph: Optional[Neo4jGraph] = None
        self.vs: Optional[Neo4jVector] = None
        self.llm = None
        self.embeddings = None
        self.chat_history = []

app_state = AppState()

# ===============================
# Helpers
# ===============================

def clean_chunks(docs: List[Document], chunk_size=800, chunk_overlap=120) -> List[Document]:
    """Split to moderately large chunks for better retrieval and context quality."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def load_and_split_file(file_path: str) -> List[Document]:
    """Load a single file (by extension) and return split docs."""
    filename = os.path.basename(file_path)
    name = filename.lower()
    _, ext = os.path.splitext(name)
    ext = ext.lstrip(".")

    try:
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext in ("docx", "doc"):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif ext == "csv":
            loader = CSVLoader(file_path, csv_args={"delimiter": ","})
        elif ext in ("xlsx", "xls"):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        else:
            print(f"Unsupported file type: {ext}")
            return []

        docs = loader.load()
        # Attach source metadata
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = filename

        return clean_chunks(docs)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []


def scrape_webpage(url: str) -> List[Document]:
    """Scrape a single URL (no crawling), extract visible text, split into chunks."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()

        # Restrict to likely content areas to reduce nav noise
        main_candidates = soup.select("main, article, section, .content, .post, .entry") or [soup.body or soup]
        texts = []
        for node in main_candidates:
            if node:
                text = node.get_text(separator=" ", strip=True)
                if text and len(text) > 50:  # Only keep substantial text
                    texts.append(text)
        
        joined = " ".join(texts).strip()
        if not joined or len(joined) < 100:
            return []

        base_doc = Document(page_content=joined, metadata={"source": url, "type": "web"})
        return clean_chunks([base_doc], chunk_size=800, chunk_overlap=120)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []


def init_models(provider: str, api_key: str):
    """Initialize LLM and embeddings for the chosen provider."""
    if provider == "Cohere":
        if not api_key:
            raise ValueError("Please provide a Cohere API key.")
        
        # Initialize Cohere LLM
        llm = Cohere(model="command", temperature=0.2, cohere_api_key=api_key)
        
        # Initialize Cohere Embeddings with user_agent parameter
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0", 
            cohere_api_key=api_key,
            user_agent="langchain-knowledge-graph-chatbot"
        )
        return llm, embeddings

    elif provider == "Gemini":
        if not api_key:
            raise ValueError("Please provide a Gemini API key.")
        
        # Chat + Embeddings via Google Generative AI (no GCP project required)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0.2, 
            google_api_key=api_key
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        return llm, embeddings

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def upsert_chunks_vector_index(
    docs: List[Document],
    embeddings,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "neo4j",
    node_label: str = "Chunk",
    text_prop: str = "text",
    embed_prop: str = "embedding",
    index_name: str = "chunk_vector_index",
    keyword_index_name: str = "chunk_keyword_index",
):
    """Create/update a Neo4j vector index with chunk nodes for retrieval."""
    # Ensure text property exists per Document
    prepared_docs = []
    for d in docs:
        content = d.page_content.strip()
        if not content:
            continue
        d.metadata = d.metadata or {}
        # Neo4jVector expects text under a known property; it will write it.
        prepared_docs.append(Document(page_content=content, metadata=d.metadata))

    if not prepared_docs:
        return None

    vs = Neo4jVector.from_documents(
        documents=prepared_docs,
        embedding=embeddings,
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        database=database,
        node_label=node_label,
        text_node_property=text_prop,
        embedding_node_property=embed_prop,
        index_name=index_name,
        keyword_index_name=keyword_index_name,
        # Removed search_type parameter - let it use default
    )
    return vs


def build_kg_with_llm(
    docs: List[Document],
    graph: Neo4jGraph,
    llm,
    allowed_nodes: List[str],
    allowed_rels: List[str],
):
    """Extract a lean, controllable KG from your documents and persist in Neo4j."""
    try:
        # Try to import json_repair, install if missing
        try:
            import json_repair
        except ImportError:
            print("Installing json-repair package...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "json-repair"])
            import json_repair
        
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_rels,
            node_properties=False,
            relationship_properties=False,
        )
        
        # Process documents in smaller batches to avoid token limits
        batch_size = 3
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                graph_docs = transformer.convert_to_graph_documents(batch)
                if graph_docs:
                    graph.add_graph_documents(graph_docs, include_source=True)
                    print(f"Successfully processed batch {batch_num}")
                else:
                    print(f"No graph documents generated for batch {batch_num}")
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                continue
                
    except Exception as e:
        print(f"Knowledge graph extraction error: {e}")
        raise e


def query_knowledge_graph(graph: Neo4jGraph, question: str, llm) -> str:
    """Query the knowledge graph using natural language and return results."""
    try:
        # Create a GraphCypherQAChain to query the knowledge graph
        cypher_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True  # Allow complex queries
        )
        
        # Query the knowledge graph
        result = cypher_chain(question)
        
        # Extract the answer and query information
        answer = result.get("result", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Format the response with graph context
        formatted_answer = f"**Knowledge Graph Answer:**\n{answer}"
        
        # Add query information if available
        if intermediate_steps:
            for step in intermediate_steps:
                if "query" in step:
                    formatted_answer += f"\n\n*Graph Query Used:* `{step['query']}`"
        
        return formatted_answer
        
    except Exception as e:
        return f"Error querying knowledge graph: {e}"


def hybrid_retrieval_answer(
    question: str, 
    graph: Neo4jGraph, 
    vs: Neo4jVector, 
    llm
) -> str:
    """Combine knowledge graph querying with vector search for comprehensive answers."""
    
    # 1. Query the Knowledge Graph first
    kg_answer = query_knowledge_graph(graph, question, llm)
    
    # 2. Get vector search results
    try:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        relevant_docs = retriever.get_relevant_documents(question)
        
        context_texts = []
        for d in relevant_docs:
            src = d.metadata.get("source", "unknown")
            snippet = d.page_content[:1200]
            context_texts.append(f"[Source: {src}] {snippet}")
            
        vector_context = "\n\n---\n\n".join(context_texts)
    except Exception as e:
        vector_context = f"Vector search error: {e}"
    
    # 3. Combine both approaches for a comprehensive answer
    combined_prompt = f"""
You are a helpful assistant that must provide comprehensive answers using BOTH knowledge graph data and document context.

KNOWLEDGE GRAPH RESULTS:
{kg_answer}

DOCUMENT CONTEXT:
{vector_context}

USER QUESTION: {question}

Instructions:
- Synthesize information from BOTH the knowledge graph and document context
- If the knowledge graph provides structured relationships, highlight those
- If the documents provide additional details, include those
- Always cite sources when possible
- If information conflicts, note the discrepancy
- If neither source has sufficient information, say so clearly

Provide a comprehensive answer that leverages both structured knowledge and document content:
"""
    
    try:
        response = llm.invoke(combined_prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        return f"Error generating combined answer: {e}"


# ===============================
# Gradio Interface Functions
# ===============================

def connect_neo4j(neo4j_url: str, neo4j_user: str, neo4j_password: str) -> str:
    """Connect to Neo4j database and check for existing data."""
    try:
        app_state.graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
        
        # Check if there's existing data in the database
        try:
            # Check for existing chunks/nodes
            chunk_count = app_state.graph.query("MATCH (n:Chunk) RETURN count(n) as count")[0]["count"]
            entity_count = app_state.graph.query("MATCH (n) WHERE NOT n:Chunk RETURN count(n) as count")[0]["count"]
            
            status_msg = f"✅ Successfully connected to Neo4j!\n"
            status_msg += f"📊 Found {chunk_count} document chunks and {entity_count} knowledge graph entities"
            
            # Try to reconnect to existing vector store if chunks exist
            if chunk_count > 0:
                try:
                    # We need embeddings to reconnect to vector store
                    # For now, just indicate data exists but needs model setup
                    status_msg += f"\n💡 Existing data detected! Please set up your LLM provider and click 'Reconnect to Existing Data' to restore full functionality."
                except Exception as e:
                    status_msg += f"\n⚠️ Data found but vector store needs reconnection."
            
            return status_msg
            
        except Exception as e:
            return "✅ Successfully connected to Neo4j! (Empty database)"
            
    except Exception as e:
        return f"❌ Neo4j connection failed: {e}"


def reconnect_existing_data(
    provider: str,
    api_key: str,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str
) -> str:
    """Reconnect to existing vector store and LLM models."""
    if app_state.graph is None:
        return "❌ Please connect to Neo4j first."
    
    try:
        # Initialize models
        llm, embeddings = init_models(provider, api_key)
        app_state.llm = llm
        app_state.embeddings = embeddings
        
        # Check if chunks exist
        chunk_count = app_state.graph.query("MATCH (n:Chunk) RETURN count(n) as count")[0]["count"]
        
        if chunk_count == 0:
            return "❌ No existing data found. Please ingest new data first."
        
        # Reconnect to existing vector store
        try:
            app_state.vs = Neo4jVector(
                embedding=embeddings,
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password,
                database="neo4j",
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
                index_name="chunk_vector_index",
                keyword_index_name="chunk_keyword_index",
            )
            
            # Test the vector store
            test_results = app_state.vs.similarity_search("test", k=1)
            
            return f"✅ Successfully reconnected to existing data! Found {chunk_count} chunks. Vector store is ready for chat."
            
        except Exception as vs_error:
            # If vector store connection fails, try to rebuild it
            return f"⚠️ Vector store connection failed: {vs_error}. You may need to re-ingest your data."
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Reconnection error: {error_details}")
        return f"❌ Reconnection failed: {str(e)}"


def wipe_database() -> str:
    """Wipe the Neo4j database."""
    if app_state.graph is None:
        return "❌ Please connect to Neo4j first."
    
    try:
        app_state.graph.query("MATCH (n) DETACH DELETE n;")
        return "✅ Database successfully wiped!"
    except Exception as e:
        return f"❌ Failed to wipe database: {e}"


def process_knowledge(
    provider: str,
    api_key: str,
    files: List[str],
    urls: str,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str
) -> str:
    """Process files and URLs to build knowledge graph."""
    if app_state.graph is None:
        return "❌ Please connect to Neo4j first."
    
    try:
        # Initialize models
        llm, embeddings = init_models(provider, api_key)
        app_state.llm = llm
        app_state.embeddings = embeddings
        
        all_docs: List[Document] = []
        processed_files = 0
        processed_urls = 0
        
        # Process uploaded files
        if files:
            for file_path in files:
                if file_path:  # Check if file_path is not None/empty
                    try:
                        print(f"Processing file: {file_path}")
                        file_docs = load_and_split_file(file_path)
                        all_docs.extend(file_docs)
                        processed_files += 1
                        print(f"Successfully processed {file_path}: {len(file_docs)} chunks")
                    except Exception as e:
                        print(f"Failed to process file {file_path}: {e}")
                        continue
        
        # Process URLs
        if urls and urls.strip():
            url_list = [u.strip() for u in urls.splitlines() if u.strip()]
            for url in url_list:
                try:
                    print(f"Processing URL: {url}")
                    url_docs = scrape_webpage(url)
                    all_docs.extend(url_docs)
                    processed_urls += 1
                    print(f"Successfully processed {url}: {len(url_docs)} chunks")
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    continue
        
        if not all_docs:
            return f"⚠️ No data extracted. Processed {processed_files} files and {processed_urls} URLs, but no usable content found."
        
        print(f"Total documents to process: {len(all_docs)}")
        
        # Build Knowledge Graph
        allowed_nodes = ["Entity", "Concept", "Person", "Organization", "Location", "Event", "Fact"]
        allowed_rels = ["RELATED_TO", "MENTIONS", "PART_OF", "CAUSES", "ASSOCIATED_WITH"]
        
        try:
            print("Building knowledge graph...")
            build_kg_with_llm(all_docs, app_state.graph, llm, allowed_nodes, allowed_rels)
            print("Knowledge graph built successfully")
        except Exception as e:
            print(f"KG extraction error: {e}")
            return f"❌ KG extraction failed: {e}"
        
        # Build Vector Index
        try:
            print("Building vector index...")
            vs = upsert_chunks_vector_index(
                docs=all_docs,
                embeddings=embeddings,
                neo4j_url=neo4j_url,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                node_label="Chunk",
                text_prop="text",
                embed_prop="embedding",
                index_name="chunk_vector_index",
                keyword_index_name="chunk_keyword_index",
            )
            app_state.vs = vs
            print("Vector index built successfully")
        except Exception as e:
            print(f"Vector indexing error: {e}")
            return f"❌ Vector indexing failed: {e}"
        
        return f"✅ Successfully processed {processed_files} files and {processed_urls} URLs ({len(all_docs)} total chunks)! Knowledge graph and vector index are ready."
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error details: {error_details}")
        return f"❌ Processing failed: {str(e)}"


def chat_with_knowledge(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Chat function using both knowledge graph and vector search."""
    if app_state.graph is None or app_state.vs is None:
        response = "❌ Please connect to Neo4j and ingest data first."
        history.append((message, response))
        return "", history
    
    if app_state.llm is None:
        response = "❌ Model not initialized. Please process some data first."
        history.append((message, response))
        return "", history
    
    try:
        # Use hybrid approach: Knowledge Graph + Vector Search
        answer = hybrid_retrieval_answer(
            question=message,
            graph=app_state.graph,
            vs=app_state.vs,
            llm=app_state.llm
        )
        
        if not answer or answer.strip() == "":
            answer = "I don't have enough information to answer that based on the ingested data."
        
        history.append((message, answer))
        return "", history
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Chat error details: {error_details}")
        response = f"❌ Error during chat: {str(e)}"
        history.append((message, response))
        return "", history


def clear_chat_history():
    """Clear the chat history."""
    app_state.chat_history = []
    return []


# ===============================
# Gradio Interface
# ===============================

def create_interface():
    """Create the Gradio interface."""
    load_dotenv()
    
    with gr.Blocks(title="Knowledge Graph Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 Knowledge Graph Chatbot (Strict)")
        gr.Markdown("Upload documents, scrape URLs, and chat with your knowledge using Neo4j and vector search!")
        
        with gr.Tab("🔧 Setup & Configuration"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model Settings")
                    provider = gr.Dropdown(
                        choices=["Cohere", "Gemini"],
                        value="Cohere",
                        label="LLM Provider"
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        value=os.getenv("COHERE_API_KEY", ""),
                        placeholder="Enter your API key"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Neo4j Configuration")
                    neo4j_url = gr.Textbox(
                        label="Neo4j URL",
                        value=os.getenv("NEO4J_URI", "neo4j+s://your-neo4j-url"),
                        placeholder="neo4j+s://your-neo4j-url"
                    )
                    neo4j_user = gr.Textbox(
                        label="Username",
                        value=os.getenv("NEO4J_USER", "neo4j")
                    )
                    neo4j_password = gr.Textbox(
                        label="Password",
                        type="password",
                        value=os.getenv("NEO4J_PASSWORD", "")
                    )
            
            with gr.Row():
                connect_btn = gr.Button("🔗 Connect to Neo4j", variant="primary")
                wipe_btn = gr.Button("🗑️ Wipe Database", variant="stop")
                reconnect_btn = gr.Button("🔄 Reconnect to Existing Data", variant="secondary")
            
            connection_status = gr.Textbox(
                label="Connection Status",
                interactive=False,
                placeholder="Click 'Connect to Neo4j' to establish connection"
            )
            
            connect_btn.click(
                fn=connect_neo4j,
                inputs=[neo4j_url, neo4j_user, neo4j_password],
                outputs=[connection_status]
            )
            
            wipe_btn.click(
                fn=wipe_database,
                outputs=[connection_status]
            )
            
            reconnect_btn.click(
                fn=reconnect_existing_data,
                inputs=[provider, api_key, neo4j_url, neo4j_user, neo4j_password],
                outputs=[connection_status]
            )
        
        with gr.Tab("📁 Data Ingestion"):
            gr.Markdown("### Upload Knowledge Sources")
            
            files = gr.File(
                label="Upload Files",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".csv", ".xls", ".xlsx"],
                file_count="multiple"
            )
            
            urls = gr.Textbox(
                label="URLs to Scrape",
                placeholder="Enter URLs, one per line",
                lines=5
            )
            
            process_btn = gr.Button("🚀 Process & Build Knowledge Graph", variant="primary")
            
            processing_status = gr.Textbox(
                label="Processing Status",
                interactive=False,
                placeholder="Click 'Process & Build Knowledge Graph' to start"
            )
            
            process_btn.click(
                fn=process_knowledge,
                inputs=[provider, api_key, files, urls, neo4j_url, neo4j_user, neo4j_password],
                outputs=[processing_status]
            )
        
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Chat with Your Knowledge Graph")
            gr.Markdown("Ask questions about your ingested data. The system uses **both knowledge graph queries and vector search** for comprehensive answers.")
            
            chatbot = gr.Chatbot(
                label="Knowledge Graph Chat",
                height=500,
                placeholder="Your conversation will appear here..."
            )
            
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about entities, relationships, or any content from your data...",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary")
            
            # Example questions
            with gr.Accordion("💡 Example Questions", open=False):
                gr.Markdown("""
                **Entity-based questions:**
                - "What organizations are mentioned in the documents?"
                - "Tell me about [person name] and their relationships"
                - "What events are connected to [organization]?"
                
                **Relationship queries:**
                - "How are [entity1] and [entity2] related?"
                - "What causes [concept] according to the documents?"
                - "Show me all connections to [topic]"
                
                **Content questions:**
                - "Summarize the main concepts in the documents"
                - "What are the key findings about [topic]?"
                - "Explain [concept] based on the ingested data"
                """)
            
            # Chat functionality
            msg_box.submit(
                fn=chat_with_knowledge,
                inputs=[msg_box, chatbot],
                outputs=[msg_box, chatbot]
            )
            
            send_btn.click(
                fn=chat_with_knowledge,
                inputs=[msg_box, chatbot],
                outputs=[msg_box, chatbot]
            )
            
            clear_btn.click(
                fn=clear_chat_history,
                outputs=[chatbot]
            )
        
        with gr.Tab("ℹ️ Instructions"):
            gr.Markdown("""
            ## How to Use This Knowledge Graph Chatbot
            
            ### 1. Setup & Configuration
            - Choose your LLM provider (Cohere or Gemini)
            - Enter your API key for the chosen provider
            - Configure your Neo4j connection details
            - Click "Connect to Neo4j" to establish the database connection
            
            ### 1.5. Reconnecting to Existing Data
            **If you already have data in Neo4j from a previous session:**
            - After connecting to Neo4j, if you see existing data detected
            - Set up your LLM provider and API key
            - Click "🔄 Reconnect to Existing Data" instead of re-ingesting
            - This will restore your vector store and enable chat without re-processing documents
            
            ### 2. Data Ingestion
            - Upload files (PDF, DOCX, TXT, CSV, XLS, XLSX) or enter URLs to scrape
            - Click "Process & Build Knowledge Graph" to:
              - Extract text from your sources
              - Build a knowledge graph using LLM-based entity extraction
              - Create a vector index for semantic search
            
            ### 3. Chat
            - Ask questions about your ingested data
            - The chatbot will provide **strict** answers only based on your uploaded content
            - If the answer isn't in your data, it will explicitly say so
            
            ### Features
            - **Knowledge Graph Queries**: Direct Cypher queries to find entities and relationships
            - **Vector Semantic Search**: Dense vector similarity search for relevant content
            - **Hybrid Intelligence**: Combines structured graph data with unstructured document content
            - **Source Attribution**: Answers include references to source files/URLs
            - **Strict Mode**: Only answers from your ingested data, no hallucination
            - **Entity Extraction**: Automatically identifies people, organizations, locations, events
            - **Relationship Mapping**: Discovers and queries connections between entities
            - **Batch Processing**: Handles large document collections efficiently
            
            ### Requirements
            - Neo4j database (Neo4j Aura or self-hosted)
            - API key for Cohere or Google Gemini
            - Documents or URLs to process
            
            ### Required Packages for Kaggle
            Run this in a Kaggle cell before using the interface:
            ```python
            !pip install gradio langchain neo4j beautifulsoup4 requests python-dotenv
            !pip install langchain-community langchain-experimental
            !pip install langchain-google-genai cohere
            !pip install json-repair  # Required for knowledge graph extraction
            !pip install unstructured[all-docs]  # For better document parsing
            ```
            
            ### For Kaggle Notebooks
            This interface is optimized for Kaggle notebooks. Make sure to:
            1. Install required packages in your notebook
            2. Set up your API keys as environment variables or enter them in the interface
            3. Use a cloud-hosted Neo4j instance (like Neo4j Aura)
            """)
    
    return demo


# ===============================
# Main Function
# ===============================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,  # Create a public link for sharing
        debug=True,
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )
