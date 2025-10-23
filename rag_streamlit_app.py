"""
Simple RAG Streamlit App
========================

A beginner-friendly web interface for the RAG system using Streamlit.

How to run:
1. Install requirements: pip install streamlit
2. Run the app: streamlit run rag_streamlit_app.py

Author: Your Name
Date: October 2025
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Import RAG components
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set page config
st.set_page_config(
    page_title="Simple RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)"""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('HUGGINGFACE_API_TOKEN')
        
        if not api_key:
            st.error("🔑 HUGGINGFACE_API_TOKEN not found in .env file!")
            return None, "API token missing"
        
        # Initialize LLM
        with st.spinner("🤖 Initializing AI model..."):
            llm_base = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                temperature=0.1,
                max_new_tokens=500,
                huggingfacehub_api_token=api_key
            )
            llm = ChatHuggingFace(llm=llm_base)
        
        # Initialize embeddings
        with st.spinner("🔢 Setting up embeddings..."):
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        
        # Load PDF
        pdf_path = "Chapter 01.pdf"
        if not os.path.exists(pdf_path):
            return None, f"PDF file '{pdf_path}' not found"
        
        with st.spinner("📄 Loading PDF document..."):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        
        # Split documents
        with st.spinner("✂️ Processing document..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased for more complete information
                chunk_overlap=100  # Increased overlap to avoid missing connections
            )
            splits = text_splitter.split_documents(docs)
        
        # Create vector store
        with st.spinner("🗃️ Creating knowledge base..."):
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embedding_model
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 6}  # Retrieve more chunks for comprehensive answers
            )
        
        # Create prompt template
        template = """
You are a helpful assistant that answers questions based on the provided context. 

IMPORTANT: Use ALL the relevant information from the context to provide a comprehensive answer. If the question asks about multiple items or functionalities, make sure to include ALL of them in your response, not just the first few.

Question: {question}

Context:
{context}

Instructions:
- Provide a complete and comprehensive answer using ALL relevant information from the context
- If the context mentions multiple items (like a list of functionalities), include ALL of them
- Structure your answer clearly with bullet points or numbers when appropriate
- If some information is not in the context, say so, but still provide what IS available

Answer:
"""
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()
        
        # Create RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )
        
        return chain, f"✅ System ready! Loaded {len(docs)} pages, {len(splits)} chunks."
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Simple RAG Chat System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Info")
        
        # Initialize RAG system
        if 'rag_chain' not in st.session_state:
            st.info("🔄 Initializing RAG system...")
            chain, status = initialize_rag_system()
            st.session_state.rag_chain = chain
            st.session_state.init_status = status
        
        # Display initialization status
        if st.session_state.rag_chain is not None:
            st.markdown(f'<div class="success-box">{st.session_state.init_status}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">{st.session_state.init_status}</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System information
        st.subheader("🔧 Configuration")
        st.write("**Model:** Mistral-7B-Instruct-v0.3")
        st.write("**Embeddings:** all-mpnet-base-v2")
        st.write("**Document:** Chapter 01.pdf")
        st.write("**Chunk Size:** 1000 tokens (improved)")
        st.write("**Retrieval:** 6 chunks (comprehensive)")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("💡 How to Use")
        st.write("1. Type your question in the chat box")
        st.write("2. Press Enter or click Send")
        st.write("3. Wait for the AI response")
        st.write("4. Ask follow-up questions!")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.rag_chain is not None:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "👋 Hello! I'm your RAG assistant. I can answer questions about your PDF document. What would you like to know?"
            })
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🤔 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🤔 You:</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)
            
            # Get AI response
            with st.spinner("🤖 AI is thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    # Extract content if it's an AIMessage object
                    if hasattr(response, 'content'):
                        answer = response.content
                    else:
                        answer = str(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Rerun to display the new message
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
    
    else:
        # Show setup instructions if RAG system failed to initialize
        st.error("🚫 RAG system is not ready. Please check the sidebar for details.")
        
        st.subheader("🔧 Setup Instructions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **1. Create .env file:**
            ```
            HUGGINGFACE_API_TOKEN=your_token_here
            ```
            """)
            
            st.markdown("""
            **2. Get HuggingFace Token:**
            - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
            - Create a new token
            - Copy and paste in .env file
            """)
        
        with col2:
            st.markdown("""
            **3. Add PDF file:**
            - Place 'Chapter 01.pdf' in the same folder
            - Or update the code with your PDF filename
            """)
            
            st.markdown("""
            **4. Install packages:**
            ```bash
            pip install -r requirements.txt
            ```
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    📚 Simple RAG Chat System | Built with Streamlit & LangChain
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()