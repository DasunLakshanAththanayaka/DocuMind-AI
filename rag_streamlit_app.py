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
def initialize_rag_system(uploaded_file_content=None, uploaded_file_name="uploaded_document.pdf"):
    """Initialize the RAG system with uploaded PDF (cached for performance)"""
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
        
        # Load PDF from uploaded file only
        if uploaded_file_content is None:
            return None, "No PDF file provided. Please upload a PDF file to get started."
        
        with st.spinner("📄 Loading PDF document..."):
            # Save uploaded file temporarily
            temp_pdf_path = f"temp_{uploaded_file_name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file_content)
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            # Clean up temporary file
            os.remove(temp_pdf_path)
            pdf_source = uploaded_file_name
        
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
        
        return chain, f"✅ System ready! Loaded '{pdf_source}': {len(docs)} pages, {len(splits)} chunks."
        
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
        
        # File Upload Section
        st.subheader("📁 Upload Your PDF (Required)")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload any PDF document you want to ask questions about"
        )
        
        # Display upload status
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size:,} bytes")
        else:
            st.warning("⚠️ Please upload a PDF file to get started")
            st.info("📤 No file uploaded yet")
        
        st.markdown("---")
        
        # Initialize or reinitialize RAG system based on uploaded file
        if 'rag_chain' not in st.session_state or 'current_file' not in st.session_state:
            st.session_state.current_file = None
            st.session_state.rag_chain = None
            st.session_state.init_status = "Not initialized"
        
        # Check if we need to reinitialize due to new file upload
        file_changed = False
        if uploaded_file is not None:
            current_file_name = uploaded_file.name
            if st.session_state.current_file != current_file_name:
                file_changed = True
                st.session_state.current_file = current_file_name
        else:
            if st.session_state.current_file is not None:
                file_changed = True
                st.session_state.current_file = None
        
        # Initialize or reinitialize RAG system
        if st.session_state.rag_chain is None or file_changed:
            if uploaded_file is not None:
                if file_changed:
                    st.info("🔄 New file detected, reinitializing system...")
                    # Clear cache for reinitialization
                    initialize_rag_system.clear()
                else:
                    st.info("🔄 Initializing RAG system...")
                
                # Use uploaded file
                chain, status = initialize_rag_system(
                    uploaded_file_content=uploaded_file.read(),
                    uploaded_file_name=uploaded_file.name
                )
                st.session_state.rag_chain = chain
                st.session_state.init_status = status
            else:
                # No file uploaded - system cannot initialize
                st.session_state.rag_chain = None
                st.session_state.init_status = "⚠️ Please upload a PDF file to start using the system."
        
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
        
        # Show current document
        if st.session_state.current_file:
            st.write(f"**Document:** {st.session_state.current_file}")
        else:
            st.write("**Document:** None (upload required)")
            
        st.write("**Chunk Size:** 1000 tokens")
        st.write("**Retrieval:** 6 chunks")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("💡 How to Use")
        st.write("1. 📁 Upload your PDF file above (required)")
        st.write("2. ✅ Wait for system initialization")
        st.write("3. 💬 Type your question in the chat box")
        st.write("4. 📤 Press Enter or click Send")
        st.write("5. ⏳ Wait for the AI response")
        st.write("6. 🔄 Ask follow-up questions!")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reload System"):
                initialize_rag_system.clear()
                st.session_state.rag_chain = None
                st.rerun()
    
    # Main chat interface
    if st.session_state.rag_chain is not None:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            welcome_msg = "👋 Hello! I'm your RAG assistant. I can answer questions about your PDF document. "
            if st.session_state.current_file:
                welcome_msg += f"I'm currently analyzing '{st.session_state.current_file}'. "
            welcome_msg += "What would you like to know?"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg
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
            with st.spinner("🤖Thinking..."):
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
        if uploaded_file is None:
            st.info("📁 Please upload a PDF file in the sidebar to start chatting with your documents!")
            st.markdown("""
            ### � Getting Started
            
            1. **Upload your PDF** using the file uploader in the sidebar
            2. **Wait for processing** - the system will analyze your document
            3. **Start chatting** - ask questions about your PDF content
            
            ### 📋 Supported Files
            - Any PDF document
            - Academic papers, books, manuals, reports
            - Text-based PDFs work best
            """)
        else:
            st.error("🚫 RAG system failed to initialize. Please check the sidebar for details.")
        
        
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    📚 Simple RAG Chat System | Built with Streamlit & LangChain
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()