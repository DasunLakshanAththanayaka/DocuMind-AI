"""
🧠 RAG Chat System with Memory
A beginner-friendly RAG system that remembers your conversation for follow-up questions.
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Configure page
st.set_page_config(
    page_title=" RAG Chat with Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 5px solid;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .memory-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        font-size: 0.9rem;
        opacity: 0.8;
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
    .memory-stats {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_chat_history():
    """Get formatted conversation history for the AI"""
    if 'conversation_memory' not in st.session_state or not st.session_state.conversation_memory:
        return "This is the start of our conversation."
    
    # Get last 5 exchanges to keep context manageable
    recent_memory = st.session_state.conversation_memory[-5:]
    
    history = "Previous conversation:\n"
    for i, exchange in enumerate(recent_memory, 1):
        history += f"Q{i}: {exchange['question']}\n"
        history += f"A{i}: {exchange['answer'][:200]}...\n\n"  # Truncate long answers
    
    return history

def save_to_memory(question, answer):
    """Save Q&A exchange to conversation memory"""
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = []
    
    # Add new exchange
    st.session_state.conversation_memory.append({
        'question': question,
        'answer': answer,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Keep only last 10 exchanges to prevent memory overflow
    if len(st.session_state.conversation_memory) > 10:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-10:]

def clear_memory():
    """Clear conversation memory"""
    st.session_state.conversation_memory = []
    st.session_state.messages = []

@st.cache_resource
def initialize_rag_system_with_memory(uploaded_file_content=None, uploaded_file_name="uploaded_document.pdf"):
    """Initialize the RAG system with memory capabilities"""
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
                chunk_size=1000,  # Good chunk size for context
                chunk_overlap=100  # Overlap to maintain connections
            )
            splits = text_splitter.split_documents(docs)
        
        # Create vector store
        with st.spinner("🗃️ Creating knowledge base..."):
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embedding_model
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 6}  # Retrieve multiple chunks
            )
        
        # Create enhanced prompt template with memory
        template = """
You are a helpful assistant that answers questions based on the provided context and conversation history.

Conversation History:
{chat_history}

Current Question: {question}

Document Context:
{context}

Instructions:
- Use the conversation history to understand what the current question is referring to
- If the question uses pronouns like "it", "that", "them", "this", determine what they refer to from the conversation history
- Answer the question directly and naturally without mentioning that you're using conversation history
- Provide a comprehensive answer using the document context
- Do NOT say phrases like "as discussed in the conversation history" or "referring to the previous topic"
- Simply answer the question as if the context is naturally understood
- Keep your answer focused, direct, and natural

Answer:
"""
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()
        
        # Create RAG chain with memory
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: get_chat_history()
            }
            | prompt
            | llm
            | output_parser
        )
        
        return chain, f"✅ System with Memory ready! Loaded '{pdf_source}': {len(docs)} pages, {len(splits)} chunks."
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def display_memory_sidebar():
    """Display conversation memory in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Conversation Memory")
    
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = []
    
    memory_count = len(st.session_state.conversation_memory)
    if memory_count > 0:
        st.sidebar.markdown(f'<div class="memory-stats">💭 Remembered: {memory_count} exchanges</div>', 
                           unsafe_allow_html=True)
        
        # Show recent memory
        st.sidebar.write("**Recent topics:**")
        for i, exchange in enumerate(st.session_state.conversation_memory[-3:], 1):
            st.sidebar.write(f"{i}. {exchange['question'][:50]}...")
        
        # Memory controls
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🧹 Clear Memory"):
                clear_memory()
                st.rerun()
        with col2:
            if st.button("📜 Show All"):
                st.session_state.show_memory_details = not st.session_state.get('show_memory_details', False)
        
        # Show detailed memory if requested
        if st.session_state.get('show_memory_details', False):
            st.sidebar.markdown("**Memory Details:**")
            for i, exchange in enumerate(st.session_state.conversation_memory, 1):
                with st.sidebar.expander(f"Exchange {i} ({exchange['timestamp']})"):
                    st.write(f"**Q:** {exchange['question']}")
                    st.write(f"**A:** {exchange['answer'][:100]}...")
    # else:
    #     st.sidebar.info("💭 No conversation history yet. Start asking questions!")

def main():
    """Main Streamlit app with memory"""
    
    # Initialize conversation memory
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = []
    
    # Header
    st.markdown('<h1 class="main-header">🧠 DocuMind AI</h1>', unsafe_allow_html=True)
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
            # st.info(f"📊 File size: {uploaded_file.size:,} bytes")
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
                # Clear memory when switching documents
                clear_memory()
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
                    initialize_rag_system_with_memory.clear()
                else:
                    st.info("🔄 Initializing RAG system with memory...")
                
                # Use uploaded file
                chain, status = initialize_rag_system_with_memory(
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
        
        # # System information
        # st.subheader("🔧 Configuration")
        # st.write("**Model:** Mistral-7B-Instruct-v0.3")
        # st.write("**Embeddings:** all-mpnet-base-v2")
        # st.write("**Memory:** ✅ Enabled (10 exchanges)")
        
        # Show current document
        if st.session_state.current_file:
            st.write(f"**Document:** {st.session_state.current_file}")
        else:
            st.write("**Document:** None (upload required)")
            
        # st.write("**Chunk Size:** 1000 tokens")
        # st.write("**Retrieval:** 6 chunks")
        
        # Display memory information
        display_memory_sidebar()
        
        st.markdown("---")
        
        # Instructions
        st.subheader("💡 How to Use with Memory")
        st.write("1. 📁 Upload your PDF file above")
        st.write("2. ✅ Wait for system initialization")
        st.write("3. 💬 Ask your questions")
        # st.write("4. 🧠 Ask follow-up questions using 'it', 'that', etc.")
        # st.write("5. 📜 Check memory sidebar for conversation history")
        # st.write("6. 🧹 Clear memory to start fresh")
        
        
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reload System"):
                initialize_rag_system_with_memory.clear()
                st.session_state.rag_chain = None
                st.rerun()
    
    # Main chat interface
    if st.session_state.rag_chain is not None:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            welcome_msg = "👋 Hello! I'm your RAG assistant with memory! 🧠\n\n"
            welcome_msg += "I can remember our conversation, so you can ask follow-up questions like:\n"
            welcome_msg += "- 'What are its main features?' (after asking about something)\n"
            welcome_msg += "- 'Can you explain that further?'\n"
            welcome_msg += "- 'How does it work?'\n\n"
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
        if prompt := st.chat_input("Ask me anything about your document...."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🤔 You:</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)
            
            # Show memory usage if available
            if st.session_state.conversation_memory:
                memory_count = len(st.session_state.conversation_memory)
                # st.markdown(f"""
                # <div class="chat-message memory-message">
                #     <strong>🧠 Using Memory:</strong> I remember {memory_count} previous exchanges to understand your question better.
                # </div>
                # """, unsafe_allow_html=True)
            
            # Get AI response
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    # Extract content if it's an AIMessage object
                    if hasattr(response, 'content'):
                        answer = response.content
                    else:
                        answer = str(response)
                    
                    # Save to memory
                    save_to_memory(prompt, answer)
                    
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
            ### 🚀 Getting Started with Memory
            
            1. **Upload your PDF** using the file uploader in the sidebar
            2. **Wait for processing** - the system will analyze your document
            3. **Start chatting** - ask your first question
            4. **Ask follow-ups** - use phrases like "What about that?", "How does it work?", etc.
            5. **Check memory** - see conversation history in the sidebar
            
            ### 🧠 Memory Features
            - Remembers up to 10 conversation exchanges
            - Understands follow-up questions with pronouns (it, that, they)
            - Maintains context across multiple questions
            - Clear memory button to start fresh
            
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
    🧠 RAG Chat System with Memory | Built with Streamlit & LangChain | Follow-up Questions Enabled
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()