import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

# FIXED IMPORTS - Updated for latest LangChain
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    from langchain.memory import ChatMessageHistory  # Fallback

from huggingface_hub import InferenceClient
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Page config
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #000000 !important;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        color: #000000 !important;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
        color: #000000 !important;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #E8F5E8;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for all variables
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'demo_loaded' not in st.session_state:
    st.session_state.demo_loaded = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False


# Demo pdf path
_demo_pdf_path = Path(__file__).parent / "demo_pdf.pdf"


def initialize_hf_client(api_token):
    """Initialize HuggingFace client with error handling"""
    try:
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = api_token
        client = InferenceClient(api_key=api_token)
        
        # Test client
        _ = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=5
        )
        st.session_state.client = client
        return True
    except Exception as e:
        st.error(f"Failed to initialize HuggingFace client: {e}")
        st.info("💡 Make sure your API token is valid and has proper permissions")
        return False


def process_pdf(pdf_file):
    """Process uploaded PDF and create vector database"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save file
        status_text.text("📄 Saving PDF file...")
        progress_bar.progress(10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Step 2: Load PDF
        status_text.text("📖 Loading PDF content...")
        progress_bar.progress(25)
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        if not documents:
            st.error("❌ No content found in PDF. Please check the file.")
            return False
        
        # Step 3: Split text
        status_text.text("✂️ Splitting text into chunks...")
        progress_bar.progress(40)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            st.error("❌ No text chunks created. PDF might be empty or corrupted.")
            return False
        
        # Step 4: Create embeddings
        status_text.text("🧠 Creating embeddings (this may take a moment)...")
        progress_bar.progress(60)
        
        # Use a more lightweight embedding model for better compatibility
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # Force CPU to avoid GPU issues
            )
        except Exception as e:
            st.error(f"❌ Error creating embeddings: {e}")
            st.info("💡 Try restarting the application or using a different environment")
            return False
        
        # Step 5: Create vector store
        status_text.text("💾 Building vector database...")
        progress_bar.progress(80)
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Step 6: Initialize memory
        status_text.text("🧠 Setting up conversation memory...")
        progress_bar.progress(90)
        
        try:
            chat_history = ChatMessageHistory()
            memory = ConversationBufferMemory(
                chat_memory=chat_history,
                memory_key="chat_history",
                return_messages=True
            )
        except Exception as e:
            # Fallback memory initialization
            st.warning(f"Memory initialization warning: {e}")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Step 7: Store metadata
        metadata = {
            'total_chunks': len(texts),
            'total_pages': len(documents),
            'pdf_name': pdf_file.name,
            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Clean up
        progress_bar.progress(100)
        status_text.text("✅ PDF processed successfully!")
        
        # Store everything in session_state
        st.session_state.vectorstore = vectorstore
        st.session_state.memory = memory
        st.session_state.metadata = metadata
        st.session_state.pdf_processed = True
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        st.info("💡 Troubleshooting tips:\n- Make sure the PDF is not password protected\n- Try a smaller PDF file\n- Restart the application")
        return False
    finally:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()


def ask_question(question, num_results=5):
    """Ask question and get answer with memory"""
    vectorstore = st.session_state.vectorstore
    memory = st.session_state.memory
    client = st.session_state.client
    
    if not vectorstore or not memory:
        return "❌ System not initialized. Please upload and process a PDF first.", []
    
    try:
        # Retrieve relevant documents
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=num_results)
        
        # Prepare context
        context_parts = []
        retrieved_pages = []
        for doc, score in docs_with_scores:
            page = doc.metadata.get('page', 'Unknown')
            retrieved_pages.append(f"Page {page} (Relevance: {score:.3f})")
            context_parts.append(f"[Page {page}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Get conversation history
        try:
            chat_history = memory.chat_memory.messages if hasattr(memory, 'chat_memory') and memory.chat_memory.messages else []
            history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-4:]])
        except:
            history_text = ""
        
        # Generate answer
        if client:
            try:
                messages = [
                    {
                        "role": "system", 
                        "content": f"You are a helpful assistant. Answer questions based on the provided document context and conversation history. Always mention page numbers when possible. Keep answers concise but comprehensive.\n\nConversation History:\n{history_text}"
                    },
                    {
                        "role": "user", 
                        "content": f"Document Context:\n{context[:3000]}\n\nCurrent Question: {question}\n\nPlease provide a clear answer:"
                    }
                ]
                
                response = client.chat_completion(
                    messages=messages,
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    max_tokens=400,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content
                
            except Exception as e:
                answer = f"⚠️ AI generation error: {e}\n\n📝 **Fallback Answer based on document content:**\n\n{context[:1500]}..."
        
        else:
            answer = f"📝 **Based on your question, here are the most relevant sections from the document:**\n\n{context[:1500]}..."
        
        # Save to memory
        try:
            if hasattr(memory, 'chat_memory'):
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
            else:
                # Alternative memory saving method
                memory.save_context({"input": question}, {"output": answer})
        except Exception as e:
            st.warning(f"Memory save warning: {e}")
        
        return answer, retrieved_pages
        
    except Exception as e:
        error_msg = f"❌ Error processing question: {e}"
        return error_msg, []


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 SmartPDF AI: RAG-Based Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown(
    "<div style='text-align: center; font-size:18px;'>"
    "Upload a PDF document and ask questions about its content with AI-powered responses!"
    "</div>",
    unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Token Configuration
        st.subheader("🔑 HuggingFace API Setup")
        
        # Help for getting API token
        with st.expander("ℹ️ How to get API token"):
            st.markdown("""
            1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
            2. Click "New token"
            3. Choose "Read" permission
            4. Copy the token and paste it below
            """)
        
        api_token = st.text_input(
            "Enter your HF API token:", 
            type="password", 
            placeholder="hf_...",
            help="Required for AI-powered responses"
        )
        
        if st.button("🔗 Configure API", type="primary"):
            if api_token and api_token.startswith('hf_'):
                with st.spinner("Testing API connection..."):
                    if initialize_hf_client(api_token):
                        st.session_state.api_configured = True
                        st.success("✅ API configured successfully!")
                        st.balloons()
                    else:
                        st.session_state.api_configured = False
            else:
                st.error("❌ Please enter a valid HuggingFace token (starts with 'hf_')")
        
        # Show API status
        if st.session_state.api_configured:
            st.success("🟢 API Status: Connected")
        else:
            st.error("🔴 API Status: Not Connected")
        
        st.markdown("---")
        
        # PDF Document Selection
        st.subheader("📄 PDF Document Selection")
        
        pdf_source = st.radio(
            "Choose a PDF:",
            ("Use Demo PDF", "Upload Your PDF")
        )
        
        uploaded_file = None
        if pdf_source == "Upload Your PDF":
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type="pdf",
                help="Supported: PDF files up to 200MB"
            )
        
        # Handle demo PDF loading
        if pdf_source == "Use Demo PDF" and not st.session_state.demo_loaded:
            if st.button("🚀 Load Demo PDF", type="primary"):
                try:
                    if _demo_pdf_path.exists():
                        with open(_demo_pdf_path, 'rb') as f:
                            class DemoPDF:
                                name = "Comprehensive Guide for College Courses for New Students in Tamilnadu."
                                def getvalue(self):
                                    return f.read()
                            demo_pdf_file = DemoPDF()
                            if process_pdf(demo_pdf_file):
                                st.markdown('<div class="success-box">✅ Demo PDF loaded successfully! You can now ask questions.</div>', unsafe_allow_html=True)
                                st.session_state.demo_loaded = True
                                
                                # Display metadata for demo
                                if st.session_state.metadata:
                                    meta = st.session_state.metadata
                                    st.markdown(f"""
                                    **📊 Document Statistics:**
                                    - **Pages:** {meta['total_pages']}
                                    - **Text Chunks:** {meta['total_chunks']}
                                    - **Processed:** {meta['processed_at']}
                                    """)
                            else:
                                st.error("❌ Failed to load demo PDF.")
                    else:
                        st.error("❌ Demo PDF file missing in app folder.")
                except Exception as e:
                    st.error(f"❌ Error loading demo PDF: {e}")
        elif pdf_source == "Use Demo PDF" and st.session_state.demo_loaded:
            st.success("✅ Demo PDF already loaded!")
        
        # Handle uploaded PDF
        if uploaded_file:
            st.info(f"📁 Selected: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
            
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF... This may take a few moments"):
                    if process_pdf(uploaded_file):
                        st.markdown('<div class="success-box">✅ PDF processed successfully! You can now ask questions.</div>', unsafe_allow_html=True)
                        
                        # Display metadata
                        if st.session_state.metadata:
                            meta = st.session_state.metadata
                            st.markdown(f"""
                            **📊 Document Statistics:**
                            - **Pages:** {meta['total_pages']}
                            - **Text Chunks:** {meta['total_chunks']}
                            - **Processed:** {meta['processed_at']}
                            """)
                    else:
                        st.error("❌ Failed to process PDF. Please try again.")
        
        st.markdown("---")
        
        # Memory Management
        st.subheader("🧠 Conversation Memory")
        if st.session_state.memory:
            try:
                msg_count = len(st.session_state.memory.chat_memory.messages) if hasattr(st.session_state.memory, 'chat_memory') else 0
                st.metric("Messages Stored", msg_count)
            except:
                st.write("Memory active")
            
            if st.button("🗑️ Clear Memory"):
                try:
                    st.session_state.memory.clear()
                    st.session_state.chat_history = []
                    st.success("🧹 Memory cleared!")
                    st.rerun()
                except Exception as e:
                    st.warning(f"Memory clear warning: {e}")
        else:
            st.write("⭕ No active memory session")
            
        st.markdown(
            "<div style='text-align: center; font-size: medium; color: gray; margin-top: 10px;'>"
            "Created by Sankaran S | "
            "<a href='https://github.com/sankaran-s2001' target='_blank' style='color: gray; text-decoration: underline;'>🔗 GitHub</a>"
            "</div>", 
            unsafe_allow_html=True)

    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions About Your PDF")
        
        # Status checks
        ready_to_use = st.session_state.api_configured and st.session_state.vectorstore
        
        if not st.session_state.api_configured:
            st.markdown('<div class="warning-box">⚠️ Please configure your HuggingFace API token in the sidebar</div>', unsafe_allow_html=True)
        
        if not st.session_state.vectorstore:
            st.markdown('<div class="warning-box">⚠️ Please upload and process a PDF document first</div>', unsafe_allow_html=True)
        
        if ready_to_use:
            st.markdown('<div class="success-box">✅ System ready! Ask your questions below</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "💭 Your question:", 
            placeholder="e.g., What are the main topics discussed in this document?",
            disabled=not ready_to_use
        )
        
        # Buttons
        col_ask, col_examples = st.columns([1, 1])

        with col_ask:
            ask_button = st.button(
                "🚀 Ask Question", 
                disabled=not ready_to_use or not question.strip(),
                type="primary"
            )

        with col_examples:
            if st.button("💡 Show Examples"):
                if st.session_state.demo_loaded:
                    examples = [
                        "What is this pdf about?",
                        "Can I study computer science without maths in 12th?",
                        "Which colleges accept 45% marks for BCom?",
                        "What's the cheapest engineering college in Tamil Nadu?",
                        "How long is a BCom course?"
                    ]
                else:
                    examples = [
                        "What is this document about?",
                        "Can you summarize the key points?",
                        "What are the main conclusions?",
                        "List all important dates mentioned",
                        "Explain the methodology used",
                        "What are the recommendations?"
                    ]
                st.markdown("💡 **Example questions:**\n\n" + "\n".join([f"- {q}" for q in examples]))
        
        # Process question
        if ask_button and question:
            with st.spinner("🤖 Generating answer..."):
                answer, retrieved_pages = ask_question(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'retrieved_pages': retrieved_pages,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Clear input
                st.rerun()
        
        # Display latest answer
        if st.session_state.chat_history:
            latest_chat = st.session_state.chat_history[-1]
            st.markdown("### 🤖 Latest Response:")
            st.markdown(f'<div class="chat-message user-message"><strong>❓ You asked:</strong> {latest_chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Assistant:</strong> {latest_chat["answer"]}</div>', unsafe_allow_html=True)
            
            if latest_chat.get('retrieved_pages'):
                with st.expander("📄 View Sources"):
                    st.write("**Retrieved sections:**")
                    for page in latest_chat['retrieved_pages']:
                        st.write(f"• {page}")
    
    with col2:
        st.header("📊 System Dashboard")
        
        # System metrics
        if st.session_state.metadata:
            meta = st.session_state.metadata
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("📄 Pages", meta['total_pages'])
                st.metric("💬 Questions", len(st.session_state.chat_history))
            with col_metric2:
                st.metric("✂️ Chunks", meta['total_chunks'])
                api_status = "🟢 Active" if st.session_state.api_configured else "🔴 Inactive"
                st.metric("🔗 API", api_status)
        
        st.markdown("---")
        
        # Conversation history
        if st.session_state.chat_history:
            st.header("📚 Chat History")
            
            # Show recent conversations
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                chat_num = len(st.session_state.chat_history) - i
                with st.expander(f"💬 Q{chat_num}: {chat['question'][:30]}..."):
                    st.write(f"**🕐 Time:** {chat['timestamp']}")
                    st.write(f"**❓ Question:** {chat['question']}")
                    st.write(f"**🤖 Answer:** {chat['answer'][:200]}...")
            
            # Download chat history
            if st.button("📥 Download Full History"):
                chat_text = f"PDF Q&A Chat History\n{'='*50}\n\n"
                for i, chat in enumerate(st.session_state.chat_history):
                    chat_text += f"Question {i+1} ({chat['timestamp']}):\n{chat['question']}\n\n"
                    chat_text += f"Answer {i+1}:\n{chat['answer']}\n\n"
                    chat_text += "-" * 50 + "\n\n"
                
                st.download_button(
                    label="💾 Download as TXT",
                    data=chat_text,
                    file_name=f"pdf_qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        # Help section
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **🚀 Quick Start:**
            1. Enter your HuggingFace API token
            2. Upload a PDF document or use demo
            3. Wait for processing
            4. Start asking questions!
            
            **💡 Pro Tips:**
            - Ask specific, clear questions
            - Use follow-up questions - I remember context!
            - Try different question types: summaries, lists, explanations
            - Check the sources to verify information
            
            **🔧 Troubleshooting:**
            - Restart if you see import errors
            - Use smaller PDFs if processing fails
            - Make sure your API token is valid
            """)


if __name__ == "__main__":
    main()
