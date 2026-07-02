import os
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Enterprise HR Auditor", page_icon="🛡️", layout="wide")

# ==========================================
# CUSTOM CSS 
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        background-image: radial-gradient(#cbd5e1 1px, transparent 1px) !important;
        background-size: 24px 24px !important;
    }
    
    [data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottom"] > div { 
        background-color: transparent !important; 
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        box-shadow: 4px 0 20px rgba(15, 23, 42, 0.15) !important;
    }
    [data-testid="stSidebar"] > div:first-child::before {
        content: ""; display: block; height: 4px; background: linear-gradient(90deg, #6366f1, #0ea5e9, #6366f1) !important;
    }
    
    /* Sidebar Headers and Labels */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { 
        color: #f1f5f9 !important; 
    }
    
    /* Fix Input Boxes (Dark background, White text) */
    [data-testid="stSidebar"] .stTextInput > div > div, 
    [data-testid="stSidebar"] .stTextArea > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] input, 
    [data-testid="stSidebar"] textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Fix File Uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
        background-color: #1e293b !important;
        border: 1px dashed #475569 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
        color: #cbd5e1 !important;
    }

    /* Chat Bubbles */
    .stChatMessage { 
        background-color: rgba(255, 255, 255, 0.95) !important; 
        border-radius: 14px !important; 
        border: 1px solid #cbd5e1 !important; 
        padding: 20px !important; 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important; 
        margin-bottom: 15px !important; 
    }
    .stChatMessage * { color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DYNAMIC BACKEND PROCESSING
# ==========================================
@st.cache_resource
def load_models():
    """Load the LLM and NLI models once to save time."""
    llm = ChatOllama(model="llama3", temperature=0)
   
    nli = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return llm, nli, embeddings

local_llm, nli_model, local_embeddings = load_models()

def process_uploaded_pdf(uploaded_file):
    """Saves the uploaded file to a temp directory and builds the Vector DB."""
    with st.spinner("Ingesting corporate policies... building vector database..."):
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        # Use an ephemeral Chroma store (no persist directory) so it resets on new uploads
        vector_db = Chroma.from_documents(documents=chunks, embedding=local_embeddings)
        os.unlink(tmp_file_path) # Clean up temp file
        
        return vector_db.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# SIDEBAR: ENTERPRISE CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 70px; margin-bottom: 0;'>🛡️</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>HR Auditor Pro</h3>", unsafe_allow_html=True)
    st.divider()
    
    st.title("1. Enterprise Setup")
    company_name = st.text_input("Company Name:", value="Yeshiva University")
    emp_name = st.text_input("Employee Name:", value="Vishal")
    
    st.title("2. Compliance Rules")
    st.caption("What rules MUST the L1 Critic inject into every positive response?")
    custom_rule_1 = st.text_area("Rule 1:", value="Remote work must be approved by the Department Head.")
    custom_rule_2 = st.text_area("Rule 2:", value="Employees must submit the official FWA Request Form via the HR Portal.")
    
    st.title("3. Upload Document")
    uploaded_pdf = st.file_uploader("Upload Handbook (PDF)", type="pdf")

# ==========================================
# MAIN INTERFACE LOGIC
# ==========================================
st.title("🛡️ Enterprise Compliance Auditor")
st.markdown("**Agnostic Dual-Layer RAG Architecture**")

if uploaded_pdf is None:
    # Beautiful empty state before upload
    st.info("👈 Please upload a PDF Handbook in the sidebar to initialize the system.")
    st.markdown("""
        <div style='text-align: center; padding: 50px; color: #64748b;'>
            <h2>No Active Workspace</h2>
            <p>Upload a corporate handbook to begin the compliance audit.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Initialize the retriever only if a file is uploaded
    if "retriever" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_pdf.name:
        st.session_state.retriever = process_uploaded_pdf(uploaded_pdf)
        st.session_state.last_uploaded = uploaded_pdf.name
        st.session_state.messages = [] # Clear chat on new upload
        st.toast(f"Successfully processed {uploaded_pdf.name}!", icon="✅") # Sleek popup notification

    retriever = st.session_state.retriever

    # Setup the pipelines with dynamic company name
    template = f"""You are a Specialized HR Flexible Work Coordinator for {company_name}. 
    Your ONLY job is to answer questions based ONLY on the context provided.
    If the information is not in the context, strictly state: "I do not have information regarding this specific policy."
    Context:\n{{context}}\nQuestion: {{question}}\nHelpful Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | local_llm | StrOutputParser())

    def run_l1_critic(query, initial_answer, context):
        if "do not have information" in initial_answer.lower():
            return f"Dear {emp_name},\n\nI'm sorry, but the {company_name} Employee Handbook does not explicitly cover policies regarding this specific scenario. Please reach out to your HR representative.\n\nBest regards, {company_name} HR Auditor"

        critic_template = f"""
        You are an HR Compliance Auditor for {company_name}. 
        Original Question: {{query}}
        Draft Answer: {{initial_answer}}

        Your job is to REWRITE the Draft Answer into a final, polite HR response to {emp_name}. 
        You MUST start the response with "Dear {emp_name}," and sign off as "Best regards, {company_name} HR Auditor".

        Explicitly include these mandatory rules in the body:
        1. {custom_rule_1}
        2. {custom_rule_2}

        CRITICAL: Output ONLY the final letter. Do not include any conversational filler.
        Final Rewritten Answer:
        """
        critic_prompt = ChatPromptTemplate.from_template(critic_template)
        critic_chain = critic_prompt | local_llm | StrOutputParser()
        return critic_chain.invoke({"query": query, "initial_answer": initial_answer, "context": context})

    def run_l2_nli_check(context, final_answer):
        if "does not explicitly cover policies" in final_answer:
            return "⚠️ L2 WARNING: Out of Bounds. The requested topic is not present in the verified document."

        scores = nli_model.predict([(context, final_answer)])
        contradiction_score = scores[0][0]
        entailment_score = scores[0][1]
        
        if contradiction_score > entailment_score:
            return f"⛔ L2 CRITICAL: Hallucination Detected! The generated answer logically contradicts the {company_name} document."
        else:
            return f"✅ L2 VERIFIED: The answer is logically supported by the {company_name} document."

    # --- THE UI UPGRADES ---
    
    # 1. The Welcome Dashboard (Only shows if chat is empty)
    if len(st.session_state.messages) == 0:
        st.markdown(f"""
            <div style='background-color: #f1f5f9; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 20px;'>
                <h3 style='margin-top: 0; color: #1e293b;'>Welcome to the {company_name} Auditor</h3>
                <p style='color: #475569;'>The system has successfully ingested the handbook. You can now ask compliance questions, and the Dual-Layer pipeline will enforce your custom rules.</p>
            </div>
        """, unsafe_allow_html=True)

    # 2. Custom Avatars in Chat History
    for message in st.session_state.messages:
        avatar_icon = "🧑‍💼" if message["role"] == "user" else "🛡️"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    # 3. Chat Input
    if user_query := st.chat_input(f"Ask your HR question about {company_name} here..."):
        
        st.chat_message("user", avatar="🧑‍💼").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner(f"Scanning {company_name} policies..."):
                context_docs = retriever.invoke(user_query)
                formatted_context = format_docs(context_docs)
                
                initial_ans = rag_chain.invoke(user_query)
                l1_verified_ans = run_l1_critic(user_query, initial_ans, formatted_context)
                l2_status = run_l2_nli_check(formatted_context, l1_verified_ans)
                
                st.markdown(f"{l1_verified_ans}")
                
                with st.expander("🔍 View L2 Verification Status"):
                    if "VERIFIED" in l2_status:
                        st.success(l2_status)
                        st.toast(f"L2 Check Passed for {emp_name}", icon="✅") # Toast popup!
                    else:
                        st.warning(l2_status)
                        st.toast("L2 Warning Triggered!", icon="⚠️")

                st.session_state.messages.append({"role": "assistant", "content": f"{l1_verified_ans}"})