import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Indian Economic Research Assistant",
    page_icon="🇮🇳",
    layout="wide",
)

st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #0f2744, #1a3a5c);
        border: 1px solid #1d4ed855;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
    }
    .header-box h1 { color: #f0f6ff; font-size: 1.7rem; margin: 0; }
    .header-box p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── API Keys from .env only ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN     = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it and restart.")
    st.stop()

# ── Functions ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore(chunks, embeddings):
    from langchain_chroma import Chroma
    return Chroma.from_documents(chunks, embeddings)

def process_pdfs(uploaded_files):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source_name"] = f.name
        all_docs.extend(chunks)
        os.unlink(tmp_path)

    return all_docs

def ask_question(question, vectorstore):
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert Indian Economic Research Assistant helping M.A. Economics students.

You specialize in:
- RBI monetary policy, repo rates, CRR, SLR, inflation targeting
- Union Budget: fiscal deficit, revenue expenditure, capital expenditure, key schemes
- Economic Survey: GDP growth projections, sectoral analysis
- Macroeconomic indicators: CPI, WPI, IIP, CAD, forex reserves

Use the document excerpts below to answer the question.
Structure your answer with:
1. Direct answer
2. Key data points / figures (use Rs. for rupee amounts)
3. Key Takeaway at the end

If documents don't contain the answer, say so and give general knowledge.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = []
    seen = set()
    for doc in docs:
        key = (doc.metadata.get("source_name", "?"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source_name", "Unknown"),
                "page":   doc.metadata.get("page", "?"),
                "snippet": doc.page_content,
            })

    return answer, sources

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("vectorstore", None),
    ("doc_stats", {}),
    ("processed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>Indian Economic Research Assistant</h1>
  <p>RAG-powered · RBI Reports · Union Budget · Economic Surveys · Groq LLaMA3 + HuggingFace</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.success("API Keys loaded from .env")

    st.divider()
    st.markdown("## Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        for f in uploaded_files:
            st.markdown(f"📄 `{f.name}`")

    process_btn = st.button(
        "Process Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_btn:
        with st.spinner("Reading and embedding documents..."):
            try:
                chunks = process_pdfs(uploaded_files)
                embeddings = load_embeddings()
                vs = get_vectorstore(chunks, embeddings)

                st.session_state.vectorstore = vs
                st.session_state.processed = True
                st.session_state.messages = []

                doc_names = list({c.metadata.get("source_name", "unknown") for c in chunks})
                st.session_state.doc_stats = {
                    "chunks": len(chunks),
                    "docs":   len(doc_names),
                    "names":  doc_names,
                }
                st.success(f"{len(chunks)} chunks from {len(doc_names)} document(s) ready!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.processed and st.session_state.doc_stats:
        st.divider()
        st.markdown("## Indexed Documents")
        stats = st.session_state.doc_stats
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["docs"])
        col2.metric("Chunks", stats["chunks"])
        for name in stats["names"]:
            st.markdown(f"✅ `{name}`")

    st.divider()
    st.markdown("## Quick Questions")

    quick_qs = {
        "Inflation trends":    "Summarize the inflation trends. Include CPI and WPI data if available.",
        "Repo rate policy":    "What does the document say about the repo rate and RBI monetary policy stance?",
        "Budget breakdown":    "Compare revenue expenditure and capital expenditure. What are the major allocations?",
        "GDP projections":     "What are the GDP growth rate projections? Compare with previous year if possible.",
        "Fiscal deficit":      "Summarize the fiscal deficit figures and targets as % of GDP.",
        "Capital expenditure": "What is mentioned about capital expenditure and infrastructure spending?",
    }

    for label, question in quick_qs.items():
        if st.button(label, key=f"q_{label}", use_container_width=True):
            st.session_state["_pending"] = question

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_about = st.tabs(["Research Chat", "How it works"])

with tab_chat:
    if not st.session_state.processed:
        st.info("Upload PDFs from the sidebar and click Process Documents to begin.")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**RBI Reports**\n\nAnnual Report, Monetary Policy Report, Financial Stability Report")
        c2.markdown("**Union Budget**\n\nBudget Speech, Budget at a Glance, Expenditure Profile")
        c3.markdown("**Economic Survey**\n\nVol 1 & 2, Statistical Appendix, Sectoral chapters")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "user" else "🇮🇳"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.caption(f"{s['source']} — page {s['page']}")
                            st.text(s["snippet"][:300] + "...")

        pending = st.session_state.pop("_pending", None)
        prompt  = pending or st.chat_input("Ask about inflation, repo rates, GDP, fiscal deficit...")

        if prompt:
            with st.chat_message("user", avatar="🎓"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant", avatar="🇮🇳"):
                with st.spinner("Searching documents..."):
                    try:
                        answer, sources = ask_question(prompt, st.session_state.vectorstore)
                        st.markdown(answer)
                        if sources:
                            with st.expander("Sources"):
                                for s in sources:
                                    st.caption(f"{s['source']} — page {s['page']}")
                                    st.text(s["snippet"][:300] + "...")
                        st.session_state.messages.append({
                            "role": "assistant", "content": answer, "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab_about:
    st.markdown("""
## How RAG works in this app

RAG = Retrieval-Augmented Generation. The AI reads YOUR documents and answers from real data.

### Pipeline
1. **Upload** — RBI reports, budget PDFs, economic surveys
2. **Chunking** — Split into 1000-character chunks
3. **Embeddings** — HuggingFace all-MiniLM-L6-v2 converts text to vectors
4. **ChromaDB** — Vectors stored for fast similarity search
5. **Retrieval** — Top 5 most relevant chunks fetched for your question
6. **Generation** — Groq LLaMA3 reads chunks + economics prompt and gives answer
7. **Citations** — Source document and page number shown

### Tech Stack
| Component | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| LLM | Groq LLaMA3-8B |
| RAG Framework | LangChain |
| PDF Reading | PyPDF |
""")