import streamlit as st
import tempfile
from agent import create_agent_graph
from tools.rag import initialize_rag
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="RAG AI — Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0e1117 50%, #0d1321 100%);
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }

    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        backdrop-filter: blur(20px);
        padding: 20px;
        margin-bottom: 16px;
    }

    .stChatInputContainer { padding-bottom: 20px; }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
        max-width: 85%;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="chatAvatarIcon-user"] { background: linear-gradient(135deg, #007CF0, #00A3FF); }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        margin-left: auto;
        background: linear-gradient(135deg, rgba(0, 124, 240, 0.1), rgba(0, 163, 255, 0.05));
        border: 1px solid rgba(0, 124, 240, 0.2);
        border-bottom-right-radius: 4px;
    }

    [data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg, #00DFD8, #007CF0); }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-right: auto;
        background: linear-gradient(135deg, rgba(0, 223, 216, 0.05), rgba(0, 124, 240, 0.03));
        border: 1px solid rgba(0, 223, 216, 0.15);
        border-bottom-left-radius: 4px;
    }

    h1 {
        background: linear-gradient(135deg, #007CF0 0%, #00DFD8 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #6b7280;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    .sidebar .sidebar-content { background: rgba(15, 17, 23, 0.95); }

    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-supervisor { background: rgba(168, 85, 247, 0.2); color: #A855F7; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-document { background: rgba(0, 124, 240, 0.2); color: #007CF0; border: 1px solid rgba(0, 124, 240, 0.3); }
    .badge-research { background: rgba(0, 223, 216, 0.2); color: #00DFD8; border: 1px solid rgba(0, 223, 216, 0.3); }
    .badge-data { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #007CF0, #00DFD8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; }

    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("RAG AI — Multi-Agent System")
st.markdown("### Intelligent Multi-Agent Assistant with MCP Protocol")

with st.sidebar:
    st.markdown("## Agent Configuration")

    agent_mode = st.selectbox(
        "Agent Mode",
        ["Auto (Supervisor)", "Document Agent", "Research Agent", "Data Agent"],
        index=0,
        help="Auto mode intelligently routes your query to the best specialist agent."
    )

    st.markdown("---")

    st.markdown("## 📁 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDFs to enable document Q&A with the Document Agent."
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            temp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)

            with st.spinner(f"Indexing {len(uploaded_files)} document(s)..."):
                try:
                    msg = initialize_rag(temp_paths)
                    st.success(msg)
                except Exception as e:
                    st.error(f"Error indexing PDF: {e}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_traces", None)
        st.session_state.pop("last_route", None)
        st.session_state.pop("last_agent", None)
        st.rerun()

    st.markdown("---")

    st.markdown("## System Status")
    status_cols = st.columns(2)
    with status_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Agents</div>
        </div>
        """, unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">7</div>
            <div class="metric-label">Tools</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info(
        "**Model:** Qwen 2.5-7B\n\n"
        "**Embeddings:** EmbeddingGemma-300m\n\n"
        "**Vector DB:** FAISS\n\n"
        "**Protocol:** MCP v1.0"
    )

    with st.expander("Agent Capabilities"):
        st.markdown("""
        **Supervisor** — Routes queries to the best specialist

        **Document Agent** — PDF Q&A with RAG retrieval

        **Research Agent** — Web search, page reading, summarization

        **Data Agent** — Math computation, code analysis
        """)

    with st.expander("Available Tools"):
        st.markdown("""
        1. `retrieve_context` — RAG document search
        2. `fetch_weather` — Real-time weather data
        3. `web_search` — DuckDuckGo web search
        4. `read_webpage` — URL content extraction
        5. `summarize_text` — LLM text summarization
        6. `calculate` — Safe math evaluation
        7. `analyze_code` — Python code analysis
        """)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    try:
        with st.spinner("Initializing Multi-Agent System..."):
            st.session_state.agent = create_agent_graph()
    except Exception as e:
        st.error(f"Failed to initialize agent system: {e}")

if "last_agent" in st.session_state and st.session_state.last_agent:
    agent_name = st.session_state.last_agent
    route = st.session_state.get("last_route", "")

    badge_class = "badge-supervisor"
    if "document" in agent_name:
        badge_class = "badge-document"
    elif "research" in agent_name:
        badge_class = "badge-research"
    elif "data" in agent_name:
        badge_class = "badge-data"

    st.markdown(f"""
    <div class="glass-panel" style="padding: 12px 20px; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 0.85rem; color: #6b7280;">Last active:</span>
        <span class="agent-badge {badge_class}">{agent_name}</span>
        <span style="font-size: 0.8rem; color: #4b5563;">Route: {route}</span>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

if "last_traces" in st.session_state and st.session_state.last_traces:
    with st.expander("Tool Execution Traces", expanded=False):
        traces = st.session_state.last_traces
        for agent_name, agent_traces in traces.items():
            if agent_traces:
                st.markdown(f"**{agent_name}**")
                for trace in agent_traces:
                    status_icon = "✅" if trace.get("status") == "success" else "⚠️"
                    latency = trace.get("latency_ms", 0)
                    st.markdown(
                        f"  {status_icon} `{trace.get('tool', 'unknown')}` "
                        f"— {latency:.0f}ms"
                    )
                    if trace.get("result_preview"):
                        st.caption(trace["result_preview"][:150])

if prompt := st.chat_input("Ask anything — documents, web search, math, code, weather..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "agent" in st.session_state:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            status_container = st.empty()
            status_container.markdown(
                '<div class="pulse" style="color: #00DFD8; font-size: 0.85rem;">'
                '⚡ Multi-agent system processing...</div>',
                unsafe_allow_html=True,
            )

            try:
                history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))

                agent = st.session_state.agent

                if agent_mode == "Document Agent":
                    result = agent.document_agent.invoke(history)
                    result["_active_agent"] = "document_agent"
                    result["_route"] = "document (forced)"
                elif agent_mode == "Research Agent":
                    result = agent.research_agent.invoke(history)
                    result["_active_agent"] = "research_agent"
                    result["_route"] = "research (forced)"
                elif agent_mode == "Data Agent":
                    result = agent.data_agent.invoke(history)
                    result["_active_agent"] = "data_agent"
                    result["_route"] = "data (forced)"
                else:
                    result = agent.invoke({"messages": history})

                last_msg = result["messages"][-1]
                response_text = last_msg.content

                status_container.empty()
                message_placeholder.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                st.session_state.last_agent = result.get("_active_agent", "supervisor")
                st.session_state.last_route = result.get("_route", "auto")
                st.session_state.last_traces = agent.get_all_traces()

            except Exception as e:
                status_container.empty()
                st.error(f"Error during execution: {e}")
