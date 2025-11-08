import streamlit as st
from backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ================================= Utilities =================================

def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["chat_titles"][thread_id] = "Untitled Chat"

def add_thread(thread_id, title=None):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    if title and thread_id not in st.session_state["chat_titles"]:
        st.session_state["chat_titles"][thread_id] = title

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# =========================== Session Initialization ===========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads() or []

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

add_thread(st.session_state["thread_id"])

# =============================== Custom CSS ==================================

st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at 30% 30%, #0d1b2a, #000000);
            color: #e0e0e0;
            font-family: 'Orbitron', sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #001f3f, #000);
            border-right: 2px solid #00aaff;
        }
        .css-1d391kg, .css-qrbaxs {
            color: #00bfff !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #0077ff, #00ffff);
            color: black;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 10px;
            font-weight: bold;
            box-shadow: 0px 0px 10px #00ffffaa;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #00ffff, #0077ff);
        }
        div[data-testid="stChatMessageContent"] {
            background-color: rgba(0, 60, 100, 0.25);
            border: 1px solid #00bfff66;
            border-radius: 10px;
            padding: 10px;
            margin: 8px 0;
        }
        .stTextInput > div > div > input {
            background-color: #001a33;
            color: #00ffff;
        }
        h1 {
            color: #00bfff;
            text-shadow: 0 0 10px #0077ff;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# =============================== Sidebar UI ==================================

st.sidebar.title("🤖 LangGraph AI Console")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.markdown("---")

if st.sidebar.button("✨ New Chat", key="new_chat_button"):
    reset_chat()

st.sidebar.header("🧠 My Conversations")

for idx, thread_id in enumerate(st.session_state["chat_threads"][::-1]):
    title = st.session_state["chat_titles"].get(thread_id, str(thread_id))
    if st.sidebar.button(f"💠 {title}", key=f"chat_{idx}_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                temp_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                temp_messages.append({"role": "assistant", "content": f"🔧 Tool used: {msg.name}"})
        st.session_state["message_history"] = temp_messages

# =============================== Main UI =====================================

st.title("🤖 AI Assistant v2.0")
st.markdown("### Blue Horizon Neural Chat Interface")

# Render chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("💬 Type your message...")

if user_input:
    # Append user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Auto-name new chat
    if len(st.session_state["message_history"]) == 1:
        preview_text = user_input[:30] + ("..." if len(user_input) > 30 else "")
        st.session_state["chat_titles"][st.session_state["thread_id"]] = preview_text

    # Assistant streaming with tool status visualization
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Show tool execution status
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "Tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream AI output
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Close tool status box if used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save AI message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
