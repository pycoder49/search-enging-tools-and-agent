from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import streamlit as st


# creating tools
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2, doc_content_chars_max=1000,
)
arxiv_tool = ArxivQueryRun(arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2, doc_content_chars_max=1000,
)
wiki_tool = WikipediaQueryRun(wiki_wrapper)

ddg_search = DuckDuckGoSearchRun(name="DuckDuckGo Search")


# streamlit app
st.title("Search Engine Tools and Agent with GROQ")
# StreamlitCallbackHandler lets you see the thoughts/actoins of the agent in real-time

# side bar for settings
st.sidebar.title("Configurations")
groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "How can I assist you today?",
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

