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
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2, doc_content_chars_max=1000,
)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

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

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        streaming=True,
    )

    tools = [arxiv_tool, wiki_tool, ddg_search]

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True,
        )

        response = search_agent.run(
            st.session_state.messages,
            callbacks=[
                callback,
            ]
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        st.write(response)


