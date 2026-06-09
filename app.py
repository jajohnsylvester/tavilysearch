import os
import asyncio
import streamlit as st
from google.genai import types  
from google.adk.agents import LlmAgent  
from google.adk.models.lite_llm import LiteLlm 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from tavily import TavilyClient  

# -----------------------------------------------------------------------------
# 1. Streamlit App Layout & Configurations
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Kaggle Research Agent", page_icon="🤖", layout="wide")

st.title("🤖 Kaggle Llama 3.2 Research Agent")
st.markdown(
    "This application uses **Google ADK**, **LiteLLM**, and **Tavily Search** "
    "to create a local RAG-capable research agent."
)

# Sidebar Configuration Inputs
st.sidebar.header("Configuration")
ngrok_url = st.sidebar.text_input(
    "Ngrok URL", 
    value="https://evident-lens-surpass.ngrok-free.dev",
    help="Enter your live Kaggle Ngrok address"
)
tavily_key = st.sidebar.text_input(
    "Tavily API Key", 
    type="password",
    value="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"
)

# Apply configurations to environment variables
os.environ["TAVILY_API_KEY"] = tavily_key
os.environ["OPENAI_EXTRA_HEADERS"] = '{"ngrok-skip-browser-warning": "true"}'

# -----------------------------------------------------------------------------
# 2. Native ADK Tool Setup
# -----------------------------------------------------------------------------
def tavily_web_search(query: str) -> str:
    """
    Search the internet for real-time information.
    """
    try:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = client.search(query=query, search_depth="advanced")
        results = response.get("results", [])
        if not results:
            return f"No web search results found for: '{query}'"
            
        formatted_results = []
        for res in results[:3]:  
            formatted_results.append(f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}\n---")
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing Tavily web search: {str(e)}"

# -----------------------------------------------------------------------------
# 3. Asynchronous Execution Logic
# -----------------------------------------------------------------------------
async def run_agent_query(query_text, status_placeholder, response_placeholder):
    status_placeholder.info("🤖 Initializing Session and Runner engines...")
    
    custom_llm = LiteLlm(
        model="openai/llama3.2",
        api_key="not-needed-for-ollama", 
        base_url=f"{ngrok_url.rstrip('/')}/v1"
    )

    research_agent = LlmAgent(
        name="kaggle_researcher",
        model=custom_llm,
        instruction=(
            "You are a helpful research assistant with access to web queries. "
            "When asked questions about recent occurrences, always consult your tavily_web_search tool "
            "first to pull real-time data before formulating a response."
        ),
        tools=[tavily_web_search]  
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="KaggleAgentApp", user_id="streamlit_user", session_id="session_streamlit"
    )
    
    runner = Runner(agent=research_agent, app_name="KaggleAgentApp", session_service=session_service)
    structured_message = types.Content(role='user', parts=[types.Part(text=query_text)])
    
    status_placeholder.info("🚀 Submitting query to remote Kaggle LLM...")
    
    async for event in runner.run_async(user_id="streamlit_user", session_id=session.id, new_message=structured_message):
        if event.get_function_calls():
            status_placeholder.warning("⚙️ [Tool Request]: Llama 3.2 is invoking Tavily Web Search...")
        elif event.get_function_responses():
            status_placeholder.success("📊 [Tool Result Received]: Feeding insights back to Llama 3.2...")
        elif event.is_final_response():
            if event.content and event.content.parts:
                status_placeholder.empty() 
                response_placeholder.markdown("### 📝 Agent Final Response")
                response_placeholder.markdown(event.content.parts[0].text)

# -----------------------------------------------------------------------------
# 4. Streamlit UI Logic Handlers & Async Loop Patch
# -----------------------------------------------------------------------------
query_input = st.text_input(
    "What would you like to research today?", 
    value="What are the latest breakthroughs regarding DeepSeek models this week?"
)

if st.button("Run Research Agent", type="primary"):
    if not tavily_key or not ngrok_url:
        st.error("Please provide both your Ngrok URL and Tavily API key in the sidebar.")
    else:
        status_box = st.empty()
        response_box = st.empty()
        
        # Safe async management for web server threads
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.run_until_complete(run_agent_query(query_input, status_box, response_box))
