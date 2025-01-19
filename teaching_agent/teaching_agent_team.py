#Let's set up our imports:
import streamlit as st
from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
from composio_phidata import Action, ComposioToolSet
import os
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.utils.pprint import pprint_run_response
from phi.tools.serpapi_tools import SerpApiTools


#Initialize session state and API configuration:
st.set_page_config(page_title = 'Teaching Agent with AI', page_icon = '🤖')


if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = '' 
if 'composio_api_key' not in st.session_state:
    st.session_state.composio_api_key = ''                     
if 'serpapi_api_key' not in st.session_state:
    st.session_state.serpapi_api_key = ''

#Set up Composio tools:
#Initialize the ComposioToolSet class with the API key:
composio_toolset = ComposioToolSet(api_key=st.session_state['composio_api_key'])
google_docs_tool = composio_toolset.get_tools(actions=[Action.GOOGLEDOCS_CREATE_DOCUMENT])[0]
google_docs_tool_update = composio_toolset.get_tools(actions=[Action.GOOGLEDOCS_UPDATE_EXISTING_DOCUMENT])[0]  

#Create the Professor Agent
professor_agent = Agent(name='Research Associate',
                        role='Research Specialist',
                        model=OpenAIChat(id="gpt-4o-mini"),
                        tools=[google_docs_tool],
                        instructions=[
        "Create comprehensive knowledge base",
        "Explain from first principles",
        "Include key terminology",
        "Create formatted Google Doc"
    ]
)

#Create the Academic Advisor Agent:
academic_advisor_agent = Agent(
    name='Academic Advisor',
    role='Academic Advisor',
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[google_docs_tool_update],
    instructions=[
        "Create detailed learning roadmap",
        "Break down into subtopics",
        "Include time estimates",
        "Create formatted Google Doc"
    ]
)

#Create the Research Librarian Agent:
research_librarian_agent = Agent(
    name='Research Librarian',
    role='Research Librarian',
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[google_docs_tool,
           SerpApiTools(api_key=st.session_state['serpapi_api_key'])],
    instructions=[
        "Find quality learning resources",
        "Use SerpApi for searches",
        "Include various resource types",
        "Create curated Google Doc"
    ]
)

#Create the Teaching Assistant Agent:
teaching_assistant_agent = Agent(
    name='Teaching Assistant',
    role='Teaching Assistant',
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[google_docs_tool,SerpApiTools()],
    instructions=[
        "Create practice materials",
        "Include progressive exercises",
        "Add real-world applications",
        "Create Google Doc with solutions"
    ]
)

#Set up Streamlit interface:
with st.sidebar:
    st.title('Teaching Agent Team')
    st.write('Configure API keys')
    st.session_state['openai_api_key']=st.text_input('OpenAI API Key',type='password')
    st.session_state['composio_api_key']=st.text_input('Composio API Key',type='password')
    st.session_state['serpapi_api_key']=st.text_input('SerpApi API Key',type='password')


#Implement topic processing:
topic = st.text_input('Enter a topic:', 'e.g., Quantum Computing')  
if st.button("Start"):
    with st.spinner("Generating Knowledge Base..."):
        professor_response = professor_agent.run(
            f"topic: {topic}"
        )
professor_doc_link = "afi"

#Extract Google Doc links:        
def extract_google_doc_link(response_content):
    if "https://docs.google.com" in response_content:
        return response_content.split("https://docs.google.com")[1].split("")[0]
    return None

st.markdown('### Google Doc Links:')
if professor_doc_link:
    st.markdown(f"- **Professor Document:** [View](https://docs.google.com{professor_doc_link})")
                

#Display agent responses:
st.markdown('### Professor Response:')
st.markdown(professor_response.content)
pprint_run_response(professor_response)     

#Error handling and validation:
if not st.session_state['openai_api_key'] or not st.session_state['composio_api_key'] :
    st.error("Please enter all required API keys.")
    st.stop()

try:
   composio_toolset = ComposioToolSet(api_key=st.session_state['composio_api_key'])
except Exception as e:
    st.error(f"Error initializing Composio: {e}")
    st.stop()       

#Progress tracking:
with st.spinner("Creating Practice Material..."):
    teaching_assistant_response = teaching_assistant_agent.run(f"topic: {topic}") 
    st.success("Practice Material Created!")