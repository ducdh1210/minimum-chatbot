"""
FastAPI OpenAI Multi-Agent Chatbot Application

This is a FastAPI application that implements a multi-agent chatbot system using the OpenAI SDK.
Features:
- Router agent that directs queries to specialized agents
- Specialized agents for different tasks (writing and math)
- Session management with proper role handling
- Basic HTML interface for user interaction
- Uses OpenAI's API for generating responses
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import os
import uuid
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="OpenAI Multi-Agent Chatbot")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Create Pydantic models for request/response validation
class Message(BaseModel):
    # OpenAI uses "system", "user", and "assistant" roles
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSession(BaseModel):
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    current_agent: str = "Router"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent: str


class AgentMetadata(BaseModel):
    name: str
    description: str
    system_prompt: str


# Define our agents with system prompts
class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        model: str = "gpt-3.5-turbo",
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.model = model

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
        )

    async def process(self, messages: List[Message]) -> str:
        # Format messages for OpenAI API
        # Start with the system prompt
        formatted_messages = [{"role": "system", "content": self.system_prompt}]

        # Add the conversation history
        for msg in messages:
            if msg.role != "system":  # Skip any system messages in the history
                formatted_messages.append({"role": msg.role, "content": msg.content})

        # Call OpenAI API
        response = client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=500,
            temperature=0.7,
        )

        return response.choices[0].message.content


# Define our specialized agents
writing_agent = Agent(
    name="WritingAssistant",
    description="Helps with writing tasks, including drafting, editing, and creative writing.",
    system_prompt="You are a skilled writing assistant. You excel at helping users with drafting, editing, and creative writing tasks. Your responses should be well-structured, clear, and tailored to the writing needs of the user.",
)

math_agent = Agent(
    name="MathExpert",
    description="Helps with mathematical problems and explanations.",
    system_prompt="You are a math expert. You excel at solving mathematical problems and providing clear explanations of mathematical concepts. Show your work step by step and explain your reasoning clearly.",
)

# Router agent to determine which specialized agent to use
router_agent = Agent(
    name="Router",
    description="Analyzes queries and routes them to the appropriate specialized agent.",
    system_prompt="""You are a router agent that analyzes user queries and decides which specialized agent should handle the response.

For each user message, you must determine the most appropriate agent from the following options:
1. WritingAssistant - For writing tasks, drafting, editing, creative writing, etc.
2. MathExpert - For mathematical problems, equations, calculations, etc.

Respond with ONLY the name of the agent to use (WritingAssistant or MathExpert).
If the query could be handled by either agent, choose the one that seems most appropriate.
""",
)

# Dictionary to store our agents
agents = {
    "WritingAssistant": writing_agent,
    "MathExpert": math_agent,
    "Router": router_agent,
}

# In-memory storage for chat sessions
chat_sessions: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> ChatSession:
    """
    Get an existing session or create a new one if it doesn't exist
    """
    if session_id and session_id in chat_sessions:
        session = chat_sessions[session_id]
        session.updated_at = datetime.now()
        return session

    # Create a new session
    new_session_id = session_id or str(uuid.uuid4())
    new_session = ChatSession(session_id=new_session_id)
    chat_sessions[new_session_id] = new_session
    return new_session


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """
    Returns the chat interface HTML page
    """
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/agents", response_model=Dict[str, AgentMetadata])
async def get_agents():
    """
    Returns metadata about available agents
    """
    return {name: agent.get_metadata() for name, agent in agents.items()}


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Process chat messages using a multi-agent system and return AI response
    """
    try:
        # Get or create a session
        session = get_or_create_session(chat_request.session_id)

        # Add user message to session
        user_message = Message(role="user", content=chat_request.message)
        session.messages.append(user_message)

        # First, use the router agent to determine which specialized agent to use
        router_messages = [Message(role="user", content=chat_request.message)]
        router_response = await router_agent.process(router_messages)

        # Clean up the response to ensurxe it's just the agent name
        selected_agent_name = router_response.strip().split("\n")[0]

        # Default to WritingAssistant if the router's response doesn't match an agent
        if selected_agent_name not in agents or selected_agent_name == "Router":
            selected_agent_name = "WritingAssistant"

        # Update the current agent in the session
        session.current_agent = selected_agent_name

        # Get the selected agent
        selected_agent = agents[selected_agent_name]

        # Process the messages with the selected agent
        agent_response = await selected_agent.process(session.messages)

        # Add assistant message to session
        assistant_message = Message(role="assistant", content=agent_response)
        session.messages.append(assistant_message)

        # Update session
        session.updated_at = datetime.now()
        chat_sessions[session.session_id] = session

        # Return the response along with the agent that generated it
        return ChatResponse(
            session_id=session.session_id,
            response=agent_response,
            agent=selected_agent_name,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@app.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """
    Get a specific chat session
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return chat_sessions[session_id]


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
