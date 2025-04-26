"""
FastAPI OpenAI Chatbot Application with Session Management

This is a minimal FastAPI application that implements a chatbot using the OpenAI SDK.
Features:
- Simple chat endpoint that accepts text messages and returns AI responses
- Session management to maintain conversation history
- Proper handling of system, user, and assistant roles
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
app = FastAPI(title="OpenAI Chatbot")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default system message
DEFAULT_SYSTEM_MESSAGE = "You are a helpful, friendly, and knowledgeable assistant. You provide clear, concise, and accurate information to the user's questions."


# Create Pydantic models for request/response validation
class Message(BaseModel):
    # OpenAI uses "system", "user", and "assistant" roles
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSession(BaseModel):
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    system_message: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


# In-memory storage for chat sessions
# In a production app, you would use a database
chat_sessions: Dict[str, ChatSession] = {}


def get_or_create_session(
    session_id: Optional[str] = None, system_message: Optional[str] = None
) -> ChatSession:
    """
    Get an existing session or create a new one if it doesn't exist
    """
    if session_id and session_id in chat_sessions:
        session = chat_sessions[session_id]
        session.updated_at = datetime.now()

        # Update system message if provided
        if system_message:
            session.system_message = system_message

        return session

    # Create a new session
    new_session_id = session_id or str(uuid.uuid4())
    new_session = ChatSession(
        session_id=new_session_id,
        system_message=system_message or DEFAULT_SYSTEM_MESSAGE,
    )
    chat_sessions[new_session_id] = new_session
    return new_session


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """
    Returns the chat interface HTML page
    """
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Process a chat message and return AI response
    """
    try:
        # Get or create a session
        session = get_or_create_session(
            chat_request.session_id, chat_request.system_message
        )

        # Add user message to session
        user_message = Message(role="user", content=chat_request.message)
        session.messages.append(user_message)

        # Format messages for OpenAI API
        # Start with the system message
        formatted_messages = [{"role": "system", "content": session.system_message}]

        # Add the conversation history
        for msg in session.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=formatted_messages,
            max_tokens=500,
            temperature=0.7,
        )

        # Extract the response
        assistant_response = response.choices[0].message.content

        # Add assistant message to session
        assistant_message = Message(role="assistant", content=assistant_response)
        session.messages.append(assistant_message)

        # Update session
        session.updated_at = datetime.now()
        chat_sessions[session.session_id] = session

        # Return the response
        return ChatResponse(session_id=session.session_id, response=assistant_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


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
