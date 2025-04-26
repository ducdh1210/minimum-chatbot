import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
app = FastAPI(title="OpenAI chatbot")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in chat_request.messages
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=formatted_messages,
            max_tokens=500,
            temperature=0.1,
        )
        return ChatResponse(response=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calling OpenAI API: {str(e)}"
        )


@app.post("/chat_stream")
async def chat_stream(chat_request: ChatRequest):
    try:
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in chat_request.messages
        ]

        # OpenAI API call with stream=True
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=formatted_messages,
            max_tokens=500,
            temperature=0.1,
            stream=True,  # <- Enable streaming here
        )

        async def event_generator():
            try:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        output = {"type": "graph_stream", "output": text}

                        yield f"data: {json.dumps(output)}\n\n"
                    await asyncio.sleep(0)  # Allow FastAPI event loop to breathe
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(event_generator(), media_type="text/event_stream")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calling OpenAI API: {str(e)}"
        )
