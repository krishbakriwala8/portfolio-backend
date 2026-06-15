from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
from typing import Optional, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ═══════════════════════════════════════════════════════════════
# PORTFOLIO CHATBOT
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are Krish's AI portfolio assistant. Your job is to answer two types of questions:

1. QUESTIONS ABOUT KRISH — answer ONLY using the facts below. Never invent or guess.
2. GENERAL AI/ML/TECH QUESTIONS — answer helpfully and accurately (e.g. "what is computer vision?", "why use Hugging Face?", "what is Grad-CAM?"). Relate back to Krish's work when relevant.

If someone asks about Krish but the answer is not in the facts below, say: "I don't have that information, but you can reach Krish at krishbakriwala8@gmail.com"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KRISH'S PORTFOLIO FACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NAME: Krish Akshay Bakriwala
DEGREE: M.Sc. Artificial Intelligence, Brandenburg University of Technology Cottbus-Senftenberg, Germany
BACKGROUND: Computer Engineering undergraduate + AI postgraduate
OPEN TO: AI internships and full-time roles in Machine Learning, Backend Engineering, Software Development — across Europe, especially Germany
EMAIL: krishbakriwala8@gmail.com
LINKEDIN: https://www.linkedin.com/in/krish-akshay-bakriwala-3885a61b8
GITHUB: https://github.com/krishbakriwala8
AVAILABLE: August 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL SKILLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agentic AI: Google ADK, LangChain, LangGraph, RAG Pipelines, LLM Tool Calling, Groq API, LiteLLM, ChromaDB, FAISS
Machine Learning: Deep Learning, PyTorch, TensorFlow, NLP/Transformers, Hugging Face, Scikit-learn, Prompt Engineering
Computer Vision: CLIP (ViT-B/32), OpenCV, Grad-CAM, Zero-Shot Learning, Albumentations
Backend: Python, FastAPI, REST APIs, Docker, Streamlit, Git, Linux
Data Engineering: BeautifulSoup, Scrapy, PostgreSQL, ETL Pipelines, Batch Scheduling
Automation: Power Automate, N8N, SharePoint
AI Agent Development: Claude API patterns, MCP tool integration, session memory, context harness strategies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Multi-Agent Travel Intelligence System — Google ADK, LLaMA 3.3 70B, Groq
2. Industrial Defect Detection — CLIP, Grad-CAM, PyTorch, MVTec AD
3. Multi-Tool AI Research Agent — Groq, LLaMA 3.3 70B, Tool Calling, Session Memory, ReAct pattern
4. Battery Test Failure Assistant — LangChain, RAG, ChromaDB, Groq
5. Web Scraping Data Pipeline — BeautifulSoup, Scrapy, PostgreSQL
6. Document Workflow Automation — Power Automate, N8N, SharePoint
7. Real-time Sentiment Analysis — FastAPI, BERT, Docker
8. Smart Document Q&A — LangChain, FAISS
9. Schema-Based ETL Pipeline
10. Scenario Generation & Anomaly Detection
11. Fine-Tuned Niche Content Generator
12. Email Summarizer — GPT-4, Gmail API
13. AQI Predictor, Movie Recommender, Car Price Predictor
14. Fitness App (Android), Online Car Rental System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Be concise, friendly, and professional
- Keep responses under 200 words unless detail is needed
"""


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
def chat(req: ChatRequest):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in req.history[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": req.message})

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.4
    }
    response = requests.post(GROQ_URL, headers=headers, json=data)
    result = response.json()
    reply = result["choices"][0]["message"]["content"]
    return {"reply": reply}


# ═══════════════════════════════════════════════════════════════
# MULTI-TOOL AI RESEARCH AGENT
# ═══════════════════════════════════════════════════════════════

# ── Tool definitions (MCP-style schema) ───────────────────────
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic. Use this for facts, news, definitions, or anything requiring up-to-date knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for any numerical calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate e.g. '2 ** 10' or '(100 * 3.14) / 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a long piece of text into concise key points. Use when you have too much text and need to condense it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize"
                    },
                    "num_points": {
                        "type": "integer",
                        "description": "Number of key bullet points to extract (default: 5)"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_text",
            "description": "Analyze text for sentiment, key themes, entities, or specific patterns. Returns structured analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: 'sentiment', 'themes', 'entities', or 'general'",
                        "enum": ["sentiment", "themes", "entities", "general"]
                    }
                },
                "required": ["text", "analysis_type"]
            }
        }
    }
]

# ── Agent system prompt ───────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are a production-grade AI research agent built by Krish Bakriwala.

You have access to tools and must use them to answer the user's task thoroughly.

## Your approach (ReAct pattern):
1. REASON — Think about what information you need
2. ACT — Call the appropriate tool(s)
3. OBSERVE — Review tool results
4. SYNTHESIZE — Combine findings into a clear, structured answer

## Rules:
- Always use tools when you need current information or calculations
- Use multiple tools if needed — be thorough
- Format your final answer clearly with headers (##), bullet points, and structure
- If a search returns no results, try a different search query
- Be specific and accurate — never fabricate facts
- End your response with a brief "## Summary" section

## Available tools:
- web_search: Find current information on any topic
- calculate: Evaluate mathematical expressions
- summarize_text: Condense long text into key points
- analyze_text: Extract sentiment, themes, or entities from text"""

# ── In-memory session store ───────────────────────────────────
# Stores last 10 messages per session for context continuity
agent_sessions: dict = {}


# ── Tool implementations ──────────────────────────────────────
def tool_web_search(query: str) -> str:
    """Search DuckDuckGo for information."""
    try:
        url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
        r = requests.get(url, timeout=8)
        data = r.json()

        results = []
        if data.get("AbstractText"):
            results.append(f"Overview: {data['AbstractText']}")
        if data.get("Answer"):
            results.append(f"Direct Answer: {data['Answer']}")
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"• {topic['Text']}")

        if results:
            return "\n".join(results)

        # Fallback — try instant answer endpoint
        url2 = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&ia=web"
        r2 = requests.get(url2, timeout=8)
        data2 = r2.json()
        if data2.get("AbstractText"):
            return data2["AbstractText"]

        return f"Search returned no direct results for '{query}'. Try rephrasing or use a more specific query."
    except Exception as e:
        return f"Search error: {str(e)}"


def tool_calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Whitelist only safe characters
        safe_chars = set("0123456789+-*/().,** %")
        cleaned = expression.replace("^", "**").replace("x", "*")
        if not all(c in safe_chars for c in cleaned.replace(" ", "")):
            return f"Invalid expression — only basic math operations allowed (+, -, *, /, **, parentheses)"
        result = eval(cleaned, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def tool_summarize_text(text: str, num_points: int = 5) -> str:
    """Summarize text using LLM."""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": f"Summarize the following text into exactly {num_points} concise bullet points. Start each point with '•'. Be specific and factual."},
                {"role": "user", "content": text[:3000]}  # cap to avoid token limits
            ],
            "max_tokens": 400,
            "temperature": 0.2
        }
        r = requests.post(GROQ_URL, headers=headers, json=data)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Summarization error: {str(e)}"


def tool_analyze_text(text: str, analysis_type: str = "general") -> str:
    """Analyze text for sentiment, themes, entities, or general insights."""
    try:
        prompts = {
            "sentiment": "Analyze the sentiment of this text. Rate it as Positive/Negative/Neutral with a confidence score (0-100%) and explain why.",
            "themes": "Extract the 3-5 main themes or topics from this text. For each theme, give a 1-sentence explanation.",
            "entities": "Extract all named entities from this text: people, organizations, locations, dates, and key terms. List them by category.",
            "general": "Provide a structured analysis of this text covering: main topic, key arguments, tone, and notable insights."
        }
        prompt = prompts.get(analysis_type, prompts["general"])
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text[:3000]}
            ],
            "max_tokens": 400,
            "temperature": 0.2
        }
        r = requests.post(GROQ_URL, headers=headers, json=data)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Analysis error: {str(e)}"


# ── Tool dispatcher ───────────────────────────────────────────
TOOL_MAP = {
    "web_search":    lambda args: tool_web_search(args["query"]),
    "calculate":     lambda args: tool_calculate(args["expression"]),
    "summarize_text":lambda args: tool_summarize_text(args["text"], args.get("num_points", 5)),
    "analyze_text":  lambda args: tool_analyze_text(args["text"], args.get("analysis_type", "general")),
}


# ── Request / Response models ─────────────────────────────────
class AgentRequest(BaseModel):
    task: str
    session_id: Optional[str] = "default"


class AgentStep(BaseModel):
    type: str          # "tool_call" | "tool_result" | "thinking"
    tool: Optional[str] = None
    args: Optional[dict] = None
    result: Optional[str] = None
    content: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    steps: List[dict]
    session_id: str
    tools_used: List[str]


# ── Main agent endpoint ───────────────────────────────────────
@app.post("/agent/run")
def run_agent(req: AgentRequest):
    session_id = req.session_id or "default"

    # Load or create session memory
    if session_id not in agent_sessions:
        agent_sessions[session_id] = []

    history = agent_sessions[session_id]
    history.append({"role": "user", "content": req.task})

    # Build message context: system + memory + current task
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + history

    steps = []
    tools_used = []
    max_iterations = 6  # prevent infinite loops

    for iteration in range(max_iterations):
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "tools": AGENT_TOOLS,
            "tool_choice": "auto",
            "max_tokens": 1500,
            "temperature": 0.3
        }

        try:
            r = requests.post(GROQ_URL, headers=headers, json=data, timeout=30)
            result = r.json()

            if "error" in result:
                return {"response": f"LLM error: {result['error']['message']}", "steps": steps, "session_id": session_id, "tools_used": tools_used}

            msg = result["choices"][0]["message"]
            messages.append(msg)

        except Exception as e:
            return {"response": f"Request error: {str(e)}", "steps": steps, "session_id": session_id, "tools_used": tools_used}

        # Agent wants to call tool(s)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                # Log the tool call
                steps.append({
                    "type": "tool_call",
                    "tool": fn_name,
                    "args": fn_args
                })
                if fn_name not in tools_used:
                    tools_used.append(fn_name)

                # Execute the tool
                if fn_name in TOOL_MAP:
                    tool_result = TOOL_MAP[fn_name](fn_args)
                else:
                    tool_result = f"Unknown tool: {fn_name}"

                # Log the result
                steps.append({
                    "type": "tool_result",
                    "tool": fn_name,
                    "result": tool_result
                })

                # Feed result back to agent
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })

        else:
            # Agent finished — return final response
            final_response = msg.get("content", "No response generated.")

            # Update session memory (keep last 10 messages)
            history.append({"role": "assistant", "content": final_response})
            agent_sessions[session_id] = history[-10:]

            return {
                "response": final_response,
                "steps": steps,
                "session_id": session_id,
                "tools_used": tools_used
            }

    # If max iterations reached
    return {
        "response": "Agent reached maximum reasoning steps. Please try a more specific question.",
        "steps": steps,
        "session_id": session_id,
        "tools_used": tools_used
    }


@app.delete("/agent/session/{session_id}")
def clear_session(session_id: str):
    """Clear agent memory for a session."""
    if session_id in agent_sessions:
        del agent_sessions[session_id]
    return {"message": f"Session {session_id} cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "endpoints": ["/chat", "/agent/run"]}
