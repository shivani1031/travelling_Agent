import os
import operator
import uuid
from typing import TypedDict, Annotated

from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

load_dotenv()

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL. Set it to your PostgreSQL connection string.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing Groq API key. Set GROQ_API_KEY in your environment."
    )

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=GROQ_API_KEY,
)


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight results fetched")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: TravelState):
    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner"),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def final_agent(state: TravelState):
    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([HumanMessage(content=final_prompt)])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_agent", final_agent)

    graph.add_edge(START, "flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)

    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        kwargs={"autocommit": True},
        min_size=1,
        max_size=5,
        max_idle=240,
        check=ConnectionPool.check_connection,
        open=True,
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return graph.compile(checkpointer=checkpointer)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.urandom(32)
travel_agent = build_graph()


@app.get("/")
def home():
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True)
    user_input = payload.get("message", "").strip() if isinstance(payload, dict) else ""
    if not user_input:
        return jsonify({"error": "Enter a travel request to continue."}), 400
    if len(user_input) > 2000:
        return jsonify({"error": "Please keep your request under 2,000 characters."}), 400

    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())

    result = travel_agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config={"configurable": {"thread_id": session["thread_id"]}},
    )
    return jsonify({"reply": result["messages"][-1].content})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))