# Video 14 — Hierarchical Supervisor Multi-Agent System with LangGraph

**Framework**: LangGraph
**Application**: A supervisor agent that dynamically routes complex queries to specialized sub-agents (Financial, Legal, Technical, Summary)
**Difficulty**: Advanced

---

## What You'll Build

A hierarchical multi-agent system where:
- **Supervisor** receives any query and decides which specialist to call
- **Financial Agent** — handles revenue, margins, guidance, valuations
- **Legal/Risk Agent** — handles risk factors, legal proceedings, compliance
- **Technical Agent** — handles product descriptions, R&D, technology strategy
- **Summary Agent** — handles general overview and executive summary requests
- Supervisor synthesizes outputs from multiple agents into one answer

---

## Install

```bash
pip install ragwire langgraph langchain-ollama
```

---

## Code: `supervisor_system.py`

```python
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from ragwire import RAGWire, Config
import operator

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

AgentName = Literal["financial", "legal_risk", "technical", "summary", "FINISH"]

# --- State ---

class SupervisorState(TypedDict):
    query: str
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: AgentName
    agent_outputs: dict   # {agent_name: output_text}
    final_answer: str
    iteration: int

# --- Specialist Agent Factory ---

def make_specialist_agent(
    name: str,
    domain: str,
    search_focus: str,
    additional_instructions: str = ""
):
    """Factory to create a domain-specific RAG agent node."""

    def agent_node(state: SupervisorState) -> SupervisorState:
        query = state["query"]

        # Domain-specific search
        search_query = f"{search_focus} {query}"
        docs = pipeline.retrieve(search_query, top_k=5)

        if not docs:
            output = f"No relevant {domain} information found."
        else:
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            prompt = f"""You are a {domain} specialist.
{additional_instructions}
Answer this query from a {domain} perspective using only the provided context.

Query: {query}

Context:
{context}

Provide a precise, well-structured {domain} analysis:"""

            result = llm.invoke([HumanMessage(content=prompt)])
            output = result.content

        return {
            **state,
            "agent_outputs": {**state["agent_outputs"], name: output},
            "messages": [AIMessage(content=f"[{name.upper()} AGENT]: {output[:200]}...")]
        }

    return agent_node

# --- Specialist Nodes ---

financial_agent = make_specialist_agent(
    name="financial",
    domain="Financial Analysis",
    search_focus="revenue income profit margin financial statements",
    additional_instructions=(
        "Focus on: revenue figures, growth rates, profit margins, "
        "cash flow, debt levels, and forward guidance. "
        "Always include specific numbers and percentages."
    )
)

legal_risk_agent = make_specialist_agent(
    name="legal_risk",
    domain="Legal & Risk Assessment",
    search_focus="risk factors legal proceedings regulatory compliance",
    additional_instructions=(
        "Focus on: material risk factors, legal proceedings, regulatory risks, "
        "compliance issues, and contingent liabilities. "
        "Categorize risks by type and severity."
    )
)

technical_agent = make_specialist_agent(
    name="technical",
    domain="Technology & Product Analysis",
    search_focus="product technology research development innovation",
    additional_instructions=(
        "Focus on: product portfolio, R&D spending, technology strategy, "
        "innovation initiatives, and competitive technical advantages."
    )
)

summary_agent = make_specialist_agent(
    name="summary",
    domain="Executive Summary",
    search_focus="overview business strategy key highlights",
    additional_instructions=(
        "Provide a high-level executive summary covering: "
        "company overview, key business highlights, and strategic direction. "
        "Keep it concise and accessible to non-technical readers."
    )
)

# --- Supervisor Node ---

SUPERVISOR_SYSTEM = """You are a supervisor managing specialized document analysis agents.

Available agents:
- financial: revenue, profits, margins, cash flow, guidance
- legal_risk: risks, legal issues, compliance, regulatory matters
- technical: products, technology, R&D, innovation
- summary: general overview, executive summary, strategic highlights

Current query: {query}
Already called agents: {called_agents}
Agent outputs so far: {outputs_summary}

Decide: which agent to call next, or FINISH if you have enough information.

Rules:
- Call relevant specialist(s) for the query
- Don't call the same agent twice
- Call FINISH when you have sufficient information to answer comprehensively
- For broad queries, you may call 2-3 agents

Respond with exactly one word: financial | legal_risk | technical | summary | FINISH"""

def supervisor(state: SupervisorState) -> SupervisorState:
    """Route to next specialist or finish."""
    called = list(state["agent_outputs"].keys())
    outputs_summary = "\n".join(
        f"- {k}: {v[:100]}..." for k, v in state["agent_outputs"].items()
    ) or "None yet"

    prompt = SUPERVISOR_SYSTEM.format(
        query=state["query"],
        called_agents=called or "none",
        outputs_summary=outputs_summary
    )

    result = llm.invoke([HumanMessage(content=prompt)])
    decision = result.content.strip().lower()

    valid = {"financial", "legal_risk", "technical", "summary", "finish"}
    if decision not in valid:
        decision = "finish"

    next_agent = "FINISH" if decision == "finish" else decision

    return {
        **state,
        "next_agent": next_agent,
        "iteration": state["iteration"] + 1
    }

# --- Synthesis Node ---

def synthesize(state: SupervisorState) -> SupervisorState:
    """Combine all agent outputs into a final coherent answer."""
    if not state["agent_outputs"]:
        return {**state, "final_answer": "No relevant information found."}

    agents_content = "\n\n".join(
        f"### {name.replace('_', ' ').title()} Analysis\n{content}"
        for name, content in state["agent_outputs"].items()
    )

    result = llm.invoke([HumanMessage(content=f"""Synthesize these specialist analyses into one comprehensive answer.

Original Query: {state['query']}

Specialist Inputs:
{agents_content}

Write a unified, well-structured answer that integrates all relevant findings.
Use clear headings if multiple aspects are covered.""")])

    return {**state, "final_answer": result.content}

# --- Routing ---

def route_supervisor(state: SupervisorState) -> str:
    """Route based on supervisor's decision."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH" or state["iteration"] >= 4:
        return "synthesize"
    return next_agent

# --- Build Graph ---

def build_supervisor_graph():
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("financial", financial_agent)
    graph.add_node("legal_risk", legal_risk_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("summary", summary_agent)
    graph.add_node("synthesize", synthesize)

    # Entry → Supervisor
    graph.set_entry_point("supervisor")

    # Supervisor routes to specialists or synthesis
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "financial": "financial",
            "legal_risk": "legal_risk",
            "technical": "technical",
            "summary": "summary",
            "synthesize": "synthesize"
        }
    )

    # All specialists return to supervisor
    for agent in ["financial", "legal_risk", "technical", "summary"]:
        graph.add_edge(agent, "supervisor")

    graph.add_edge("synthesize", END)

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


supervisor_graph = build_supervisor_graph()


def ask(query: str, thread_id: str = "supervisor-demo") -> str:
    """Submit a query to the supervisor system."""
    initial = SupervisorState(
        query=query,
        messages=[],
        next_agent="financial",  # will be overridden by supervisor
        agent_outputs={},
        final_answer="",
        iteration=0
    )

    config = {"configurable": {"thread_id": thread_id}}
    result = supervisor_graph.invoke(initial, config=config)

    print(f"\nAgents called: {list(result['agent_outputs'].keys())}")
    print(f"Iterations: {result['iteration']}")
    print(f"\n{'='*60}\nFINAL ANSWER:\n{'='*60}")
    print(result["final_answer"])
    return result["final_answer"]


if __name__ == "__main__":
    # Test different query types
    ask("What was Apple's revenue growth rate and what are the main risk factors?")
    print("\n" + "="*80 + "\n")
    ask("Give me an executive overview of the company's strategy")
    print("\n" + "="*80 + "\n")
    ask("What R&D investments is Apple making in AI?")
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| Supervisor routing pattern | `supervisor` node + `route_supervisor` |
| Agent factory function | `make_specialist_agent()` |
| Dynamic routing to N agents | `add_conditional_edges` with 5 targets |
| Accumulating outputs | `agent_outputs` dict in state |
| Synthesis from multiple agents | `synthesize` node |
| Loop guard | `iteration >= 4` max loops |

---

## What to Explain in Video

1. Hierarchical agent patterns — supervisor vs peer agents (5 min)
2. Agent factory function — DRY principle for agents (5 min)
3. Supervisor prompt design — what makes routing reliable (7 min)
4. `add_conditional_edges` with many targets (5 min)
5. The `synthesize` node — combining multi-agent outputs (5 min)
6. Loop guards — preventing infinite routing (3 min)
7. Live demo showing routing trace with different query types (10 min)
