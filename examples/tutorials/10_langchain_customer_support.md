# Video 10 — Customer Support Knowledge Base with LangChain

**Framework**: LangChain
**Application**: Support bot backed by RAGWire that handles queries, escalates edge cases, and logs unresolved tickets
**Difficulty**: Intermediate

---

## What You'll Build

A customer support agent that:
- Answers product questions from a knowledge base (docs, FAQs, manuals)
- Classifies query intent: FAQ / Troubleshooting / Billing / Escalate
- Provides step-by-step troubleshooting answers
- Detects when it can't help and creates a support ticket
- Tracks conversation sessions per customer

---

## Install

```bash
pip install ragwire langchain langchain-ollama langgraph
```

---

## Code: `support_agent.py`

```python
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from ragwire import RAGWire, Config
from datetime import datetime
import json
import uuid

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

# Simple in-memory ticket store (replace with DB in production)
tickets = {}

# --- Tools ---

@tool
def search_knowledge_base(query: str, category: str = None) -> str:
    """
    Search the support knowledge base for answers.
    category: optional filter — 'faq', 'troubleshooting', 'billing', 'product'
    Returns relevant support documentation passages.
    """
    filters = {}
    if category:
        filters["doc_type"] = [category]

    docs = pipeline.retrieve(query, filters=filters if filters else None, top_k=4)
    if not docs:
        return "No relevant documentation found for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        results.append(f"[KB-{i}] {meta.get('file_name', 'doc')}\n{doc.page_content[:500]}")
    return "\n\n".join(results)

@tool
def classify_intent(customer_message: str) -> str:
    """
    Classify the customer's intent to route to appropriate response strategy.
    Returns: FAQ | TROUBLESHOOTING | BILLING | ESCALATE
    """
    prompt = f"""Classify this customer support message into ONE category:
- FAQ: general product questions
- TROUBLESHOOTING: technical issues, errors, not working
- BILLING: pricing, charges, refunds, subscriptions
- ESCALATE: complaints, account issues, security, or if very angry/urgent

Message: {customer_message}
Category (respond with just the word):"""

    result = llm.invoke([HumanMessage(content=prompt)])
    category = result.content.strip().upper()
    if category not in ("FAQ", "TROUBLESHOOTING", "BILLING", "ESCALATE"):
        category = "FAQ"
    return category

@tool
def create_support_ticket(
    customer_id: str,
    issue_summary: str,
    priority: str = "normal"
) -> str:
    """
    Create a support ticket when the agent cannot resolve the issue.
    priority: 'low', 'normal', 'high', 'urgent'
    Returns a ticket ID for the customer.
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    tickets[ticket_id] = {
        "id": ticket_id,
        "customer_id": customer_id,
        "issue": issue_summary,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(),
        "agent": "AI Support Bot"
    }
    return (f"Support ticket created: {ticket_id}\n"
            f"Priority: {priority}\n"
            f"A human agent will contact you within "
            f"{'1 hour' if priority == 'urgent' else '24 hours'}.")

@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up the status of an existing support ticket."""
    ticket = tickets.get(ticket_id.upper())
    if not ticket:
        return f"No ticket found with ID: {ticket_id}"
    return json.dumps(ticket, indent=2)

@tool
def check_known_issues() -> str:
    """
    Check for any known ongoing issues or outages.
    Call this for troubleshooting queries before searching the knowledge base.
    """
    # In production, this would query a status page API
    return (
        "Current known issues:\n"
        "- No active outages reported\n"
        "- Maintenance window: Sundays 2-4 AM UTC\n"
        "Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M")
    )

# --- Support Agent ---

SUPPORT_SYSTEM_PROMPT = """You are a helpful, empathetic customer support agent.

Guidelines:
1. Always greet the customer by name if known
2. Classify intent first, then search the knowledge base
3. For TROUBLESHOOTING: check known issues first, then KB, give step-by-step instructions
4. For BILLING: always be accurate and don't make promises about refunds
5. For ESCALATE: acknowledge their frustration, create a ticket with 'high' priority
6. If you can't find a good answer after 2 KB searches, create a support ticket
7. Be concise but complete. Bullet points for steps. Plain language.
8. End every response with: "Is there anything else I can help you with?"
"""

memory = InMemorySaver()
support_agent = create_agent(
    llm,
    tools=[classify_intent, check_known_issues, search_knowledge_base,
           create_support_ticket, get_ticket_status],
    checkpointer=memory,
    system_prompt=SUPPORT_SYSTEM_PROMPT
)

def support_session(customer_id: str):
    """Run an interactive support session for a customer."""
    config = {"configurable": {"thread_id": f"support-{customer_id}"}}
    print(f"Support session started for customer: {customer_id}")
    print("Type 'quit' to end session\n")

    while True:
        user_input = input("Customer: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Support session ended.")
            break

        response = support_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        print(f"\nSupport Agent: {response['messages'][-1].content}\n")

def demo():
    """Demo with preset customer messages."""
    scenarios = [
        ("CUST-001", "My login isn't working, I keep getting error 401"),
        ("CUST-002", "How do I export my data to CSV?"),
        ("CUST-002", "I was charged twice this month and nobody is helping me! This is unacceptable!"),
    ]
    for customer_id, message in scenarios:
        print(f"\n{'='*60}")
        print(f"Customer {customer_id}: {message}")
        print('='*60)
        cfg = {"configurable": {"thread_id": f"support-{customer_id}"}}
        response = support_agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=cfg
        )
        print(f"Agent: {response['messages'][-1].content}")

if __name__ == "__main__":
    demo()
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| Intent classification tool | `classify_intent` before KB search |
| Escalation logic | `create_support_ticket` with priority |
| Per-customer sessions | `thread_id=f"support-{customer_id}"` |
| Knowledge base filtering | `filters={"doc_type": [category]}` |
| Stateful multi-turn chat | `InMemorySaver` per customer |

---

## What to Explain in Video

1. Support bot architecture — classify then retrieve (5 min)
2. Knowledge base structure — how to organize support docs for RAG (5 min)
3. Intent classification as routing logic (5 min)
4. Escalation detection — when to create a ticket (5 min)
5. Per-customer session isolation with `thread_id` (3 min)
6. Production considerations: DB for tickets, logging, analytics (5 min)
7. Live demo with angry customer scenario (7 min)
