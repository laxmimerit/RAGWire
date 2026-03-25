# Video 11 — Competitive Intelligence Dashboard with CrewAI + Streamlit

**Framework**: CrewAI + Streamlit
**Application**: Multi-agent competitive analysis that runs in the background and displays results in a live dashboard
**Difficulty**: Intermediate–Advanced

---

## What You'll Build

A Streamlit dashboard that:
- Lets users select companies to compare
- Triggers a CrewAI crew in a background thread
- Streams agent status updates to the UI in real time
- Displays a structured competitive analysis with comparison table
- Exports the report as downloadable markdown

---

## Install

```bash
pip install ragwire crewai streamlit langchain-ollama
```

---

## Code: `competitive_dashboard.py`

```python
import streamlit as st
import threading
import queue
import time
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional
import json

st.set_page_config(page_title="Competitive Intelligence", layout="wide")
st.title("Competitive Intelligence Dashboard")

@st.cache_resource
def load_pipeline():
    return RAGWire(Config("config.yaml"))

pipeline = load_pipeline()
llm = ChatOllama(model="qwen2.5:7b")

# --- Tools ---

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    company: Optional[str] = Field(default=None)

class CompanySearchTool(BaseTool):
    name: str = "company_search"
    description: str = "Search documents for information about a specific company."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, company: str = None) -> str:
        filters = {"company_name": [company.lower()]} if company else {}
        docs = pipeline.retrieve(query, filters=filters or None, top_k=4)
        return "\n\n".join(d.page_content[:500] for d in docs) if docs else "No data found."

search_tool = CompanySearchTool()

# --- CrewAI Crew Factory ---

def build_competitive_crew(companies: list[str], status_queue: queue.Queue) -> Crew:

    def status(msg):
        status_queue.put(msg)

    analyst_a = Agent(
        role="Company Analyst A",
        goal=f"Deeply analyze {companies[0] if companies else 'Company A'}",
        backstory="Expert in financial and strategic analysis of the first company.",
        tools=[search_tool],
        llm=llm,
        verbose=False,
        step_callback=lambda step: status(f"[{companies[0]}] {str(step)[:100]}")
    )

    analyst_b = Agent(
        role="Company Analyst B",
        goal=f"Deeply analyze {companies[1] if len(companies) > 1 else 'Company B'}",
        backstory="Expert in financial and strategic analysis of the second company.",
        tools=[search_tool],
        llm=llm,
        verbose=False,
        step_callback=lambda step: status(f"[{companies[1] if len(companies) > 1 else 'B'}] {str(step)[:100]}")
    )

    comparator = Agent(
        role="Competitive Strategist",
        goal="Create a comprehensive competitive comparison",
        backstory="You identify competitive advantages, moats, and strategic positioning across companies.",
        llm=llm,
        verbose=False
    )

    company_a = companies[0] if companies else "Company A"
    company_b = companies[1] if len(companies) > 1 else "Company B"

    task_a = Task(
        description=f"Analyze {company_a}: revenue, growth, margins, competitive strengths, key risks.",
        expected_output=f"Structured profile of {company_a} with key metrics and strategic assessment",
        agent=analyst_a
    )

    task_b = Task(
        description=f"Analyze {company_b}: revenue, growth, margins, competitive strengths, key risks.",
        expected_output=f"Structured profile of {company_b} with key metrics and strategic assessment",
        agent=analyst_b
    )

    compare_task = Task(
        description=f"""
        Compare {company_a} vs {company_b} across:
        - Revenue scale and growth trajectory
        - Profitability (margins comparison)
        - Business model and competitive moat
        - Risk profile
        - Strategic positioning and outlook

        Format as a markdown comparison table, then narrative analysis.
        Include a "Competitive Edge" verdict for each company.
        """,
        expected_output="Markdown comparison table + competitive analysis narrative",
        agent=comparator,
        context=[task_a, task_b]
    )

    return Crew(
        agents=[analyst_a, analyst_b, comparator],
        tasks=[task_a, task_b, compare_task],
        process=Process.sequential,
        verbose=False
    )


def run_crew_async(companies: list[str], result_container: dict, status_queue: queue.Queue):
    """Run crew in background thread."""
    try:
        status_queue.put("Starting competitive analysis...")
        crew = build_competitive_crew(companies, status_queue)
        status_queue.put("Crew initialized. Agents working...")
        result = crew.kickoff()
        result_container["output"] = result.raw
        result_container["done"] = True
        status_queue.put("DONE")
    except Exception as e:
        result_container["error"] = str(e)
        result_container["done"] = True
        status_queue.put(f"ERROR: {e}")

# --- Streamlit UI ---

# Company selection
col1, col2 = st.columns(2)
with col1:
    company_a = st.text_input("Company A", value="Apple Inc.")
with col2:
    company_b = st.text_input("Company B", value="Microsoft")

run_button = st.button("Run Competitive Analysis", type="primary")

if run_button and company_a and company_b:
    st.session_state["result_container"] = {"done": False, "output": None, "error": None}
    st.session_state["status_queue"] = queue.Queue()
    st.session_state["running"] = True

    thread = threading.Thread(
        target=run_crew_async,
        args=(
            [company_a, company_b],
            st.session_state["result_container"],
            st.session_state["status_queue"]
        ),
        daemon=True
    )
    thread.start()
    st.session_state["thread"] = thread

if st.session_state.get("running"):
    result_container = st.session_state["result_container"]
    status_queue = st.session_state["status_queue"]

    status_placeholder = st.empty()
    result_placeholder = st.empty()

    log_lines = []
    while not result_container.get("done"):
        try:
            msg = status_queue.get(timeout=0.5)
            log_lines.append(msg)
            status_placeholder.text_area("Agent Activity Log", "\n".join(log_lines[-10:]), height=200)
        except queue.Empty:
            time.sleep(0.1)

    if result_container.get("error"):
        st.error(f"Analysis failed: {result_container['error']}")
    elif result_container.get("output"):
        st.session_state["running"] = False
        report = result_container["output"]

        st.success("Analysis complete!")
        st.markdown(report)

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"{company_a.lower().replace(' ', '_')}_vs_{company_b.lower().replace(' ', '_')}.md",
            mime="text/markdown"
        )
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| Background thread execution | `threading.Thread` + `queue.Queue` |
| `step_callback` for live updates | Real-time agent status in UI |
| Parallel company analysis | Two agents run concurrently |
| Streamlit session state | `st.session_state` for async results |
| Downloadable report | `st.download_button` |

---

## What to Explain in Video

1. Problem: CrewAI blocks — how to run async in Streamlit (5 min)
2. `threading.Thread` + `queue.Queue` pattern (5 min)
3. `step_callback` for real-time agent monitoring (5 min)
4. Streamlit `st.session_state` for cross-rerun data (5 min)
5. Designing parallel agent tasks vs sequential (5 min)
6. Live demo with Apple vs Microsoft comparison (10 min)
