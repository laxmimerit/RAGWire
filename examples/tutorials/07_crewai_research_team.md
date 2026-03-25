# Video 07 — Multi-Agent Research Team with CrewAI

**Framework**: CrewAI
**Application**: 3-agent crew (Researcher, Analyst, Writer) that produces a structured research brief from documents
**Difficulty**: Intermediate

---

## What You'll Build

A CrewAI crew of 3 specialized agents that collaborate sequentially:
1. **Researcher** — retrieves relevant document passages using RAGWire tools
2. **Analyst** — synthesizes findings, identifies patterns and key insights
3. **Writer** — produces a polished, structured research brief in markdown

---

## Install

```bash
pip install ragwire crewai crewai-tools langchain-ollama
```

---

## Code: `research_crew.py`

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional

# --- Initialize RAGWire ---

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

# --- Custom RAGWire Tools for CrewAI ---

class DocumentSearchInput(BaseModel):
    query: str = Field(description="Search query to find relevant document chunks")
    company: Optional[str] = Field(default=None, description="Company name filter")
    year: Optional[int] = Field(default=None, description="Fiscal year filter")

class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = (
        "Search through ingested documents using semantic + keyword hybrid search. "
        "Returns relevant passages with source metadata."
    )
    args_schema: type[BaseModel] = DocumentSearchInput

    def _run(self, query: str, company: str = None, year: int = None) -> str:
        filters = {}
        if company:
            filters["company_name"] = [company.lower()]
        if year:
            filters["fiscal_year"] = [year]

        docs = pipeline.retrieve(query, filters=filters if filters else None)
        if not docs:
            return "No relevant documents found for this query."

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            results.append(
                f"[Source {i}] {meta.get('company_name', 'Unknown')} | "
                f"{meta.get('doc_type', '')} | FY{meta.get('fiscal_year', '')}\n"
                f"{doc.page_content[:600]}"
            )
        return "\n\n---\n\n".join(results)


class MetadataExplorerInput(BaseModel):
    topic: str = Field(description="Topic or query to explore available metadata for")

class MetadataExplorerTool(BaseTool):
    name: str = "metadata_explorer"
    description: str = (
        "Explore what companies, document types, and years are available in the document collection. "
        "Use this first to understand what data is available before searching."
    )
    args_schema: type[BaseModel] = MetadataExplorerInput

    def _run(self, topic: str) -> str:
        return pipeline.get_filter_context(topic)


search_tool = DocumentSearchTool()
metadata_tool = MetadataExplorerTool()

# --- Define Agents ---

researcher = Agent(
    role="Senior Financial Researcher",
    goal="Retrieve comprehensive, accurate information from documents to answer the research question",
    backstory=(
        "You are a meticulous researcher with 15 years of experience in financial document analysis. "
        "You excel at finding specific data points, quotes, and passages from complex documents. "
        "You always verify information from multiple document sections before concluding."
    ),
    tools=[metadata_tool, search_tool],
    llm=llm,
    verbose=True,
    max_iter=5
)

analyst = Agent(
    role="Strategic Analyst",
    goal="Synthesize raw research findings into structured insights and identify key patterns",
    backstory=(
        "You are a strategic analyst who transforms raw data into actionable intelligence. "
        "You identify trends, contradictions, and strategic implications. "
        "You excel at comparing across time periods and entities."
    ),
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Research Writer",
    goal="Produce a clear, professional research brief suitable for executives",
    backstory=(
        "You are a skilled technical writer who translates complex analysis into "
        "accessible, well-structured documents. You write concisely, use clear headings, "
        "and always cite sources."
    ),
    llm=llm,
    verbose=True
)

# --- Define Tasks ---

def build_crew(research_question: str) -> Crew:

    research_task = Task(
        description=f"""
        Research the following question using the available document search tools:

        QUESTION: {research_question}

        Steps:
        1. First use metadata_explorer to understand what documents are available
        2. Search for relevant passages using multiple targeted queries
        3. Search for context, background, and supporting data
        4. Compile all relevant passages with their source citations

        Output a comprehensive collection of relevant excerpts and their sources.
        """,
        expected_output="A structured collection of relevant document excerpts with source citations",
        agent=researcher
    )

    analysis_task = Task(
        description=f"""
        Analyze the research findings to answer: {research_question}

        Your analysis should:
        1. Identify the key facts and data points
        2. Note any trends, changes over time, or comparisons
        3. Highlight contradictions or uncertainties
        4. Derive strategic implications
        5. Rate confidence level (High/Medium/Low) for each finding

        Build on the researcher's findings — do not re-search.
        """,
        expected_output="Structured analysis with key findings, trends, and confidence ratings",
        agent=analyst,
        context=[research_task]
    )

    writing_task = Task(
        description=f"""
        Write a professional research brief answering: {research_question}

        Format:
        # Research Brief: [Topic]

        ## Executive Summary
        [2-3 sentence answer to the question]

        ## Key Findings
        [Bullet points with specific data, cite sources as (Source N)]

        ## Analysis
        [Paragraph analysis of trends and implications]

        ## Limitations
        [What the documents don't cover or where confidence is low]

        ## Sources Referenced
        [List document sources]
        """,
        expected_output="A formatted markdown research brief ready for executive review",
        agent=writer,
        context=[research_task, analysis_task]
    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True
    )


if __name__ == "__main__":
    question = "What are Apple's main revenue drivers and growth strategy for 2025?"
    crew = build_crew(question)
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FINAL RESEARCH BRIEF")
    print("="*60)
    print(result.raw)
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| `BaseTool` with `args_schema` | Typed CrewAI tool wrappers |
| Agent roles and backstories | Specialized agent behavior |
| `context=[...]` in Task | Agent knowledge sharing |
| `Process.sequential` | Ordered agent pipeline |
| RAGWire in CrewAI tools | `pipeline.retrieve()` inside tools |

---

## What to Explain in Video

1. CrewAI architecture — agents, tasks, tools, process (5 min)
2. Wrapping RAGWire in `BaseTool` with Pydantic schema (7 min)
3. Agent role design — why backstories matter for LLM behavior (5 min)
4. Task `context` parameter — how agents share knowledge (5 min)
5. Sequential vs hierarchical process (3 min)
6. Reading CrewAI verbose output — understanding agent reasoning (5 min)
7. Live demo (10 min)
