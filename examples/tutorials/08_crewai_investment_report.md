# Video 08 — Automated Investment Report Generator with CrewAI

**Framework**: CrewAI
**Application**: Automated equity research report pipeline — ingests 10-K filings, produces investor-grade PDF report
**Difficulty**: Intermediate–Advanced

---

## What You'll Build

A CrewAI crew that runs a full investment research pipeline:
1. **Data Collector** — ingests and indexes SEC filings automatically
2. **Financial Analyst** — extracts financial metrics (revenue, margins, debt)
3. **Risk Analyst** — identifies and scores risk factors from filings
4. **Investment Strategist** — synthesizes buy/hold/sell recommendation
5. **Report Writer** — generates a formatted markdown investment memo

---

## Install

```bash
pip install ragwire crewai crewai-tools langchain-ollama
```

---

## Code: `investment_report.py`

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from ragwire import RAGWire, Config
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional
import os

config = Config("config.yaml")
pipeline = RAGWire(config)
llm = ChatOllama(model="qwen2.5:7b")

# --- Tools ---

class SearchInput(BaseModel):
    query: str = Field(description="Financial query to search")
    company: Optional[str] = Field(default=None)

class FinancialSearchTool(BaseTool):
    name: str = "financial_search"
    description: str = "Search financial documents for specific metrics, statements, or disclosures."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, company: str = None) -> str:
        filters = {"company_name": [company.lower()]} if company else {}
        docs = pipeline.retrieve(query, filters=filters or None, top_k=6)
        if not docs:
            return "No data found."
        return "\n\n".join(
            f"[{d.metadata.get('doc_type', '')} FY{d.metadata.get('fiscal_year', '')}]\n{d.page_content[:700]}"
            for d in docs
        )

class RiskSearchTool(BaseTool):
    name: str = "risk_search"
    description: str = "Search specifically for risk factors, uncertainties, and legal disclosures."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, company: str = None) -> str:
        risk_query = f"risk factors {query}"
        filters = {"company_name": [company.lower()]} if company else {}
        docs = pipeline.retrieve(risk_query, filters=filters or None, top_k=8)
        if not docs:
            return "No risk data found."
        return "\n\n".join(f"[Risk] {d.page_content[:500]}" for d in docs)

search_tool = FinancialSearchTool()
risk_tool = RiskSearchTool()

# --- Agents ---

data_collector = Agent(
    role="Data Collection Specialist",
    goal="Verify what financial data is available and compile key company information",
    backstory="You specialize in data availability assessment and document cataloging for investment research.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

financial_analyst = Agent(
    role="Senior Financial Analyst (CFA)",
    goal="Extract and analyze all key financial metrics including revenue, margins, cash flow, and balance sheet",
    backstory=(
        "A CFA charterholder with deep expertise in financial statement analysis. "
        "You extract precise numbers, calculate ratios, and identify financial trends. "
        "You flag data quality issues and state confidence levels."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True
)

risk_analyst = Agent(
    role="Risk Assessment Specialist",
    goal="Identify, categorize, and score all material risk factors from the filings",
    backstory=(
        "You specialize in enterprise risk assessment and regulatory analysis. "
        "You categorize risks as: Market, Operational, Regulatory, Competitive, or Financial. "
        "You score each risk 1-5 for likelihood and impact."
    ),
    tools=[risk_tool],
    llm=llm,
    verbose=True
)

strategist = Agent(
    role="Investment Strategist",
    goal="Synthesize financial and risk analysis into a clear investment recommendation",
    backstory=(
        "A portfolio manager with 20 years of experience. You weigh financial strength "
        "against risk exposure to form clear, evidence-based investment views. "
        "You always provide a Buy/Hold/Sell recommendation with price target rationale."
    ),
    llm=llm,
    verbose=True
)

report_writer = Agent(
    role="Investment Report Writer",
    goal="Produce a professional investment memo in standard sell-side format",
    backstory="You write institutional-grade research reports with precise language, clear structure, and source citations.",
    llm=llm,
    verbose=True
)

# --- Tasks ---

def build_investment_crew(company_name: str, ticker: str) -> Crew:

    collect_task = Task(
        description=f"""
        Verify available data for {company_name} ({ticker}).
        Check: what document types exist (10-K, 10-Q), which fiscal years are covered.
        Retrieve the company overview, business description, and primary products/services.
        """,
        expected_output=f"Company overview and data availability summary for {company_name}",
        agent=data_collector
    )

    financial_task = Task(
        description=f"""
        Analyze the financial performance of {company_name} ({ticker}).
        Extract and calculate:
        - Revenue (last 3 years if available), YoY growth rate
        - Gross margin, Operating margin, Net margin
        - Free cash flow and capital allocation
        - Debt-to-equity ratio, interest coverage
        - Any guidance or forward-looking statements

        Cite all numbers with their source (document type and fiscal year).
        """,
        expected_output="Structured financial metrics table with trends and source citations",
        agent=financial_analyst,
        context=[collect_task]
    )

    risk_task = Task(
        description=f"""
        Assess the material risks for {company_name} ({ticker}).
        For each risk:
        - Category (Market/Operational/Regulatory/Competitive/Financial)
        - Description (1-2 sentences from the filing)
        - Likelihood score (1=rare, 5=very likely)
        - Impact score (1=minor, 5=severe)
        - Mitigation mentioned in filing (if any)

        Focus on risks that could materially affect the investment thesis.
        """,
        expected_output="Risk matrix with scored risk factors and mitigations",
        agent=risk_analyst,
        context=[collect_task]
    )

    strategy_task = Task(
        description=f"""
        Synthesize the financial analysis and risk assessment for {company_name} ({ticker}).
        Provide:
        - Overall investment view: Buy / Hold / Sell with conviction level
        - 3 key reasons supporting the recommendation
        - Primary risks to the thesis
        - Key metrics to watch going forward
        """,
        expected_output="Investment recommendation with rationale and key metrics",
        agent=strategist,
        context=[financial_task, risk_task]
    )

    report_task = Task(
        description=f"""
        Write a complete investment memo for {company_name} ({ticker}).

        # Investment Memo: {company_name} ({ticker})
        **Date**: [Today]  **Analyst**: Research Team  **Rating**: [Buy/Hold/Sell]

        ## Executive Summary
        ## Company Overview
        ## Financial Analysis
        ### Revenue & Growth
        ### Profitability
        ### Balance Sheet & Cash Flow
        ## Risk Assessment
        ## Investment Recommendation
        ## Key Metrics to Monitor
        ## Appendix: Data Sources
        """,
        expected_output="Complete formatted investment memo in markdown",
        agent=report_writer,
        context=[collect_task, financial_task, risk_task, strategy_task],
        output_file=f"{ticker.lower()}_investment_memo.md"
    )

    return Crew(
        agents=[data_collector, financial_analyst, risk_analyst, strategist, report_writer],
        tasks=[collect_task, financial_task, risk_task, strategy_task, report_task],
        process=Process.sequential,
        verbose=True
    )


if __name__ == "__main__":
    crew = build_investment_crew("Apple Inc.", "AAPL")
    result = crew.kickoff()
    print("\nInvestment memo saved to: aapl_investment_memo.md")
```

---

## Key Concepts Covered

| Concept | Code location |
|---------|---------------|
| 5-agent sequential pipeline | Full research workflow |
| `output_file` in Task | Automatic file output |
| Domain-specific tools | `FinancialSearchTool`, `RiskSearchTool` |
| Task context chaining | All tasks feed final report |
| Risk matrix generation | Structured scoring from documents |

---

## What to Explain in Video

1. Sell-side research report structure (5 min)
2. Splitting financial vs risk analysis into separate agents (5 min)
3. `output_file` parameter — auto-save task output (3 min)
4. How context chaining works in CrewAI (5 min)
5. LLM temperature for financial precision (2 min)
6. Live demo generating Apple investment memo (15 min)
