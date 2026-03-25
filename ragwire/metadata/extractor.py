"""
Metadata extraction using LLM with structured output.

Extracts structured metadata from document content using language models.
Uses Pydantic models and LangChain's with_structured_output for reliable,
type-safe extraction — no manual JSON parsing.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Type

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


# Default Pydantic schema for financial documents
class FinancialMetadata(BaseModel):
    company_name: Optional[str] = Field(
        None,
        description=(
            "Legal or commonly known company name in lowercase. "
            "Use the full legal name if present, otherwise the trading name. "
            "Examples: 'alphabet inc.', 'amazon.com inc.', 'apple inc.', 'microsoft corporation', 'tesla inc.'"
        ),
    )
    doc_type: Optional[str] = Field(
        None,
        description=(
            "SEC filing type. Must be one of: '10-k' (annual report), '10-q' (quarterly report), "
            "'8-k' (current report / material event). "
            "Look for 'Form 10-K', 'Annual Report', 'Form 10-Q', 'Quarterly Report', 'Form 8-K' in the document header or cover page."
        ),
    )
    fiscal_quarter: Optional[str] = Field(
        None,
        description=(
            "Fiscal quarter covered by the document. Must be lowercase: 'q1', 'q2', 'q3', or 'q4'. "
            "Only populate for 10-Q filings. Leave null for 10-K (annual) and 8-K filings. "
            "Determine the quarter from the document itself — look for 'first quarter', 'second quarter', 'Q1', 'Q2', etc. "
            "Do not infer the quarter from calendar months, as fiscal years vary by company."
        ),
    )
    fiscal_year: Optional[List[int]] = Field(
        None,
        description=(
            "List of fiscal year(s) covered by the document as 4-digit integers. "
            "For a 10-K, this is typically one year (e.g. [2024]). "
            "For a 10-Q or 8-K that spans a year boundary, include both (e.g. [2023, 2024]). "
            "Look for 'fiscal year ended', 'year ended', or the filing date year on the cover page. "
            "Examples: 'Annual Report 2024' → [2024], 'Quarter ended March 31, 2025' → [2025]."
        ),
    )


class MetadataExtractor:
    """
    Extract structured metadata from documents using LLM structured output.

    Uses Pydantic models with LangChain's with_structured_output — no manual
    JSON parsing, no type coercion hacks. The LLM returns a validated,
    typed object directly.

    Example:
        >>> extractor = MetadataExtractor(llm)
        >>> metadata = extractor.extract(document_text)
        >>> print(metadata['company_name'])
    """

    EXTRACTION_PROMPT = (
        "You are an expert metadata extraction assistant. Your job is to read the document carefully "
        "and populate every metadata field in the schema with as much detail as the document provides.\n\n"
        "## Extraction Rules\n"
        "1. **Be thorough**: Extract every field you can find. A field should only be null if the "
        "information is completely absent — not because you are unsure.\n"
        "2. **Be precise**: Extract exactly what is stated. Do not infer, assume, or hallucinate "
        "values that are not present in the document.\n"
        "3. **Lists**: For list fields, scan the entire document and extract ALL matching values — "
        "not just the first occurrence.\n"
        "4. **Strings**: Normalize to lowercase. Trim extra whitespace.\n"
        "5. **Integers**: Return the numeric value only — no units, symbols, or surrounding text.\n"
        "6. **Null**: Return null only when the field is genuinely not mentioned anywhere in the document.\n\n"
        "## Document Text\n"
        "{content}\n\n"
        "## Extracted Metadata"
    )

    def __init__(self, llm, schema_model: Optional[Type[BaseModel]] = None, prompt_template: Optional[str] = None):
        """
        Initialize the metadata extractor.

        Args:
            llm: LangChain chat model instance
            schema_model: Pydantic model defining the metadata schema.
                          Defaults to FinancialMetadata if not provided.
            prompt_template: Custom extraction prompt. Must contain a {content}
                             placeholder. Defaults to EXTRACTION_PROMPT if not provided.
        """
        self.llm = llm
        self.schema_model = schema_model or FinancialMetadata
        self.prompt_template = prompt_template or self.EXTRACTION_PROMPT
        if "{content}" not in self.prompt_template:
            raise ValueError("Custom prompt must contain a {content} placeholder for document text.")
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self._structured_llm = llm.with_structured_output(self.schema_model)
        self.fields: Optional[List[str]] = None
        self.extraction_chars: int = 3000

    def extract(self, text: str, stored_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract metadata from document text.

        Args:
            text: Document content to extract metadata from (first 10,000 chars used)
            stored_values: Existing field values from the collection. When provided,
                the LLM is instructed to reuse stored entity names for consistency
                (e.g. always use 'apple inc.' if that is already stored).

        Returns:
            Dictionary containing extracted metadata with correct types.
        """
        existing = "\n".join(
            f"  {k}: {v}" for k, v in (stored_values or {}).items() if v
        )

        if existing:
            grounding = (
                f"Existing values already stored in the collection:\n{existing}\n"
                "If this document refers to the same entity as a stored value, "
                "use the stored value exactly.\n\n"
            )
            injected = self.prompt_template.replace("## Document Text\n", grounding + "## Document Text\n", 1)
            prompt = ChatPromptTemplate.from_template(injected)
        else:
            prompt = self.prompt

        chain = prompt | self._structured_llm
        result = chain.invoke({"content": text[:self.extraction_chars]})

        metadata = result.model_dump()

        # Normalize strings to lowercase
        metadata = {
            k: v.lower().strip() if isinstance(v, str) else
               [i.lower().strip() if isinstance(i, str) else i for i in v] if isinstance(v, list) else v
            for k, v in metadata.items()
        }

        logger.debug(f"Extracted metadata: {metadata}")
        return metadata

    def extract_batch(self, texts: List[str], stored_values: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract metadata from multiple documents.

        Args:
            texts: List of document texts
            stored_values: Existing field values from the collection (see extract())

        Returns:
            List of metadata dictionaries
        """
        results = []
        for text in texts:
            try:
                results.append(self.extract(text, stored_values=stored_values))
            except Exception as e:
                logger.error(f"Failed to extract metadata: {e}")
                results.append({})
        return results

    @classmethod
    def _build_schema_model(cls, fields: List[Dict[str, Any]]) -> Type[BaseModel]:
        """
        Dynamically build a Pydantic model from YAML field definitions.

        Args:
            fields: List of field dicts with name, description, type, values keys

        Returns:
            Pydantic model class
        """
        annotations: Dict[str, Any] = {}

        for field in fields:
            name = field["name"]
            desc = field.get("description", name)
            field_type = field.get("type", "string")
            values = field.get("values")

            # Enrich description with allowed/example values so LLM knows the options
            if values:
                if field_type == "list":
                    desc = f"{desc}. Example values: {', '.join(str(v) for v in values)}"
                else:
                    desc = f"{desc}. Allowed values: {' | '.join(str(v) for v in values)}"

            if field_type == "list":
                annotations[name] = (Optional[List[str]], Field(None, description=desc))
            elif field_type == "integer":
                annotations[name] = (Optional[int], Field(None, description=desc))
            else:
                annotations[name] = (Optional[str], Field(None, description=desc))

        return create_model("ExtractedMetadata", **annotations)

    @classmethod
    def from_yaml(cls, llm, yaml_path: str) -> "MetadataExtractor":
        """
        Create a MetadataExtractor configured from a YAML file.

        The YAML file must contain a 'fields' list. Each field supports:
          - name (required)
          - description (required)
          - type: "string" | "list" | "integer" (default: "string")
          - values: list of example/allowed values (optional)

        Optionally, a top-level 'prompt' key overrides the default extraction
        prompt. Must contain a {content} placeholder.

        Args:
            llm: LangChain chat model instance
            yaml_path: Path to the metadata YAML config file

        Returns:
            MetadataExtractor instance with a dynamically built Pydantic schema
        """
        import yaml

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata config file not found: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            meta_config = yaml.safe_load(f)

        fields = meta_config.get("fields")
        if not fields:
            raise ValueError(
                f"Metadata config '{yaml_path}' must define a 'fields' list"
            )

        custom_prompt = meta_config.get("prompt")

        schema_model = cls._build_schema_model(fields)
        instance = cls(llm, schema_model=schema_model, prompt_template=custom_prompt)
        instance.fields = [f["name"] for f in fields]

        if custom_prompt:
            logger.debug(f"Using custom extraction prompt from {yaml_path}")
        logger.debug(f"Built metadata schema from {len(fields)} field definitions: {instance.fields}")
        return instance

    # Keep for backward compatibility — used by API reference docs
    @classmethod
    def build_prompt_from_fields(cls, fields: List[Dict[str, Any]]) -> str:
        """Deprecated: use from_yaml() instead. Kept for backward compatibility."""
        schema_model = cls._build_schema_model(fields)
        return str(schema_model.model_json_schema())
