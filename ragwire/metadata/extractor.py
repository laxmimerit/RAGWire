"""
Metadata extraction using LLM.

Extracts structured metadata from document content using language models.
Extracts structured metadata from documents using a language model.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract structured metadata from documents using LLM.

    This extractor uses a language model to parse document content
    and extract metadata like company name, document type, fiscal period, etc.

    Attributes:
        llm: Language model instance for extraction
        prompt_template: Prompt template for metadata extraction

    Example:
        >>> extractor = MetadataExtractor(llm)
        >>> metadata = extractor.extract(document_text)
        >>> print(metadata['company_name'])
    """

    # Default prompt template for financial documents
    PROMPT_TEMPLATE = """
You are a financial document metadata extractor. Extract structured metadata from the following document text.

Return ONLY valid JSON in this exact format:
{{
  "company_name": "company name in lowercase",
  "doc_type": "10-k|10-q|8-k",
  "fiscal_quarter": "q1|q2|q3|q4|null",
  "fiscal_year": [year1, year2, ...]
}}

Document Text:
{content}

Extracted Metadata (JSON only):
"""

    def __init__(self, llm, prompt_template: Optional[str] = None):
        """
        Initialize the metadata extractor.

        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI, ChatOpenAI)
            prompt_template: Optional custom prompt template
        """
        self.llm = llm
        self.prompt_template = prompt_template or self.PROMPT_TEMPLATE

        # Create prompt chain if langchain is available
        try:
            from langchain_core.prompts import ChatPromptTemplate

            self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
            self.has_langchain = True
        except ImportError:
            self.has_langchain = False
            logger.warning(
                "LangChain not available. Using basic template substitution."
            )

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from document text.

        Args:
            text: Document content to extract metadata from

        Returns:
            Dictionary containing extracted metadata

        Raises:
            ValueError: If LLM response is not valid JSON
        """
        # Prepare the prompt
        if self.has_langchain:
            # Use langchain prompt template
            chain = self.prompt | self.llm
            response = chain.invoke({"content": text[:10000]})
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
        else:
            # Fallback to simple template substitution
            prompt = self.prompt_template.format(content=text[:10000])
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

        # Parse JSON response
        try:
            metadata = self._parse_json_response(response_text)
            logger.debug(f"Extracted metadata: {metadata}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response was: {response_text}")
            raise ValueError(f"LLM did not return valid JSON: {response_text}")

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling various formats.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON as dictionary
        """
        # Try to extract JSON from markdown code blocks
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # Try to find JSON in the response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return json.loads(response_text)

    @classmethod
    def build_prompt_from_fields(cls, fields: List[Dict[str, Any]]) -> str:
        """
        Build a JSON extraction prompt from a list of field definitions.

        Each field dict should have:
          - name: field name (required)
          - description: human-readable description (required)
          - values: list of allowed values (optional)

        Args:
            fields: List of field definition dicts

        Returns:
            Prompt template string with a {content} placeholder
        """
        json_lines = []
        for field in fields:
            name = field["name"]
            desc = field.get("description", name)
            values = field.get("values")
            if values:
                hint = "|".join(str(v) for v in values)
                json_lines.append(f'  "{name}": "{hint}"  // {desc}')
            else:
                json_lines.append(f'  "{name}": ...  // {desc}')

        json_block = "{\n" + ",\n".join(json_lines) + "\n}"

        return (
            "Extract metadata from the following document. Return ONLY valid JSON:\n"
            f"{json_block}\n\n"
            "Document Text:\n{content}\n\n"
            "Extracted Metadata (JSON only):\n"
        )

    @classmethod
    def from_yaml(cls, llm, yaml_path: str) -> "MetadataExtractor":
        """
        Create a MetadataExtractor configured from a YAML file.

        The YAML file may contain:
          - fields: list of {name, description, values} dicts (required if no prompt_template)
          - prompt_template: full custom prompt (optional — overrides auto-built prompt)

        Args:
            llm: Language model instance
            yaml_path: Path to the metadata YAML config file

        Returns:
            MetadataExtractor instance
        """
        import yaml

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata config file not found: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            meta_config = yaml.safe_load(f)

        prompt_template = meta_config.get("prompt_template")
        if not prompt_template:
            fields = meta_config.get("fields")
            if not fields:
                raise ValueError(
                    f"Metadata config '{yaml_path}' must define either 'fields' or 'prompt_template'"
                )
            prompt_template = cls.build_prompt_from_fields(fields)
            logger.debug(f"Built metadata prompt from {len(fields)} field definitions")

        return cls(llm, prompt_template=prompt_template)

    def extract_batch(self, texts: list) -> list:
        """
        Extract metadata from multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of metadata dictionaries
        """
        results = []
        for text in texts:
            try:
                metadata = self.extract(text)
                results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to extract metadata: {e}")
                results.append(
                    {
                        "company_name": None,
                        "doc_type": None,
                        "fiscal_quarter": None,
                        "fiscal_year": [],
                    }
                )
        return results
