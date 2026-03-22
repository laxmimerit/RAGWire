"""
Metadata extraction using LLM.

Extracts structured metadata from document content using language models.
Extracts structured metadata from documents using a language model.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract structured metadata from documents using LLM.

    This extractor uses a language model to parse document content
    and extract finance-specific metadata like company name,
    document type, fiscal period, etc.

    Attributes:
        llm: Language model instance for extraction
        prompt_template: Prompt template for metadata extraction

    Example:
        >>> extractor = MetadataExtractor(llm)
        >>> metadata = extractor.extract(document_text)
        >>> print(metadata['company_name'])
    """

    # Prompt template for metadata extraction
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
