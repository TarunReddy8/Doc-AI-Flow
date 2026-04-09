"""
Extraction Service — uses LLMs to extract structured data from OCR text.
Supports prompt versioning, A/B testing, and MLflow tracking.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.extraction import DocumentType

logger = get_logger(__name__)
settings = get_settings()


# ── Prompt Registry (versioned prompts for A/B testing) ──────────────────

PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "invoice_v1": {
        "version": "v1.0",
        "system": (
            "You are a document extraction specialist. Extract structured data "
            "from the provided invoice text. Return ONLY valid JSON matching the "
            "schema exactly. If a field cannot be found, use null."
        ),
        "template": """Extract the following fields from this invoice text:

**Required Fields:**
- invoice_number, invoice_date, due_date
- vendor_name, vendor_address
- customer_name, customer_address
- line_items (array of: description, quantity, unit_price, total)
- subtotal, tax, total_amount, currency, payment_terms

**Invoice Text:**
```
{ocr_text}
```

Return as JSON:
""",
    },
    "invoice_v2": {
        "version": "v2.0",
        "system": (
            "You are an expert financial document processor. Your task is to "
            "extract every piece of structured data from invoices with maximum "
            "precision. When data is ambiguous, provide your best interpretation "
            "and note the ambiguity. Return ONLY valid JSON."
        ),
        "template": """Carefully analyze this invoice and extract ALL structured data.

**Step-by-step approach:**
1. First identify the vendor and customer sections
2. Locate invoice metadata (number, dates, terms)
3. Parse each line item with quantities and prices
4. Calculate and verify totals

**Invoice Text:**
```
{ocr_text}
```

Return as JSON with these fields:
{{
  "invoice_number": str | null,
  "invoice_date": str | null,
  "due_date": str | null,
  "vendor_name": str | null,
  "vendor_address": str | null,
  "customer_name": str | null,
  "customer_address": str | null,
  "line_items": [{{ "description": str, "quantity": float, "unit_price": float, "total": float }}],
  "subtotal": float | null,
  "tax": float | null,
  "total_amount": float | null,
  "currency": str | null,
  "payment_terms": str | null
}}
""",
    },
    "contract_v1": {
        "version": "v1.0",
        "system": (
            "You are a legal document analysis specialist. Extract key terms "
            "and metadata from contracts. Return ONLY valid JSON."
        ),
        "template": """Extract the following from this contract text:

**Required Fields:**
- contract_title, parties (list), effective_date, expiration_date
- contract_value, key_terms (list), governing_law, termination_clause

**Contract Text:**
```
{ocr_text}
```

Return as JSON:
""",
    },
    "classify_v1": {
        "version": "v1.0",
        "system": "You classify documents into types. Return ONLY a single word.",
        "template": """Classify this document into one of: invoice, contract, report, receipt

**Document Text (first 500 chars):**
```
{ocr_text}
```

Type:""",
    },
}


class ExtractionService:
    """Orchestrates LLM-powered data extraction from OCR text."""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self._llm = None
        self._mock_mode = self.provider == "mock"

        if self._mock_mode:
            logger.info("extraction_mode", mode="mock", detail="No API key needed")
        else:
            self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client based on config."""
        try:
            if self.provider == "openai":
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=self.model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    api_key=settings.openai_api_key,
                )
            elif self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                self._llm = ChatAnthropic(
                    model=self.model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    api_key=settings.anthropic_api_key,
                )
            elif self.provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                    google_api_key=settings.gemini_api_key,
                )
            elif self.provider == "groq":
                from langchain_groq import ChatGroq

                self._llm = ChatGroq(
                    model=self.model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    api_key=settings.groq_api_key,
                )
            logger.info("llm_initialized", provider=self.provider, model=self.model)
        except Exception as e:
            logger.error("llm_init_failed", error=str(e))

    async def classify_document(self, ocr_text: str) -> DocumentType:
        """Auto-detect document type using LLM classification."""
        if self._mock_mode:
            from app.services.mock_extraction import mock_classify

            return mock_classify(ocr_text)

        prompt_config = PROMPT_REGISTRY["classify_v1"]
        truncated = ocr_text[:500]

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=prompt_config["system"]),
                    HumanMessage(
                        content=prompt_config["template"].format(ocr_text=truncated)
                    ),
                ]
            )

            result = response.content.strip().lower()
            for doc_type in DocumentType:
                if doc_type.value in result:
                    return doc_type
            return DocumentType.UNKNOWN
        except Exception as e:
            logger.warning("classification_failed", error=str(e))
            return DocumentType.UNKNOWN

    async def extract(
        self,
        ocr_text: str,
        document_type: DocumentType,
        prompt_version: str | None = None,
    ) -> tuple[dict[str, Any], str, float]:
        """
        Extract structured data from OCR text using the appropriate prompt.

        Returns: (extracted_data, prompt_version_used, confidence_score)
        """
        # Mock mode — no API key needed
        if self._mock_mode:
            from app.services.mock_extraction import mock_extract

            return mock_extract(ocr_text, document_type)

        start_time = time.time()

        # Select prompt
        prompt_key = self._select_prompt(document_type, prompt_version)
        prompt_config = PROMPT_REGISTRY[prompt_key]
        version_used = prompt_config["version"]

        logger.info(
            "extraction_start",
            doc_type=document_type.value,
            prompt=prompt_key,
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=prompt_config["system"]),
                    HumanMessage(
                        content=prompt_config["template"].format(ocr_text=ocr_text)
                    ),
                ]
            )

            raw_output = response.content.strip()
            extracted = self._parse_json_output(raw_output)
            confidence = self._calculate_confidence(extracted, document_type)

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                "extraction_complete",
                doc_type=document_type.value,
                confidence=confidence,
                fields_extracted=len([v for v in extracted.values() if v is not None]),
                time_ms=round(elapsed, 2),
            )

            return extracted, version_used, confidence

        except Exception as e:
            logger.error("extraction_failed", error=str(e))
            return {}, version_used, 0.0

    def _select_prompt(self, doc_type: DocumentType, version: str | None) -> str:
        """Select the appropriate prompt key."""
        if version:
            key = f"{doc_type.value}_{version}"
            if key in PROMPT_REGISTRY:
                return key

        # Default to latest version
        type_prompts = [k for k in PROMPT_REGISTRY if k.startswith(doc_type.value)]
        if type_prompts:
            return sorted(type_prompts)[-1]  # Latest version

        return "invoice_v2"  # Fallback

    def _parse_json_output(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling markdown code blocks."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON in the output
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("json_parse_failed", raw_length=len(raw))
            return {"raw_output": raw}

    def _calculate_confidence(
        self, data: dict[str, Any], doc_type: DocumentType
    ) -> float:
        """
        Calculate extraction confidence based on field completeness.
        This is a heuristic — production systems would use ground truth.
        """
        if doc_type == DocumentType.INVOICE:
            critical_fields = [
                "invoice_number",
                "vendor_name",
                "total_amount",
                "invoice_date",
            ]
            important_fields = [
                "customer_name",
                "line_items",
                "subtotal",
                "tax",
                "currency",
            ]
        elif doc_type == DocumentType.CONTRACT:
            critical_fields = ["contract_title", "parties", "effective_date"]
            important_fields = [
                "expiration_date",
                "contract_value",
                "key_terms",
                "governing_law",
            ]
        else:
            # Generic scoring
            non_null = sum(1 for v in data.values() if v is not None and v != "")
            return min(non_null / max(len(data), 1), 1.0)

        critical_score = sum(
            1 for f in critical_fields if data.get(f) is not None and data.get(f) != ""
        ) / len(critical_fields)

        important_score = sum(
            1 for f in important_fields if data.get(f) is not None and data.get(f) != ""
        ) / len(important_fields)

        # Critical fields weighted 70%, important fields 30%
        return round(critical_score * 0.7 + important_score * 0.3, 4)


# Singleton
_extraction_service: ExtractionService | None = None


def get_extraction_service() -> ExtractionService:
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = ExtractionService()
    return _extraction_service
