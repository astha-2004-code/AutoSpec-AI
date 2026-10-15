"""Evidence-grounded specification extractor with strict LLM prompts and deterministic fallback."""

from __future__ import annotations

import json
import logging
import re

from autospec.chunker import SpecificationChunk
from autospec.config import settings
from autospec.schemas import ExtractionMethod, SpecType, VehicleSpecification, VerificationStatus

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are an evidence-grounded automotive specification extraction specialist.
Your mission is to extract vehicle maintenance and repair specifications (fastener torque, fluid capacities, clearances, tolerances, and part numbers) exclusively from the provided workshop manual passages.

CRITICAL RULES:
1. ONLY extract values directly and explicitly stated in the context text.
2. DO NOT calculate, guess, round, or extrapolate numbers.
3. If a value is given in primary (e.g. Nm) and alternate units (e.g. lb-ft), populate both 'value'/'unit' and 'alternate_value'/'alternate_unit'.
4. Strictly maintain section context (e.g. distinguish Front Disc Brake vs Rear Disc Brake).
5. For each extracted record, provide the EXACT verbatim quote from the text in 'source_quote'.
6. If the requested information is NOT in the context, return an empty list: [].

Output must be a JSON array of objects conforming to:
[
  {
    "component": "Component or fastener name",
    "spec_type": "torque" | "fluid_capacity" | "dimension_clearance" | "part_number",
    "value": "Primary numerical value",
    "unit": "Nm" | "ml" | "mm" | "L",
    "alternate_value": "Alternate numerical value or null",
    "alternate_unit": "lb-ft" | "lb-in" | "pt" | "in" | null,
    "part_number": "Manufacturer part number or null",
    "material": "Material or Ford spec code or null",
    "section": "Section header",
    "page_number": 123,
    "source_quote": "Verbatim quote supporting this specification"
  }
]
"""


class SpecificationExtractor:
    """Extracts verified vehicle specifications using either an LLM or deterministic fallback."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        self.model_name = model_name or settings.llm_model
        self.api_key = api_key or settings.llm_api_key

    def extract(
        self,
        query: str,
        retrieved_chunks: list[tuple[SpecificationChunk, float]],
    ) -> list[VehicleSpecification]:
        """Extract specifications from retrieved chunks using configured provider or fallback."""
        if not retrieved_chunks:
            return []

        # If LLM provider is active and has an API key, attempt LLM extraction
        if self.provider in ("openai", "gemini", "anthropic", "groq") and self.api_key:
            try:
                specs = self._extract_with_llm(query, retrieved_chunks)
                if specs:
                    return specs
            except Exception as err:
                logger.warning("LLM extraction failed (%s); reverting to deterministic fallback.", err)

        # High-precision deterministic fallback
        return self._extract_with_fallback(query, retrieved_chunks)

    def _extract_with_fallback(
        self,
        query: str,
        retrieved_chunks: list[tuple[SpecificationChunk, float]],
    ) -> list[VehicleSpecification]:
        """High-precision offline deterministic extractor using structured chunk data and pattern matching."""
        extracted: list[VehicleSpecification] = []
        q_lower = query.lower()

        stopwords = {
            "what",
            "is",
            "the",
            "for",
            "in",
            "to",
            "at",
            "how",
            "much",
            "of",
            "and",
            "or",
            "torque",
            "spec",
            "specs",
            "specification",
            "specifications",
            "value",
            "values",
            "procedure",
            "tighten",
            "tightening",
            "requirement",
            "requirements",
        }
        raw_words = [w.strip("?,.:;\"'()") for w in q_lower.split() if len(w.strip("?,.:;\"'()")) > 2]
        discriminative = [w for w in raw_words if w not in stopwords]

        is_part_no_query = any(term in q_lower for term in ("part number", "part #", "part no", "spec code"))
        is_capacity_query = any(term in q_lower for term in ("capacity", "fluid", "fill", "oil", "how much", "quarts", "pints", "liters", "ml"))

        for chunk, score in retrieved_chunks:
            combined_text = f"{chunk.component} {chunk.content} {chunk.raw_evidence}".lower()

            # Zero hallucination: reject chunks if query contains technical nouns not found anywhere in chunk
            if discriminative:
                matching_words = [w for w in discriminative if w in combined_text]
                if not matching_words:
                    continue

            # Case 1: The retrieved chunk is already an explicitly parsed specification row
            if chunk.chunk_type == "spec_row" and chunk.component and (chunk.primary_value or chunk.part_number):
                spec_type = chunk.metadata.get("spec_type", SpecType.TORQUE.value)
                source_q = chunk.raw_evidence or f"{chunk.component} {chunk.primary_value} {chunk.primary_unit}"

                val = chunk.primary_value
                unit = chunk.primary_unit

                if is_part_no_query and (chunk.part_number or spec_type == "part_number"):
                    val = chunk.part_number or chunk.primary_value
                    unit = "part_number"
                    spec_type = SpecType.PART_NUMBER.value
                elif is_capacity_query and spec_type == "fluid_capacity":
                    val = chunk.primary_value
                    unit = "capacity"

                spec = VehicleSpecification(
                    component=chunk.component,
                    spec_type=spec_type,
                    value=val,
                    unit=unit,
                    alternate_value=chunk.alternate_value or None,
                    alternate_unit=chunk.alternate_unit or None,
                    part_number=chunk.part_number or None,
                    material=chunk.metadata.get("ford_spec"),
                    section=f"SECTION {chunk.section_id}: {chunk.section_title}",
                    subsystem=chunk.subsystem,
                    page_number=chunk.page_number,
                    source_document=chunk.document_name,
                    source_quote=source_q,
                    retrieval_score=round(score, 4),
                    verification_status=VerificationStatus.UNVERIFIED.value,
                    extraction_method=ExtractionMethod.DETERMINISTIC_FALLBACK.value,
                )
                extracted.append(spec)

            # Case 2: The chunk is a text passage: scan with regular expressions
            elif chunk.chunk_type == "text_passage":
                passage_specs = self._scan_passage_for_specifications(chunk, score)
                extracted.extend(passage_specs)

        # Deduplicate and rank results by component keyword overlap
        def compute_relevance(s: VehicleSpecification) -> tuple[int, float]:
            comp_l = s.component.lower()
            overlap = sum(1 for w in discriminative if w in comp_l)
            # Bonus if spec_type matches query intent
            type_bonus = 1 if (is_part_no_query and s.spec_type == "part_number") or (is_capacity_query and s.spec_type == "fluid_capacity") else 0
            return (overlap + type_bonus, s.retrieval_score)

        extracted.sort(key=compute_relevance, reverse=True)

        unique_specs: list[VehicleSpecification] = []
        seen = set()
        for s in extracted:
            key = (s.component.lower(), s.value, s.section, s.page_number)
            if key not in seen:
                seen.add(key)
                unique_specs.append(s)

        return unique_specs

    def _scan_passage_for_specifications(
        self,
        chunk: SpecificationChunk,
        score: float,
    ) -> list[VehicleSpecification]:
        """Regex-based specification parser for raw narrative passages."""
        specs: list[VehicleSpecification] = []
        text = chunk.raw_evidence or chunk.content

        # Pattern 1: Fastener torque: '... bolts to 250 Nm (184 lb-ft)' or '... bolts 35 Nm 26 lb-ft'
        torque_pattern = re.compile(
            r"([A-Za-z0-9\s\-]+?(?:bolts?|nuts?|screws?|valve|fitting))\s+(?:to|at)?\s*([0-9\.]+)\s*(Nm|lb-ft|lb-in)(?:\s*(?:\(|\/)?\s*([0-9\.]+)\s*(lb-ft|lb-in|Nm)\)?)?",
            re.IGNORECASE,
        )

        for match in torque_pattern.finditer(text):
            comp = match.group(1).strip()
            val1 = match.group(2).strip()
            unit1 = match.group(3).strip()
            val2 = match.group(4).strip() if match.group(4) else None
            unit2 = match.group(5).strip() if match.group(5) else None

            specs.append(
                VehicleSpecification(
                    component=comp,
                    spec_type=SpecType.TORQUE.value,
                    value=val1,
                    unit=unit1,
                    alternate_value=val2,
                    alternate_unit=unit2,
                    section=f"SECTION {chunk.section_id}: {chunk.section_title}",
                    subsystem=chunk.subsystem,
                    page_number=chunk.page_number,
                    source_document=chunk.document_name,
                    source_quote=match.group(0).strip(),
                    retrieval_score=round(score, 4),
                    verification_status=VerificationStatus.UNVERIFIED.value,
                    extraction_method=ExtractionMethod.DETERMINISTIC_FALLBACK.value,
                )
            )

        return specs

    def _extract_with_llm(
        self,
        query: str,
        retrieved_chunks: list[tuple[SpecificationChunk, float]],
    ) -> list[VehicleSpecification]:
        """Invoke external LLM provider using structured JSON output."""
        # Format context passages
        context_blocks = []
        for i, (chunk, _score) in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"--- EVIDENCE EXCERPT {i} (Page {chunk.page_number}, Section: {chunk.section_title}) ---\n"
                f"{chunk.content}\n"
                f"Raw Text:\n{chunk.raw_evidence}\n"
            )

        context_str = "\n".join(context_blocks)
        user_message = f"Query: {query}\n\nEvidence Context:\n{context_str}"

        # Standard OpenAI client integration
        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=settings.llm_base_url or None,
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed_data = json.loads(raw_content)

            # Handle either a root array or an object with 'specifications' key
            items = parsed_data if isinstance(parsed_data, list) else parsed_data.get("specifications", parsed_data.get("results", []))

            specs: list[VehicleSpecification] = []
            for item in items:
                spec = VehicleSpecification(
                    component=item.get("component", "Vehicle Component"),
                    spec_type=item.get("spec_type", SpecType.TORQUE.value),
                    value=str(item.get("value", "")),
                    unit=str(item.get("unit", "")),
                    alternate_value=str(item.get("alternate_value")) if item.get("alternate_value") else None,
                    alternate_unit=str(item.get("alternate_unit")) if item.get("alternate_unit") else None,
                    part_number=item.get("part_number"),
                    material=item.get("material"),
                    section=item.get("section", retrieved_chunks[0][0].section_title),
                    subsystem=retrieved_chunks[0][0].subsystem,
                    page_number=int(item.get("page_number", retrieved_chunks[0][0].page_number)),
                    source_document=retrieved_chunks[0][0].document_name,
                    source_quote=str(item.get("source_quote", "")),
                    retrieval_score=round(retrieved_chunks[0][1], 4),
                    verification_status=VerificationStatus.UNVERIFIED.value,
                    extraction_method=ExtractionMethod.LLM_EXTRACTION.value,
                )
                specs.append(spec)
            return specs

        return []
