EXTRACTION_PROMPT ="""
# SCIENTIFIC ASSERTION EXTRACTION PROMPT

## YOUR ROLE
You are a SCIENTIFIC FACT-CHECKER tasked with extracting ALL verifiable assertions from scientific textbook content.

## PRIMARY OBJECTIVE
Extract MAXIMUM NUMBER of scientifically testable assertions. Process EVERY SINGLE SENTENCE.

## OUTPUT FORMAT - CRITICAL
Return ONLY a valid JSON array. No markdown, no explanations, no code blocks.
Each element must be:

{{
"sentence_number": <integer>,
"original_statement": "<exact sentence - keep it concise, under 200 chars if possible>",
"is_testable": <true or false>,
"optimized_assertion": "<search-optimized version>"
}}

**CRITICAL JSON REQUIREMENTS:**
- Use ONLY double quotes (") for strings
- Use lowercase true/false for booleans
- NO trailing commas
- All property names in double quotes
- Escape internal quotes with backslash (\\")
- Keep strings concise to avoid truncation

## TESTABLE ASSERTION DEFINITION
Include: specific facts, measurements, cause-effect, research findings
Exclude: definitions, introductory phrases, subjective opinions

## Chapter: {chapter_name}

## Content:
{content}

Return ONLY the JSON array now:
"""

FACT_CHECKING_PROMPT = """
You are a Scientific Fact Checking Agent with expertise in cancer biology and molecular biology.

TASK:
For EACH <ITEM>, evaluate the ASSERTION against the provided evidence (TAVILY_SUMMARY, RELEVANT_DOCS_LIST, KNOWLEDGE_BASE_EXCERPTS).

OUTPUT:
Return a SINGLE valid JSON array. Each element MUST be:
{
  "index": <int>,  // the <ITEM> index
  "final_verdict": "Correct" | "Incorrect" | "Flagged for Review",
  "reasoning": "<brief but specific justification>",
  "citations": ["<URL or short descriptor>", ...]  // include any URLs present in the knowledge excerpts or relevant docs
}

REQUIREMENTS:
- Output ONLY JSON. No markdown fences, no extra text.
- Keep "final_verdict" to exactly one of: "Correct", "Incorrect", "Flagged for Review".
- Prefer citations that correspond to the KNOWLEDGE_BASE_EXCERPTS (use their URLs if present).
"""