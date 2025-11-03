import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import google.genai as genai
import pandas as pd
from dotenv import load_dotenv

from config import Config
from rag import SimpleRAGSystem
from prompt import EXTRACTION_PROMPT, FACT_CHECKING_PROMPT
from document_retrieval import ComprehensiveDocumentRetrieval

# Suppress lxml warning
import warnings
warnings.filterwarnings('ignore', message='.*lxml.*does not provide the extra.*')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fact_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScientificFactChecker:
    """Main class for scientific fact checking system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.rag_system = SimpleRAGSystem(
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
        
        # Primary: Free comprehensive document retrieval
        self.document_retrieval = ComprehensiveDocumentRetrieval(email=config.pubmed_email)
        
        # Fallback: Tavily (optional, only if API key provided)
        self.tavily_client = None
        if config.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
                logger.info("✓ Tavily fallback available")
            except Exception as e:
                logger.warning(f"⚠️  Tavily not available: {e}")
        
        # Configure Gemini
        self.gemini_client = genai.Client(api_key=config.google_api_key)

    def extract_assertions(self, chapter_path: str) -> List[Dict[str, str]]:
        """Extract testable assertions from a markdown chapter"""
        logger.info(f"Extracting assertions from: {chapter_path}")
        
        try:
            with open(chapter_path, "r", encoding="utf-8") as file:
                chapter_content = file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {chapter_path}")
            return []
        
        chapter_name = Path(chapter_path).stem.replace("_", " ")
        
        prompt = self._create_extraction_prompt(chapter_name, chapter_content)
        
        try:
            model_name = "gemini-2.5-flash"
            
            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=64000,
                    temperature=0.1,
                )
            )
            
            extracted_data = self._parse_extraction_response(response.text)
            
            testable_assertions = [
                {
                    "Original Statement": item["original_statement"],
                    "Assertion": item["optimized_assertion"]
                }
                for item in extracted_data if item.get("is_testable", False)
            ]
            
            logger.info(f"Extracted {len(testable_assertions)} testable assertions")
            return testable_assertions
            
        except Exception as e:
            logger.error(f"Error extracting assertions: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def _create_extraction_prompt(self, chapter_name: str, content: str) -> str:
        """Create prompt for assertion extraction with stricter JSON formatting"""
        return EXTRACTION_PROMPT.format(chapter_name=chapter_name, content=content)
    
    def _clean_json_text(self, json_text: str) -> str:
        """Clean and fix common JSON formatting issues"""
        if not json_text:
            return "[]"
        
        json_text = json_text.strip()
        
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
        
        json_text = re.sub(r'}\s*\n\s*{', '},\n{', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r'\bTrue\b', 'true', json_text)
        json_text = re.sub(r'\bFalse\b', 'false', json_text)
        json_text = re.sub(r'\bNone\b', 'null', json_text)
        json_text = re.sub(r'([,{\[]\s*)([a-zA-Z_]\w*)(\s*):', r'\1"\2"\3:', json_text)
        json_text = re.sub(r'^(\s*)([a-zA-Z_]\w*)(\s*):', r'\1"\2"\3:', json_text, flags=re.MULTILINE)
        
        lines = json_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'//.*$', '', line)
            line = re.sub(r'#.*$', '', line)
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        json_text = '\n'.join(cleaned_lines)
        
        if not json_text.startswith('['):
            json_text = '[' + json_text
        if not json_text.endswith(']'):
            json_text = json_text + ']'
        
        open_braces = json_text.count('{') - json_text.count('}')
        open_brackets = json_text.count('[') - json_text.count(']')
        
        if open_braces > 0:
            json_text = json_text + ('}' * open_braces)
        if open_brackets > 0:
            json_text = json_text + (']' * open_brackets)
        
        return json_text

    def _parse_extraction_response(self, response_text: str) -> List[Dict]:
        """Parse the extraction response with improved error handling"""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response_text)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response_text
        
        json_text = self._clean_json_text(json_text)
        
        try:
            data = json.loads(json_text)
            
            if not isinstance(data, list):
                logger.warning("JSON is not a list, attempting to wrap")
                data = [data] if isinstance(data, dict) else []
            
            valid_data = []
            for item in data:
                if isinstance(item, dict) and 'original_statement' in item:
                    orig = item.get('original_statement', '')
                    
                    if orig and not orig.endswith('...') and len(orig) > 20:
                        valid_data.append(item)
                    else:
                        logger.debug(f"Skipping malformed entry: {orig[:50]}")
            
            logger.info(f"Successfully parsed {len(valid_data)} valid assertions")
            return valid_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Problematic JSON (first 1000 chars): {json_text[:1000]}")
            
            logger.info("Attempting fallback extraction...")
            fallback_data = self._extract_fallback_data(response_text)
            if fallback_data:
                logger.info(f"Fallback extraction successful: {len(fallback_data)} items")
                return fallback_data
            
            return []
    
    def _extract_fallback_data(self, response_text: str) -> List[Dict]:
        """Fallback method to extract data when JSON parsing fails"""
        fallback_data = []
        
        sentences = re.split(r'[.!?]+', response_text)
        
        sentence_number = 1
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                testable_keywords = [
                    'is', 'are', 'was', 'were', 'has', 'have', 'had',
                    'contains', 'consists', 'includes', 'comprises',
                    'found', 'discovered', 'identified', 'observed',
                    'shows', 'demonstrates', 'indicates', 'suggests',
                    'causes', 'leads', 'results', 'produces',
                    'percentage', 'percent', 'number', 'amount',
                    'size', 'diameter', 'length', 'width', 'height'
                ]
                
                is_testable = any(keyword in sentence.lower() for keyword in testable_keywords)
                
                fallback_data.append({
                    "sentence_number": sentence_number,
                    "original_statement": sentence,
                    "is_testable": is_testable,
                    "optimized_assertion": sentence[:100] + "..." if len(sentence) > 100 else sentence
                })
                sentence_number += 1
        
        return fallback_data[:50]
    
    def find_relevant_documents(
        self, 
        assertions: List[Dict[str, str]],
        use_tavily: bool = False
    ) -> List[Dict[str, str]]:
        """
        Find relevant documents for each assertion using free APIs or Tavily fallback
        
        Args:
            assertions: List of assertions to search
            use_tavily: If True, use Tavily instead of free sources (for retries)
        """
        logger.info(f"Finding relevant documents for {len(assertions)} assertions")
        logger.info(f"Using: {'Tavily (fallback)' if use_tavily else 'Free APIs (primary)'}")
        
        results = []
        
        for index, assertion in enumerate(assertions):
            logger.info(f"Processing {index+1}/{len(assertions)}: {assertion['Original Statement'][:100]}...")
            
            result_obj = {
                "Statement": assertion['Original Statement'],
                "Assertion": assertion['Assertion'],
                "Search Query": assertion['Original Statement'],
                "Summary (Tavily Answer)": "",
                "Relevant Docs": "",
                "Raw Content": "",
                "Source Type": "Tavily" if use_tavily else "Free APIs"
            }
            
            try:
                if use_tavily and self.tavily_client:
                    search_response = self._search_with_tavily(assertion['Original Statement'])
                else:
                    search_response = self.document_retrieval.search_and_extract(
                        query=assertion['Original Statement'],
                        max_results=2
                    )
                
                num_results = len(search_response.get("results", []))
                
                if num_results == 0:
                    logger.warning(f"⚠️  No documents found for claim #{index + 1}")
                    result_obj["Summary (Tavily Answer)"] = "No documents found"
                    result_obj["Relevant Docs"] = "No relevant documents found"
                    result_obj["Raw Content"] = "No content available"
                else:
                    result_obj["Summary (Tavily Answer)"] = search_response.get("answer", "No summary available")
                    
                    formatted_results = []
                    for i, doc in enumerate(search_response["results"], 1):
                        formatted_results.append(f"{i}. {doc['title']} - {doc['url']}")
                    result_obj["Relevant Docs"] = "\n".join(formatted_results)
                    
                    all_content = []
                    for doc in search_response["results"]:
                        content = doc.get('content', '')
                        if content:
                            all_content.append(f"Source: {doc['title']}\nURL: {doc['url']}\n\n{content}")
                    
                    result_obj["Raw Content"] = "\n\n---\n\n".join(all_content)
                    
                    full_text_count = sum(1 for doc in search_response["results"] if doc.get('full_text'))
                    logger.info(f"  ✅ Retrieved {num_results} papers, {full_text_count} with full text")
                    
            except Exception as e:
                logger.error(f"Error searching for query '{assertion['Original Statement'][:100]}...': {e}")
                import traceback
                traceback.print_exc()
                result_obj["Summary (Tavily Answer)"] = f"Error: {str(e)}"
                result_obj["Relevant Docs"] = f"Error: {str(e)}"
                result_obj["Raw Content"] = f"Error: {str(e)}"
            
            results.append(result_obj)
            time.sleep(1)
        
        logger.info(f"✅ Document search completed for {len(results)} assertions")
        return results
    
    def _search_with_tavily(self, query: str) -> Dict[str, Any]:
        """
        Search using Tavily API (fallback method)
        Returns same format as free document retrieval for compatibility
        """
        try:
            logger.info(f"🔍 Using Tavily fallback for: {query[:50]}...")
            
            search_response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                include_domains=self.config.search_domains,
                max_results=2,
                include_raw_content=True,
                include_answer=True
            )
            
            results = []
            
            if search_response and "results" in search_response and search_response["results"]:
                for doc in search_response["results"]:
                    results.append({
                        'title': doc.get('title', 'No title'),
                        'url': doc.get('url', 'No URL'),
                        'content': doc.get('content', '') or doc.get('raw_content', ''),
                        'full_text': doc.get('raw_content', ''),
                        'abstract': doc.get('content', ''),
                        'source': 'Tavily',
                        'doi': ''
                    })
                
                logger.info(f"✓ Tavily found {len(results)} results")
            
            answer = search_response.get("answer", "No answer provided by Tavily")
            
            return {
                'results': results,
                'answer': answer
            }
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {'results': [], 'answer': 'Tavily search failed'}
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents into a string for storage"""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"{i}. {doc.get('title', 'No title')} - {doc.get('url', 'No URL')}")
        return "\n".join(formatted_docs)
    
    def build_knowledge_base(self, documents_data: List[Dict[str, str]]) -> str:
        """Build a knowledge base from extracted document content"""
        logger.info("Building knowledge base from document content")
        
        cleaned_blocks = []
        processed_items = 0
        
        for i, item in enumerate(documents_data):
            raw_content = item.get('Raw Content', '')
            if raw_content and raw_content not in ["No content available", "No URLs found to extract content from", "No content available"]:
                cleaned = self._clean_content(raw_content)
                if cleaned:
                    cleaned_blocks.append(cleaned)
                    processed_items += 1
                    logger.info(f"Processed item {i+1}: {len(cleaned)} characters")
        
        if cleaned_blocks:
            final_text = "\n\n".join(cleaned_blocks)
            logger.info(f"Knowledge base built with {processed_items} items, {len(final_text)} characters")
            return final_text
        else:
            logger.warning("No content to build knowledge base from")
            return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing unnecessary elements"""
        if not content or content in ["No content found in response", "No content found in extracted result"]:
            return ""
        
        patterns_to_remove = [
            r'!\[Image[^\]]*\]\([^)]*\)',
            r'https?://[^\s\)]+',
            r'www\.[^\s\)]+',
            r'<[^>]+>',
            r'Skip to main content',
            r'NCBI Homepage.*?MyNCBI Homepage.*?Main Content.*?Main Navigation',
            r'\[Log in\].*?Log out.*?',
            r'References.*?',
            r'Copyright.*?Bookshelf ID:.*?',
            r'View on publisher site.*?Download PDF.*?Add to Collections.*?Cite.*?Permalink.*?',
            r'Back to Top.*?',
            r'Follow NCBI.*?',
            r'Add to Collections.*?',
            r'Cite.*?',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

            content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)
        
        return content.strip()

    def _parse_relevant_docs_entries(self, relevant_docs_str: str) -> List[Dict[str, str]]:
        """Parse the human-readable 'Relevant Docs' string into [{'title', 'url'}, ...]"""
        entries = []
        if not relevant_docs_str:
            return entries
        for line in relevant_docs_str.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^\d+\.\s*', '', line)
            if ' - ' in line:
                title, url = line.rsplit(' - ', 1)
                title, url = title.strip(), url.strip()
                if url.startswith('http'):
                    entries.append({"title": title or "Untitled", "url": url})
        return entries

    def _build_kb_blocks_from_documents(self, documents_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Produce structured KB blocks with metadata"""
        blocks = []
        SEP = "\n\n---\n\n"
        for item in documents_data:
            relevant_docs = self._parse_relevant_docs_entries(item.get("Relevant Docs", ""))
            raw = item.get("Raw Content", "") or ""
            if not raw or raw in ["No content available", "No URLs found to extract content from"]:
                continue

            segments = raw.split(SEP) if SEP in raw else [raw]
            for idx in range(min(len(segments), len(relevant_docs))):
                seg = segments[idx]
                meta = relevant_docs[idx]
                cleaned = self._clean_content(seg)
                if cleaned:
                    blocks.append({
                        "text": cleaned,
                        "source_title": meta.get("title"),
                        "source_url": meta.get("url"),
                    })
        return blocks

    def fact_check_statements(
        self, 
        documents_data: List[Dict[str, str]], 
        knowledge_base: str,
        batch_delay: int = 5,
        enable_tavily_retry: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fact check statements using RAG-enhanced Gemini analysis
        
        Args:
            documents_data: List of documents with content
            knowledge_base: Text knowledge base
            batch_delay: Seconds to wait between batches (default: 5)
            enable_tavily_retry: If True, retry flagged/incorrect with Tavily
        """
        logger.info(f"Fact checking {len(documents_data)} statements")
        logger.info(f"Batch delay: {batch_delay} seconds between batches")
        logger.info(f"Tavily retry: {'enabled' if enable_tavily_retry else 'disabled'}")

        loaded = self.rag_system.load_index()

        if not loaded:
            kb_blocks = self._build_kb_blocks_from_documents(documents_data)
            if kb_blocks:
                self.rag_system.build_index_from_blocks(kb_blocks)
                self.rag_system.save_index()
            else:
                self.rag_system.build_index(knowledge_base)
                self.rag_system.save_index()

        results = []
        total_batches = (len(documents_data) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(documents_data))
            batch_data = documents_data[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (statements {start_idx + 1}-{end_idx})")

            batch_results = self._fact_check_batch_with_retry(batch_data, max_retries=5)
            results.extend(batch_results)

            batch_verdicts = [r.get('Final Verdict', 'Flagged for Review') for r in batch_results]
            correct_count = batch_verdicts.count('Correct')
            incorrect_count = batch_verdicts.count('Incorrect')
            review_count = batch_verdicts.count('Flagged for Review')
            logger.info(f"Batch {batch_idx + 1} results: {correct_count} Correct, {incorrect_count} Incorrect, {review_count} Flagged for Review")
            
            if batch_idx < total_batches - 1:
                logger.info(f"⏸️  Waiting {batch_delay} seconds before next batch...")
                time.sleep(batch_delay)

        # Check if we should retry with Tavily
        if enable_tavily_retry and self.tavily_client:
            failed_results = [
                r for r in results 
                if r.get('Final Verdict') in ['Incorrect', 'Flagged for Review']
            ]
            
            if failed_results:
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 TAVILY RETRY MODE")
                logger.info(f"{'='*80}")
                logger.info(f"Found {len(failed_results)} claims that were flagged or incorrect")
                logger.info(f"Retrying with Tavily for better sources...")
                
                # Get indices of failed results
                failed_indices = [
                    i for i, r in enumerate(results) 
                    if r.get('Final Verdict') in ['Incorrect', 'Flagged for Review']
                ]
                
                # Retry with Tavily
                retry_results = self.retry_with_tavily(failed_results, documents_data)
                
                # Update results with retry outcomes
                for idx, retry_result in zip(failed_indices, retry_results):
                    old_verdict = results[idx]['Final Verdict']
                    new_verdict = retry_result['Final Verdict']
                    
                    results[idx] = retry_result
                    results[idx]['Original Verdict'] = old_verdict
                    results[idx]['Verdict Changed'] = (old_verdict != new_verdict)
                    
                    if old_verdict != new_verdict:
                        logger.info(f"  ✓ Claim #{idx+1}: {old_verdict} → {new_verdict}")
                    else:
                        logger.info(f"  - Claim #{idx+1}: Still {new_verdict}")
                
                logger.info(f"{'='*80}\n")

        logger.info(f"✅ Fact checking completed for {len(results)} statements")
        return results
    
    def retry_with_tavily(
        self, 
        failed_results: List[Dict[str, Any]],
        original_documents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Retry fact-checking for flagged or incorrect claims using Tavily
        
        Args:
            failed_results: Results that were flagged or marked incorrect
            original_documents: Original document data for reference
            
        Returns:
            Updated results with Tavily-based fact checking
        """
        if not self.tavily_client:
            logger.warning("⚠️  Tavily not available for retry. Skipping retry.")
            return failed_results
        
        logger.info(f"🔄 Retrying {len(failed_results)} claims with Tavily fallback...")
        
        # Extract statements that need retry
        statements_to_retry = [
            {
                "Original Statement": result["Statement"],
                "Assertion": result["Assertion"]
            }
            for result in failed_results
        ]
        
        # Search again using Tavily
        tavily_documents = self.find_relevant_documents(
            statements_to_retry,
            use_tavily=True
        )
        
        # Build new knowledge base from Tavily results
        tavily_kb = self.build_knowledge_base(tavily_documents)
        
        # Rebuild RAG index with Tavily data
        logger.info("Building RAG index from Tavily documents...")
        kb_blocks = self._build_kb_blocks_from_documents(tavily_documents)
        if kb_blocks:
            self.rag_system.build_index_from_blocks(kb_blocks)
        else:
            self.rag_system.build_index(tavily_kb)
        
        # Re-run fact checking
        logger.info("Re-running fact checking with Tavily data...")
        retry_results = self.fact_check_statements(
            tavily_documents,
            tavily_kb,
            batch_delay=self.config.batch_delay,
            enable_tavily_retry=False  # Don't retry again
        )
        
        # Mark these as retried
        for result in retry_results:
            result["Retried with Tavily"] = True
        
        return retry_results
    
    def _extract_retry_delay(self, error_str: str) -> float:
        """Extract retry delay from error message"""
        import re
        
        match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 5
        
        match = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+)s', error_str)
        if match:
            return float(match.group(1)) + 5
        
        return 60

    def _fact_check_batch_with_retry(self, batch_data: List[Dict], max_retries: int = 5) -> List[Dict[str, Any]]:
        """Fact check a batch with automatic retry on rate limit errors"""
        
        for attempt in range(max_retries):
            try:
                return self._fact_check_batch(batch_data)
                
            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    retry_delay = self._extract_retry_delay(error_str)
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️  Rate limit hit. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                        raise
                else:
                    raise
        
        return [
            {
                "Statement": item.get("Statement", ""),
                "Assertion": item.get("Assertion", ""),
                "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                "Relevant Docs": item.get("Relevant Docs", ""),
                "Final Verdict": "Flagged for Review",
                "Full Analysis": f"Error occurred during batch fact checking after {max_retries} retries"
            }
            for item in batch_data
        ]
    
    def _fact_check_batch(self, batch_data: List[Dict]) -> List[Dict[str, Any]]:
        """Fact check a batch of statements using RAG-enhanced analysis"""
        try:
            prompt = self._create_batch_fact_checking_prompt(batch_data)
            model_name = "gemini-2.0-flash"
            # model_name = "gemini-2.5-flash"
            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=64000,
                    temperature=0.1,
                )
            )
            
            return self._parse_batch_response(response.text, batch_data)
            
        except Exception as e:
            logger.error(f"Error in batch fact checking: {e}")
            import traceback
            traceback.print_exc()
            return [
                {
                    "Statement": item.get("Statement", ""),
                    "Assertion": item.get("Assertion", ""),
                    "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                    "Relevant Docs": item.get("Relevant Docs", ""),
                    "Final Verdict": "Flagged for Review",
                    "Full Analysis": f"Error occurred during batch fact checking: {str(e)}"
                }
                for item in batch_data
            ]
    
    def _create_batch_fact_checking_prompt(self, batch_data: List[Dict]) -> str:
        """Create a batch prompt for fact checking multiple statements"""
        blocks = []
        for i, statement_data in enumerate(batch_data, 1):
            statement = statement_data.get("Statement", "")
            assertion = statement_data.get("Assertion", "")
            tavily_summary = statement_data.get("Summary (Tavily Answer)", "")
            relevant_docs = statement_data.get("Relevant Docs", "")

            query = f"{statement} {assertion}".strip()
            rel = self.rag_system.retrieve_relevant_chunks_with_meta(query, top_k=3)
            if not rel:
                rel_texts = self.rag_system.retrieve_relevant_chunks(query, top_k=2)
                rel = [{"text": t, "source_title": None, "source_url": None} for t in rel_texts] or [{"text": "No relevant information found in knowledge base.", "source_title": None, "source_url": None}]

            kb_lines = []
            for j, rc in enumerate(rel, 1):
                src = rc.get("source_url") or ""
                ttl = rc.get("source_title") or ""
                head = f"Chunk {j}"
                if ttl or src:
                    head += f" — {ttl} {f'({src})' if src else ''}"
                kb_lines.append(f"{head}:\n{rc.get('text','')}\n")

            kb_text = '\n'.join(kb_lines)
            block = f"""
<ITEM index="{i}">
STATEMENT: {statement}
ASSERTION: {assertion}
TAVILY_SUMMARY: {tavily_summary}
RELEVANT_DOCS_LIST:
{relevant_docs}

KNOWLEDGE_BASE_EXCERPTS:
{kb_text}
</ITEM>
"""
            blocks.append(block)

        
        return FACT_CHECKING_PROMPT + "\n\n" + "\n".join(blocks)
    
    def _parse_batch_response(self, response_text: str, batch_data: List[Dict]) -> List[Dict[str, Any]]:
        """Parse strict JSON batch response"""
        text = response_text.strip()
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            text = m.group(1).strip()

        if not (text.startswith('[') and text.endswith(']')):
            m2 = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
            if m2:
                text = m2.group(0)

        try:
            arr = json.loads(text)
            if not isinstance(arr, list):
                raise ValueError("Top-level JSON is not a list")
        except Exception as e:
            logger.error(f"Failed to parse JSON batch response: {e}")
            logger.error(text[:1000])
            out = []
            for item in batch_data:
                out.append({
                    "Statement": item.get("Statement", ""),
                    "Assertion": item.get("Assertion", ""),
                    "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                    "Relevant Docs": item.get("Relevant Docs", ""),
                    "Final Verdict": "Flagged for Review",
                    "Full Analysis": "Failed to parse JSON batch response."
                })
            return out

        idx_map = {int(obj.get("index")): obj for obj in arr if isinstance(obj, dict) and "index" in obj}
        results = []
        for i, item in enumerate(batch_data, 1):
            entry = idx_map.get(i, {})
            verdict = entry.get("final_verdict", "Flagged for Review")
            reasoning = entry.get("reasoning", "")
            citations = entry.get("citations", [])

            results.append({
                "Statement": item.get("Statement", ""),
                "Assertion": item.get("Assertion", ""),
                "Summary (Tavily Answer)": item.get("Summary (Tavily Answer)", ""),
                "Relevant Docs": item.get("Relevant Docs", ""),
                "Final Verdict": verdict if verdict in ("Correct", "Incorrect", "Flagged for Review") else "Flagged for Review",
                "Full Analysis": reasoning.strip(),
                "Citations": citations if isinstance(citations, list) else []
            })
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "output"):
        """Save results to CSV and JSON files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        output_df = pd.DataFrame(results)
        
        csv_path = Path(output_dir) / "fact_checking_results.csv"
        output_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to CSV: {csv_path}")
        
        json_path = Path(output_dir) / "detailed_fact_checking_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to JSON: {json_path}")
        
        verdict_counts = output_df['Final Verdict'].value_counts()
        logger.info("\nFinal Summary of Results:")
        for verdict, count in verdict_counts.items():
            logger.info(f"{verdict}: {count}")
        
        return csv_path, json_path
    
    def load_documents_from_file(self, filepath: str = "relevant_documents.json") -> List[Dict[str, str]]:
        """Load previously saved documents data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            logger.info(f"✅ Loaded {len(documents_data)} documents from {filepath}")
            return documents_data
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON from {filepath}: {e}")
            raise

    def load_knowledge_base_from_file(self, filepath: str = "knowledge_base.txt") -> str:
        """Load previously saved knowledge base from text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                knowledge_base = f.read()
            
            logger.info(f"✅ Loaded knowledge base from {filepath} ({len(knowledge_base)} characters)")
            return knowledge_base
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {filepath}")
            raise

    def load_assertions_from_file(self, filepath: str = "assertions_list.json") -> List[Dict[str, str]]:
        """Load previously saved assertions from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                assertions = json.load(f)
            
            logger.info(f"✅ Loaded {len(assertions)} assertions from {filepath}")
            return assertions
            
        except FileNotFoundError:
            logger.error(f"❌ File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON from {filepath}: {e}")
            raise

    def continue_from_saved_data(
        self,
        documents_file: str = "relevant_documents.json",
        kb_file: str = "knowledge_base.txt",
        batch_delay: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Continue fact-checking using previously saved documents and knowledge base
        """
        logger.info("📂 Loading saved data to continue fact-checking...")
        
        documents_data = self.load_documents_from_file(documents_file)
        knowledge_base = self.load_knowledge_base_from_file(kb_file)
        
        results = self.fact_check_statements(
            documents_data, 
            knowledge_base,
            batch_delay=batch_delay
        )
        
        return results
