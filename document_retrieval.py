import os
import re
import time
import requests
import trafilatura
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from Bio import Entrez

from logger import logger

# Suppress lxml warning
import warnings
warnings.filterwarnings('ignore', message='.*lxml.*does not provide the extra.*')


class ComprehensiveDocumentRetrieval:
    """
    Comprehensive free document retrieval using ALL available free sources:
    - PubMed, PMC, NCBI Bookshelf
    - Europe PMC, Semantic Scholar
    - arXiv, bioRxiv, medRxiv
    - CORE, CrossRef
    - Web scraping fallback
    """
    
    def __init__(self, email: str):
        self.email = email
        Entrez.email = email
        Entrez.tool = "ScientificFactChecker"
        api_key = os.getenv('NCBI_API_KEY')
        if api_key:
            Entrez.api_key = api_key
            logger.info("Using NCBI API key for enhanced rate limits")
        
        # Track which sources have been tried
        self.sources_tried = []
    
    def search_and_extract(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Comprehensive multi-source search with intelligent fallback
        Tries sources in order of reliability and relevance
        """
        all_results = []
        self.sources_tried = []
        
        # Phase 1: Primary biomedical sources
        logger.info(f"🔍 Phase 1: Searching primary biomedical sources...")
        
        # 1. PubMed (most authoritative)
        pubmed_results = self._search_pubmed_comprehensive(query, max_results=2)
        all_results.extend(pubmed_results)
        self.sources_tried.append(f"PubMed ({len(pubmed_results)})")
        
        # 2. NCBI Bookshelf (textbook content)
        if len(all_results) < max_results:
            bookshelf_results = self._search_ncbi_bookshelf(query, max_results=1)
            all_results.extend(bookshelf_results)
            self.sources_tried.append(f"NCBI Bookshelf ({len(bookshelf_results)})")
        
        # Phase 2: Alternative biomedical sources
        if len(all_results) < max_results:
            logger.info(f"🔍 Phase 2: Searching alternative biomedical sources...")
            
            # 3. Europe PMC
            europepmc_results = self._search_europepmc_direct(query, max_results=1)
            all_results.extend(europepmc_results)
            self.sources_tried.append(f"Europe PMC ({len(europepmc_results)})")
        
        # Phase 3: Preprint servers
        if len(all_results) < max_results:
            logger.info(f"🔍 Phase 3: Searching preprint servers...")
            
            # 4. bioRxiv/medRxiv
            biorxiv_results = self._search_biorxiv(query, max_results=1)
            all_results.extend(biorxiv_results)
            self.sources_tried.append(f"bioRxiv/medRxiv ({len(biorxiv_results)})")
        
        # Phase 4: General academic sources
        if len(all_results) < max_results:
            logger.info(f"🔍 Phase 4: Searching general academic sources...")
            
            # 5. Semantic Scholar
            semantic_results = self._search_semantic_scholar(query, max_results=2)
            all_results.extend(semantic_results)
            self.sources_tried.append(f"Semantic Scholar ({len(semantic_results)})")
        
        # Phase 5: Open access aggregators
        if len(all_results) < max_results:
            logger.info(f"🔍 Phase 5: Searching open access aggregators...")
            
            # 6. CORE
            core_results = self._search_core(query, max_results=1)
            all_results.extend(core_results)
            self.sources_tried.append(f"CORE ({len(core_results)})")
        
        # Phase 6: arXiv (if still need more)
        if len(all_results) < max_results:
            logger.info(f"🔍 Phase 6: Searching arXiv preprints...")
            
            arxiv_results = self._search_arxiv(query, max_results=1)
            all_results.extend(arxiv_results)
            self.sources_tried.append(f"arXiv ({len(arxiv_results)})")
        
        # Phase 7: Web scraping fallback for specific known domains
        if len(all_results) < max_results:
            logger.info(f"🔍 Phase 7: Web search fallback...")
            
            web_results = self._search_web_fallback(query, max_results=1)
            all_results.extend(web_results)
            self.sources_tried.append(f"Web Search ({len(web_results)})")
        
        # Remove duplicates based on title similarity
        all_results = self._deduplicate_results(all_results)
        
        if not all_results:
            logger.warning(f"❌ No results found across ALL sources for: {query}")
            logger.info(f"Sources tried: {', '.join(self.sources_tried)}")
            return {
                'results': [], 
                'answer': f'No relevant information found after searching: {", ".join(self.sources_tried)}'
            }
        
        # Generate comprehensive content
        answer = self._generate_comprehensive_content(all_results)
        
        logger.info(f"✅ Total: {len(all_results)} unique results from {len(self.sources_tried)} sources")
        logger.info(f"Sources used: {', '.join(self.sources_tried)}")
        
        return {
            'results': all_results[:max_results],  # Limit to max_results
            'answer': answer
        }
    
    def _search_pubmed_comprehensive(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced PubMed search with better query handling"""
        results = []
        
        try:
            # Try exact query first
            pmids = self._search_pubmed(query, max_results)
            
            # If no results, try broader query
            if not pmids and len(query.split()) > 3:
                # Remove quotes and try again
                broader_query = query.replace('"', '')
                logger.info(f"Trying broader PubMed query: {broader_query[:50]}...")
                pmids = self._search_pubmed(broader_query, max_results)
            
            if not pmids:
                return results
            
            papers_metadata = self._fetch_pubmed_metadata(pmids)
            
            for paper in papers_metadata:
                pmid = paper['pmid']
                
                full_text = self._get_pmc_fulltext(pmid)
                if not full_text:
                    full_text = self._get_europepmc_fulltext(pmid)
                if not full_text and paper.get('doi'):
                    full_text = self._get_fulltext_from_doi(paper['doi'])
                
                content = paper['abstract']
                if full_text:
                    content = f"{paper['abstract']}\n\n--- FULL TEXT ---\n\n{full_text}"
                
                results.append({
                    'title': paper['title'],
                    'url': paper['url'],
                    'content': content,
                    'full_text': full_text,
                    'abstract': paper['abstract'],
                    'source': 'PubMed',
                    'doi': paper.get('doi', '')
                })
                
                time.sleep(0.4)
        
        except Exception as e:
            logger.error(f"PubMed comprehensive search error: {e}")
        
        return results
    
    def _search_semantic_scholar(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search Semantic Scholar - great for finding related papers"""
        results = []
        
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,abstract,url,year,citationCount,authors,openAccessPdf,tldr'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.debug(f"Semantic Scholar returned status {response.status_code}")
                return results
            
            data = response.json()
            papers = data.get('data', [])
            
            if not papers:
                logger.debug("No Semantic Scholar results")
                return results
            
            logger.info(f"Found {len(papers)} Semantic Scholar results")
            
            for paper in papers:
                title = paper.get('title', 'Unknown Title')
                abstract = paper.get('abstract', '')
                paper_url = paper.get('url', '')
                tldr = paper.get('tldr', {})
                
                # Build content
                content_parts = []
                
                if abstract:
                    content_parts.append(f"ABSTRACT:\n{abstract}")
                
                if tldr and tldr.get('text'):
                    content_parts.append(f"\nTL;DR (AI Summary):\n{tldr['text']}")
                
                content = '\n\n'.join(content_parts)
                
                if content:
                    results.append({
                        'title': title,
                        'url': paper_url,
                        'content': content,
                        'full_text': None,
                        'abstract': abstract,
                        'source': 'Semantic Scholar',
                        'doi': ''
                    })
                    logger.info(f"✓ Semantic Scholar: {title[:60]}...")
                
                time.sleep(0.3)
        
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
        
        return results
    
    def _search_biorxiv(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search bioRxiv and medRxiv preprint servers"""
        results = []
        
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Last year
            
            url = f"https://api.biorxiv.org/details/biorxiv/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}/0"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.debug(f"bioRxiv returned status {response.status_code}")
                return results
            
            data = response.json()
            collection = data.get('collection', [])
            
            if not collection:
                logger.debug("No bioRxiv results in date range")
                return results
            
            # Filter by query relevance
            query_terms = set(query.lower().split())
            relevant_papers = []
            
            for paper in collection:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                combined = f"{title} {abstract}"
                
                matches = sum(1 for term in query_terms if term in combined)
                if matches >= max(2, len(query_terms) * 0.5):
                    relevant_papers.append((matches, paper))
            
            relevant_papers.sort(reverse=True, key=lambda x: x[0])
            
            for _, paper in relevant_papers[:max_results]:
                title = paper.get('title', 'Unknown Title')
                abstract = paper.get('abstract', '')
                doi = paper.get('doi', '')
                
                content = f"ABSTRACT:\n{abstract}"
                paper_url = f"https://www.biorxiv.org/content/{doi}v1" if doi else ""
                
                results.append({
                    'title': title,
                    'url': paper_url,
                    'content': content,
                    'full_text': None,
                    'abstract': abstract,
                    'source': 'bioRxiv',
                    'doi': doi
                })
                logger.info(f"✓ bioRxiv: {title[:60]}...")
        
        except Exception as e:
            logger.error(f"bioRxiv search error: {e}")
        
        return results
    
    def _search_arxiv(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search arXiv preprints (focus on q-bio category)"""
        results = []
        
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.debug(f"arXiv returned status {response.status_code}")
                return results
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                logger.debug("No arXiv results")
                return results
            
            logger.info(f"Found {len(entries)} arXiv results")
            
            for entry in entries:
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                id_elem = entry.find('atom:id', ns)
                
                title = title_elem.text.strip() if title_elem is not None else 'Unknown Title'
                abstract = summary_elem.text.strip() if summary_elem is not None else ''
                paper_url = id_elem.text.strip() if id_elem is not None else ''
                
                content = f"ABSTRACT:\n{abstract}"
                
                results.append({
                    'title': title,
                    'url': paper_url,
                    'content': content,
                    'full_text': None,
                    'abstract': abstract,
                    'source': 'arXiv',
                    'doi': ''
                })
                logger.info(f"✓ arXiv: {title[:60]}...")
        
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        
        return results
    
    def _search_core(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search CORE (world's largest collection of open access papers)"""
        results = []
        
        try:
            api_key = os.getenv('CORE_API_KEY')
            if not api_key:
                logger.debug("CORE_API_KEY not set, skipping CORE search")
                return results
            
            url = "https://api.core.ac.uk/v3/search/works"
            headers = {'Authorization': f'Bearer {api_key}'}
            params = {'q': query, 'limit': max_results}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.debug(f"CORE returned status {response.status_code}")
                return results
            
            data = response.json()
            papers = data.get('results', [])
            
            if not papers:
                logger.debug("No CORE results")
                return results
            
            logger.info(f"Found {len(papers)} CORE results")
            
            for paper in papers:
                title = paper.get('title', 'Unknown Title')
                abstract = paper.get('abstract', '')
                paper_url = paper.get('downloadUrl', '') or paper.get('sourceFulltextUrls', [''])[0]
                
                full_text = None
                if paper.get('fullText'):
                    full_text = paper['fullText']
                
                content = abstract
                if full_text:
                    content = f"{abstract}\n\n--- FULL TEXT ---\n\n{full_text}"
                
                if content:
                    results.append({
                        'title': title,
                        'url': paper_url,
                        'content': content,
                        'full_text': full_text,
                        'abstract': abstract,
                        'source': 'CORE',
                        'doi': paper.get('doi', '')
                    })
                    logger.info(f"✓ CORE: {title[:60]}...")
        
        except Exception as e:
            logger.error(f"CORE search error: {e}")
        
        return results
    
    def _search_web_fallback(self, query: str, max_results: int = 1) -> List[Dict]:
        """Web search fallback using DuckDuckGo (no API key required)"""
        results = []
        
        try:
            from urllib.parse import quote_plus
            from bs4 import BeautifulSoup
            
            site_query = f"{query} (site:nih.gov OR site:cancer.gov OR site:who.int OR site:nature.com OR site:science.org)"
            
            url = f"https://html.duckduckgo.com/html/"
            params = {'q': site_query}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.post(url, data=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.debug(f"DuckDuckGo returned status {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.content, 'html.parser')
            search_results = soup.find_all('div', class_='result')[:max_results]
            
            for result in search_results:
                link = result.find('a', class_='result__a')
                if not link:
                    continue
                
                title = link.get_text(strip=True)
                url_result = link.get('href', '')
                
                snippet = result.find('a', class_='result__snippet')
                snippet_text = snippet.get_text(strip=True) if snippet else ''
                
                if url_result:
                    full_content = self._extract_from_url(url_result)
                    
                    if full_content:
                        results.append({
                            'title': title,
                            'url': url_result,
                            'content': full_content,
                            'full_text': full_content,
                            'abstract': snippet_text,
                            'source': 'Web Search',
                            'doi': ''
                        })
                        logger.info(f"✓ Web: {title[:60]}...")
                        break
                
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Web fallback search error: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on title similarity"""
        if len(results) <= 1:
            return results
        
        unique_results = []
        seen_titles = set()
        
        for result in results:
            title = result.get('title', '').lower().strip()
            title_normalized = re.sub(r'[^\w\s]', '', title)
            title_normalized = ' '.join(title_normalized.split())
            
            if title_normalized and title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_results.append(result)
        
        removed = len(results) - len(unique_results)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate results")
        
        return unique_results
    
    def _search_pubmed(self, query: str, max_results: int) -> List[str]:
        """Search PubMed and return PMIDs"""
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            pmids = record.get("IdList", [])
            if pmids:
                logger.info(f"Found {len(pmids)} PubMed results")
            return pmids
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def _fetch_pubmed_metadata(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed metadata from PubMed"""
        if not pmids:
            return []
        
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmids,
                rettype="medline",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            papers = []
            for paper in records['PubmedArticle']:
                try:
                    medline = paper['MedlineCitation']
                    article = medline['Article']
                    pmid = str(medline['PMID'])
                    
                    doi = None
                    if 'ELocationID' in article:
                        for eloc in article['ELocationID']:
                            if eloc.attributes.get('EIdType') == 'doi':
                                doi = str(eloc)
                    
                    abstract = ''
                    if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                        abstract_parts = article['Abstract']['AbstractText']
                        if isinstance(abstract_parts, list):
                            abstract = ' '.join(str(part) for part in abstract_parts)
                        else:
                            abstract = str(abstract_parts)
                    
                    papers.append({
                        'pmid': pmid,
                        'title': str(article.get('ArticleTitle', '')),
                        'abstract': abstract,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'doi': doi
                    })
                except Exception as e:
                    logger.error(f"Error parsing paper: {e}")
                    continue
            
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching metadata: {e}")
            return []
    
    def _get_pmc_fulltext(self, pmid: str) -> Optional[str]:
        """Get full text from PubMed Central (PMC)"""
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            record = Entrez.read(handle)
            handle.close()
            
            pmcid = None
            if record and record[0]['LinkSetDb']:
                for link in record[0]['LinkSetDb']:
                    if link['DbTo'] == 'pmc':
                        pmcid = link['Link'][0]['Id']
                        break
            
            if not pmcid:
                return None
            
            handle = Entrez.efetch(
                db="pmc",
                id=pmcid,
                rettype="full",
                retmode="xml"
            )
            xml_content = handle.read()
            handle.close()
            
            full_text = self._parse_pmc_xml(xml_content)
            if full_text:
                logger.info(f"✓ PMC full text ({len(full_text)} chars)")
            return full_text
            
        except Exception as e:
            logger.debug(f"PMC error: {e}")
            return None
    
    def _parse_pmc_xml(self, xml_content: str) -> str:
        """Parse PMC XML to extract readable text"""
        try:
            root = ET.fromstring(xml_content)
            body = root.find('.//body')
            if body is None:
                return ""
            
            sections = []
            for sec in body.findall('.//sec'):
                title = sec.find('.//title')
                if title is not None and title.text:
                    sections.append(f"\n## {title.text}\n")
                
                for p in sec.findall('.//p'):
                    text = ''.join(p.itertext())
                    if text.strip():
                        sections.append(text.strip())
            
            return '\n\n'.join(sections)
        except Exception as e:
            logger.error(f"Error parsing PMC XML: {e}")
            return ""
    
    def _get_europepmc_fulltext(self, pmid: str) -> Optional[str]:
        """Get full text from Europe PMC"""
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                'query': f'EXT_ID:{pmid}',
                'resultType': 'core',
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('resultList', {}).get('result'):
                return None
            
            result = data['resultList']['result'][0]
            if result.get('isOpenAccess') != 'Y':
                return None
            
            pmcid = result.get('pmcid')
            if not pmcid:
                return None
            
            fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            fulltext_response = requests.get(fulltext_url, timeout=10)
            
            if fulltext_response.status_code == 200:
                full_text = self._parse_pmc_xml(fulltext_response.text)
                if full_text:
                    logger.info(f"✓ Europe PMC full text ({len(full_text)} chars)")
                return full_text
            
            return None
        except Exception as e:
            logger.debug(f"Europe PMC error: {e}")
            return None
    
    def _search_ncbi_bookshelf(self, query: str, max_results: int = 1) -> List[Dict]:
        """Search NCBI Bookshelf for medical textbook content"""
        results = []
        
        try:
            handle = Entrez.esearch(
                db="books",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            book_ids = record.get("IdList", [])
            
            if not book_ids:
                logger.debug(f"No NCBI Bookshelf results")
                return results
            
            logger.info(f"Found {len(book_ids)} NCBI Bookshelf results")
            
            for book_id in book_ids:
                try:
                    handle = Entrez.efetch(
                        db="books",
                        id=book_id,
                        rettype="full",
                        retmode="xml"
                    )
                    xml_content = handle.read()
                    handle.close()
                    
                    book_data = self._parse_bookshelf_xml(xml_content, book_id)
                    
                    if book_data:
                        results.append({
                            'title': book_data['title'],
                            'url': book_data['url'],
                            'content': book_data['content'],
                            'full_text': book_data['content'],
                            'abstract': book_data['content'][:500] + "...",
                            'source': 'NCBI Bookshelf',
                            'doi': ''
                        })
                        logger.info(f"✓ NCBI Bookshelf: {book_data['title'][:60]}... ({len(book_data['content'])} chars)")
                    
                    time.sleep(0.4)
                    
                except Exception as e:
                    logger.error(f"Error fetching book ID {book_id}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error searching NCBI Bookshelf: {e}")
        
        return results
    
    def _parse_bookshelf_xml(self, xml_content: str, book_id: str) -> Optional[Dict]:
        """Parse NCBI Bookshelf XML content"""
        try:
            root = ET.fromstring(xml_content)
            
            title_elem = root.find('.//book-title')
            if title_elem is None:
                title_elem = root.find('.//article-title')
            title = title_elem.text if title_elem is not None else f"NCBI Bookshelf Entry {book_id}"
            
            sections = []
            body = root.find('.//body')
            if body is not None:
                for sec in body.findall('.//sec'):
                    sec_title = sec.find('.//title')
                    if sec_title is not None and sec_title.text:
                        sections.append(f"\n## {sec_title.text}\n")
                    
                    for p in sec.findall('.//p'):
                        text = ''.join(p.itertext())
                        if text.strip():
                            sections.append(text.strip())
            
            if not sections:
                all_text = ' '.join(root.itertext())
                sections = [all_text]
            
            content = '\n\n'.join(sections)
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            
            return {
                'title': title,
                'url': f"https://www.ncbi.nlm.nih.gov/books/{book_id}/",
                'content': content.strip()
            }
            
        except Exception as e:
            logger.error(f"Error parsing bookshelf XML: {e}")
            return None
    
    def _search_europepmc_direct(self, query: str, max_results: int = 1) -> List[Dict]:
        """Direct search of Europe PMC"""
        results = []
        
        try:
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                'query': query,
                'resultType': 'core',
                'format': 'json',
                'pageSize': max_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return results
            
            data = response.json()
            articles = data.get('resultList', {}).get('result', [])
            
            if articles:
                logger.info(f"Found {len(articles)} Europe PMC results")
            
            for article in articles:
                pmid = article.get('pmid')
                if pmid:
                    continue
                
                title = article.get('title', 'Unknown Title')
                abstract = article.get('abstractText', '')
                pmcid = article.get('pmcid', '')
                
                full_text = None
                if pmcid and article.get('isOpenAccess') == 'Y':
                    full_text = self._get_europepmc_fulltext_by_pmcid(pmcid)
                
                content = abstract
                if full_text:
                    content = f"{abstract}\n\n--- FULL TEXT ---\n\n{full_text}"
                
                url_str = f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else article.get('doi', '')
                
                results.append({
                    'title': title,
                    'url': url_str,
                    'content': content,
                    'full_text': full_text,
                    'abstract': abstract,
                    'source': 'Europe PMC',
                    'doi': article.get('doi', '')
                })
                
                logger.info(f"✓ Europe PMC: {title[:60]}...")
                time.sleep(0.4)
        
        except Exception as e:
            logger.error(f"Error in direct Europe PMC search: {e}")
        
        return results
    
    def _get_europepmc_fulltext_by_pmcid(self, pmcid: str) -> Optional[str]:
        """Get full text from Europe PMC using PMCID"""
        try:
            fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            response = requests.get(fulltext_url, timeout=10)
            
            if response.status_code == 200:
                return self._parse_pmc_xml(response.text)
            
            return None
        except Exception as e:
            logger.debug(f"Error getting Europe PMC full text: {e}")
            return None
    
    def _get_fulltext_from_doi(self, doi: str) -> Optional[str]:
        """Try to get full text using DOI via Unpaywall API"""
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {'email': self.email}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('is_oa'):
                return None
            
            best_oa = data.get('best_oa_location')
            if not best_oa:
                return None
            
            landing_url = best_oa.get('url_for_landing_page')
            if landing_url:
                text = self._extract_from_url(landing_url)
                if text:
                    logger.info(f"✓ Full text from DOI ({len(text)} chars)")
                    return text
            
            return None
        except Exception as e:
            logger.debug(f"DOI error: {e}")
            return None
    
    def _extract_from_url(self, url: str) -> Optional[str]:
        """Extract text content from a URL using trafilatura"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            return text
        except Exception as e:
            logger.debug(f"Extraction error: {e}")
            return None
    
    def _generate_comprehensive_content(self, results: List[Dict]) -> str:
        """Return comprehensive content from all retrieved sources"""
        if not results:
            return "No relevant information found."
        
        combined_content = []
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown Title')
            url = result.get('url', 'N/A')
            source = result.get('source', 'Unknown')
            
            section = f"\n{'='*80}\n"
            section += f"SOURCE {i} ({source}): {title}\n"
            section += f"URL: {url}\n"
            section += f"{'='*80}\n\n"
            
            full_text = result.get('full_text', '')
            abstract = result.get('abstract', '')
            
            if full_text:
                section += "FULL TEXT:\n\n"
                section += full_text
            elif abstract:
                section += "ABSTRACT:\n\n"
                section += abstract
            else:
                section += "No content available for this source."
            
            combined_content.append(section)
        
        if not combined_content:
            return "No content available from any source"
        
        final_content = "\n\n".join(combined_content)
        
        return final_content