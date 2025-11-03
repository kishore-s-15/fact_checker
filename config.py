
import os
import logging
from dataclasses import dataclass
from typing import List, Optional

# Suppress lxml warning
import warnings
warnings.filterwarnings('ignore', message='.*lxml.*does not provide the extra.*')

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


@dataclass
class Config:
    """Configuration class for the fact checking system"""
    google_api_key: str
    pubmed_email: str
    tavily_api_key: Optional[str]  # Optional for fallback
    search_domains: List[str]
    batch_size: int
    batch_delay: int
    chunk_size: int
    overlap: int = 50


def load_config() -> Config:
    """Load configuration from environment variables"""
    google_api_key = os.getenv('GEMINI_API_KEY')
    pubmed_email = os.getenv('PUBMED_EMAIL')
    tavily_api_key = os.getenv('TAVILY_API_KEY')  # Optional
    
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    if not pubmed_email:
        raise ValueError("PUBMED_EMAIL environment variable is required")
    
    if not tavily_api_key:
        logger.warning("⚠️  TAVILY_API_KEY not set. Tavily fallback will be disabled.")
    
    search_domains = os.getenv('SEARCH_DOMAINS', 'ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov').split(',')
    search_domains = [d.strip() for d in search_domains if d.strip()]
    batch_size = int(os.getenv('BATCH_SIZE', '3'))
    batch_delay = int(os.getenv('BATCH_DELAY', '10'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '500'))
    
    return Config(
        google_api_key=google_api_key,
        pubmed_email=pubmed_email,
        tavily_api_key=tavily_api_key,
        search_domains=search_domains,
        batch_size=batch_size,
        batch_delay=batch_delay,
        chunk_size=chunk_size
    )