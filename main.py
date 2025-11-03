#!/usr/bin/env python3
"""
Scientific Fact Checking System
A comprehensive tool for extracting, validating, and fact-checking scientific statements
using Google Gemini AI and free PubMed/PMC APIs with Tavily fallback.
"""

import sys
import json
from dotenv import load_dotenv

from logger import logger
from config import load_config
from checker import ScientificFactChecker

# Suppress lxml warning
import warnings
warnings.filterwarnings('ignore', message='.*lxml.*does not provide the extra.*')

# Load environment variables
load_dotenv()


def main(mode: str = "full", batch_delay: int = 10):
    """
    Main function to run the scientific fact checking pipeline
    
    Args:
        mode: "full" = run complete pipeline, "resume" = continue from saved data
        batch_delay: Seconds to wait between batches (default: 10)
    """
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        fact_checker = ScientificFactChecker(config)
        logger.info("Scientific fact checker initialized")
        
        if mode == "resume":
            logger.info("🔄 RESUME MODE: Loading from saved files...")
            
            try:
                results = fact_checker.continue_from_saved_data(
                    documents_file="relevant_documents.json",
                    kb_file="knowledge_base.txt",
                    batch_delay=batch_delay
                )
                
                csv_path, json_path = fact_checker.save_results(results)
                
                logger.info("✅ Scientific fact checking (RESUME) completed successfully!")
                logger.info(f"Results available at: {csv_path} and {json_path}")
                
            except FileNotFoundError as e:
                logger.error("❌ Could not find saved files. Please run in 'full' mode first.")
                return
        
        else:
            logger.info("🚀 FULL MODE: Running complete pipeline...")
            
            # chapter_path = "./Chapters/Chapter 02_ Introduction to Cancer_ A Disease of Deregulation-Part2.md"
            chapter_path = "./Chapters/Chapter 11_ Cancer Metabolism_ The Warburg Effect and Beyond.md"
            
            assertions = fact_checker.extract_assertions(chapter_path)
            
            if not assertions:
                logger.error("No assertions extracted. Exiting.")
                return
            
            with open("assertions_list.json", 'w', encoding='utf-8') as f:
                json.dump(assertions, f, indent=2, ensure_ascii=False)
            logger.info("Assertions saved to assertions_list.json")
            
            # Find relevant documents (NO EXCEPTIONS - always continues)
            documents_data = fact_checker.find_relevant_documents(
                assertions,
                use_tavily=False  # Start with free APIs
            )
            
            with open("relevant_documents.json", 'w', encoding='utf-8') as f:
                json.dump(documents_data, f, indent=2, ensure_ascii=False)
            logger.info("Relevant documents saved to relevant_documents.json")
            
            knowledge_base = fact_checker.build_knowledge_base(documents_data)
            
            with open("knowledge_base.txt", 'w', encoding='utf-8') as f:
                f.write(knowledge_base)
            logger.info("Knowledge base saved to knowledge_base.txt")
            
            # Fact check statements (with automatic Tavily retry)
            results = fact_checker.fact_check_statements(
                documents_data, 
                knowledge_base,
                batch_delay=batch_delay,
                enable_tavily_retry=True  # Enable automatic retry
            )
            
            csv_path, json_path = fact_checker.save_results(results)
            
            logger.info("✅ Scientific fact checking pipeline completed successfully!")
            logger.info(f"Results available at: {csv_path} and {json_path}")
        
    except Exception as e:
        logger.error(f"❌ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":   
    mode = "full"
    batch_delay = 15
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if len(sys.argv) > 2:
        batch_delay = int(sys.argv[2])
    
    main(mode=mode, batch_delay=batch_delay)