#!/usr/bin/env python3
"""Demo script showing enhanced definition processing with web search."""

import json
import os
import sys
import tempfile
from typing import Dict, Any

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from make_small_dictionary import (
    _search_web_for_context, 
    _evaluate_definition_quality,
    LMStudioConfig,
    strip_llm_artifacts
)

def demo_enhanced_processing():
    """Demonstrate the enhanced processing workflow."""
    print("Enhanced Definition Processing Demo")
    print("=" * 50)
    
    # Example definitions that might be unhelpful
    test_cases = [
        {
            "term": "acetylsalicylic acid",
            "pos": "noun",
            "original_definition": "C9H8O4, molecular formula with molecular weight of 180.16 g/mol",
            "expected_improvement": "Chemical formula should be replaced with actual meaning"
        },
        {
            "term": "sodium chloride",
            "pos": "noun", 
            "original_definition": "NaCl, ionic compound with formula weight 58.44",
            "expected_improvement": "Should explain it's table salt"
        },
        {
            "term": "quadratic formula",
            "pos": "noun",
            "original_definition": "x = (-b ± √(b²-4ac)) / 2a",
            "expected_improvement": "Should explain what it's used for"
        }
    ]
    
    # Mock LM Studio config
    cfg = LMStudioConfig(
        url="http://localhost:1234/v1/chat/completions",
        model="demo-model"
    )
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Processing: {case['term']} ({case['pos']})")
        print(f"Original definition: {case['original_definition']}")
        
        # Step 1: Evaluate if definition is helpful
        is_helpful = _evaluate_definition_quality(case['original_definition'], cfg)
        print(f"Definition evaluation: {'Helpful' if is_helpful else 'Unhelpful'}")
        
        # Step 2: If unhelpful, search for web context
        web_context = None
        if not is_helpful:
            print("Searching for additional context...")
            web_context = _search_web_for_context(case['term'], case['pos'])
            if web_context:
                print(f"Found context: {web_context[:100]}..." if len(web_context) > 100 else f"Found context: {web_context}")
            else:
                print("No additional context found")
        
        # Step 3: Would normally pass to LLM with enhanced prompt
        print(f"Expected improvement: {case['expected_improvement']}")
        
        if web_context:
            print("✓ Would be enhanced with web context")
        else:
            print("✓ Would be processed with technical notation handling")
        
        print("-" * 40)
    
    print("\nWorkflow Summary:")
    print("1. Evaluate definition quality with LLM")
    print("2. If unhelpful, search web for additional context")
    print("3. Use enhanced prompt with context to create better definition")
    print("4. Store both enhanced definition and source context in database")
    print("5. Track enhancement metadata for future processing")

def demo_database_enhancements():
    """Show the database schema enhancements."""
    print("\n" + "=" * 50)
    print("Database Schema Enhancements")
    print("=" * 50)
    
    print("New column added to definitions table:")
    print("- enhanced_source: TEXT - Stores web search context used for enhancement")
    print()
    print("Example data:")
    example_data = {
        "word": "aspirin",
        "idx": 0,
        "pos": "noun",
        "source_first_sentence": "C9H8O4, acetylsalicylic acid compound",
        "current_line": "A pain-relieving medication derived from salicylic acid",
        "enhanced_source": "Aspirin is a medication used to treat pain, fever, or inflammation | Aspirin, also known as acetylsalicylic acid"
    }
    
    for key, value in example_data.items():
        print(f"  {key}: {value}")
    
    print("\nThis allows tracking:")
    print("- Which definitions were enhanced with web search")
    print("- What context was used for enhancement")
    print("- Ability to re-process with different search terms")
    print("- Audit trail for quality improvements")

if __name__ == "__main__":
    demo_enhanced_processing()
    demo_database_enhancements()
    print("\n" + "=" * 50)
    print("Demo complete! The enhanced processing system is ready.")
    print("To use with actual LLM processing:")
    print("1. Start LM Studio with your preferred model")
    print("2. Run the main script in 'enhance' mode")
    print("3. Enhanced definitions will be created with web context when helpful")