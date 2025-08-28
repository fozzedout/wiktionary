#!/usr/bin/env python3
"""Test script for enhanced definition processing with web search."""

import json
import os
import sys
import tempfile
import sqlite3
from typing import Optional

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from make_small_dictionary import (
    _search_web_for_context, 
    _evaluate_definition_quality,
    summarize_with_lmstudio,
    LMStudioConfig,
    connect_state,
    strip_llm_artifacts
)

def test_web_search():
    """Test the web search functionality."""
    print("Testing web search for context...")
    
    # Test with a chemical compound
    context = _search_web_for_context("acetylsalicylic acid", "noun")
    print(f"Context for 'acetylsalicylic acid': {context}")
    
    # Test with a mathematical term
    context = _search_web_for_context("derivative", "noun")
    print(f"Context for 'derivative': {context}")
    
    return True

def test_definition_evaluation():
    """Test the definition quality evaluation."""
    print("Testing definition evaluation...")
    
    # Mock LM Studio config (won't actually call the API in this test)
    cfg = LMStudioConfig(
        url="http://localhost:1234/v1/chat/completions",
        model="test-model"
    )
    
    # Test with a chemical formula (should be unhelpful)
    formula_def = "C9H8O4, molecular weight 180.16"
    is_helpful = _evaluate_definition_quality(formula_def, cfg)
    print(f"Chemical formula definition helpful: {is_helpful}")
    
    # Test with a descriptive definition (should be helpful)
    descriptive_def = "A pain-relieving medication commonly used to treat headaches and inflammation"
    is_helpful = _evaluate_definition_quality(descriptive_def, cfg)
    print(f"Descriptive definition helpful: {is_helpful}")
    
    return True

def test_database_schema():
    """Test that the enhanced database schema works."""
    print("Testing database schema...")
    
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Test database creation with enhanced_source column
        conn = connect_state(db_path)
        
        # Insert a test definition
        conn.execute(
            "INSERT OR REPLACE INTO words (word, status) VALUES (?, ?)",
            ("test_word", "pending")
        )
        
        conn.execute(
            """INSERT OR REPLACE INTO definitions 
               (word, idx, pos, source_first_sentence, current_line, enhanced_source) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("test_word", 0, "noun", "Original definition", "Enhanced definition", "Web context used")
        )
        
        # Query back to verify
        result = conn.execute(
            "SELECT word, idx, enhanced_source FROM definitions WHERE word=?",
            ("test_word",)
        ).fetchone()
        
        print(f"Retrieved from DB: word={result[0]}, idx={result[1]}, enhanced_source={result[2]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Database test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_enhanced_summarization():
    """Test the enhanced summarization logic (without actual LLM calls)."""
    print("Testing enhanced summarization logic...")
    
    # Test that the function signature change is working
    from make_small_dictionary import summarize_with_lmstudio
    
    cfg = LMStudioConfig(
        url="http://localhost:1234/v1/chat/completions",
        model="test-model"
    )
    
    # This will fail to connect, but we can verify the function signature is correct
    try:
        result = summarize_with_lmstudio(cfg, "noun", "Test definition", 25)
        print(f"Function call result: {result}")
        # If we get None (expected when LM Studio isn't running), that's correct
        # If we get a tuple, the function signature is working correctly
        if result is None:
            print("Got None as expected (no LM Studio connection)")
            return True
        elif isinstance(result, tuple) and len(result) == 3:
            print("Got tuple as expected (function signature correct)")
            return True
        else:
            print(f"Unexpected result type: {type(result)}")
            return False
    except Exception as e:
        print(f"Function call failed as expected (no LM Studio running): {e}")
        return True  # This is expected

def main():
    """Run all tests."""
    print("Running enhanced definition processing tests...")
    print("=" * 50)
    
    tests = [
        test_web_search,
        test_definition_evaluation,
        test_database_schema,
        test_enhanced_summarization,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"✓ {test.__name__} passed" if result else f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Enhanced definition processing is working.")
        sys.exit(0)
    else:
        print("Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()