"""Tests for benchmark contamination audit."""

from evaltrust.audit.contamination import (
    ContaminationResult,
    _normalize_text,
    _find_exact_matches,
    _find_near_matches,
    run_contamination_audit,
)


def test_normalize_text():
    assert _normalize_text("Hello, World!") == "hello world"
    assert _normalize_text("  Extra   spaces  \n here ") == "extra spaces here"
    assert _normalize_text("NoPunctuation") == "nopunctuation"
    assert _normalize_text("123!@#") == "123"
    assert _normalize_text("") == ""


def test_find_exact_matches():
    benchmark = ["This is a test.", "Another test!", "Unique string."]
    reference = ["this is a test", "completely different", "another   test"]
    
    matches = _find_exact_matches(benchmark, reference)
    assert matches == {0, 1}


def test_find_near_matches():
    benchmark = ["This is a test.", "Another test!", "Unique string."]
    # "anothr test" is a near match (typo) for "Another test!"
    reference = ["completely different", "anothr test", "this is a test"]
    
    # Assuming exact match found index 0 ("This is a test.")
    exact_matches = {0}
    near_matches = _find_near_matches(benchmark, reference, exact_matches)
    
    # It should skip index 0, and flag index 1 as a near match
    assert near_matches == {1}


def test_run_contamination_audit_partial():
    benchmark = [
        "What is the capital of France?", 
        "Who wrote Hamlet?", 
        "What is 2+2?", 
        "Where is the Eiffel Tower?"
    ]
    reference = [
        "what is the capital of france", # Exact match for index 0
        "random text about space",       # No match for index 1
        "what is 2+2",                   # Exact match for index 2
        "where is the eiffel towr"       # Near match (typo) for index 3
    ]
    
    result = run_contamination_audit(benchmark, reference)
    
    assert result.exact_matches == 2
    assert result.near_matches == 1
    assert result.total_items == 4
    assert result.contamination_fraction == 3 / 4


def test_run_contamination_audit_empty():
    result = run_contamination_audit([], [])
    
    assert result.exact_matches == 0
    assert result.total_items == 0
    assert result.contamination_fraction == 0.0


def test_run_contamination_audit_no_overlap():
    benchmark = ["A", "B", "C"]
    reference = ["X", "Y", "Z"]
    
    result = run_contamination_audit(benchmark, reference)
    
    assert result.exact_matches == 0
    assert result.contamination_fraction == 0.0
