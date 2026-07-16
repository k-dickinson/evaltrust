"""Benchmark contamination and overlap detection."""
import re
import difflib
from dataclasses import dataclass



@dataclass
class ContaminationResult:
    exact_matches: int
    near_matches: int
    total_items: int
    contamination_fraction: float


def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation/excess whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _find_exact_matches(benchmark: list[str], reference: list[str]) -> set[int]:

    """Find exact matches between benchmark and reference sets."""

    normalized_reference = {
        _normalize_text(text)
        for text in reference
    }
    matches = set()

    for i, text in enumerate(benchmark):
        if _normalize_text(text) in normalized_reference:
            matches.add(i)
    return matches

def _find_near_matches(benchmark: list[str], reference: list[str], exact_matches: set[int], threshold: float = 0.85) -> set[int]:
    # TODO: Implement a scalable n-gram / MinHash approach for large reference corpora.
    # Currently uses a quadratic O(N*M) SequenceMatcher search.
    normalized_ref_list = [_normalize_text(text) for text in reference]
    near_matches = set()
    
    for i, text in enumerate(benchmark):
        if i in exact_matches:
            continue
        normalized = _normalize_text(text)
        normalized_len = len(normalized)
        
        for ref_text in normalized_ref_list:
            ref_len = len(ref_text)
            
            if (normalized_len + ref_len) > 0:
                max_possible_ratio = 2.0 * min(normalized_len, ref_len) / (normalized_len + ref_len)
                if max_possible_ratio < threshold:
                    continue
                    
            similarity = difflib.SequenceMatcher(None, normalized, ref_text).ratio()
            if similarity >= threshold:
                near_matches.add(i)
                break
    return near_matches


def run_contamination_audit(
    benchmark: list[str],
    reference: list[str],
) -> ContaminationResult:
    """Check for contamination between a benchmark and a reference set."""

    exact_matches = _find_exact_matches(benchmark, reference)
    near_matches_indices = _find_near_matches(benchmark, reference, exact_matches)

    total_items = len(benchmark)

    contamination_fraction = (
        (len(exact_matches) + len(near_matches_indices)) / total_items
        if total_items > 0
        else 0.0
    )
    return ContaminationResult(
        exact_matches = len(exact_matches),
        near_matches = len(near_matches_indices),
        total_items = total_items,
        contamination_fraction=contamination_fraction
    )

