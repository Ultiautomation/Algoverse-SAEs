def extract_refusal_phrases(response, refusal_terms):
    if not isinstance(response, str):
        return ""
    response_lower = response.lower()
    matched_terms = [term for term in refusal_terms if term in response_lower]
    return "; ".join(matched_terms) if matched_terms else ""