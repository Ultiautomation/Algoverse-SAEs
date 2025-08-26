# def extract_refusal_phrases(response, refusal_terms):
#     if not isinstance(response, str):
#         return ""
#     response_lower = response.lower()
#     matched_terms = [term for term in refusal_terms if term in response_lower]
#     return "; ".join(matched_terms) if matched_terms else ""

def extract_refusal_phrases(response, refusal_terms):
    """
    Identifies refusal-related phrases in a model's response.

    Args:
        response (str): The model-generated text response.
        refusal_terms (list): A list of known refusal phrases (case-insensitive).

    Returns:
        str: A semicolon-separated string of matched refusal phrases, or an empty string.
    """
    if not isinstance(response, str):
        return ""

    # Normalize the response
    response_lower = response.strip().lower()

    # Normalize the refusal terms (lowercased, stripped)
    normalized_terms = [term.strip().lower() for term in refusal_terms]

    # Match terms that appear in the response
    matched_terms = [term for term in normalized_terms if term in response_lower]

    return "; ".join(matched_terms) if matched_terms else ""