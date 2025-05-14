import torch
from model_utils import classify_patterns_with_probs, generate_llama_response

def predict(prompt: str) -> str:
    """
    Entry point for Replicate. Takes a user prompt and returns a CBT therapist response.
    """
    detected_probs = classify_patterns_with_probs(prompt)
    response = generate_llama_response(prompt, detected_probs)
    return response

