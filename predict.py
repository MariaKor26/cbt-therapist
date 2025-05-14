from cog import BasePredictor, Input
from model_utils import (
    classify_patterns_with_probs,
    generate_llama_response
)

class Predictor(BasePredictor):
    def setup(self):
        """Load models and resources into memory"""
        pass  # models are loaded automatically in model_utils on import

    def predict(
        self,
        user_input: str = Input(description="User message or thought to analyze"),
        threshold: float = Input(description="Confidence threshold", default=0.8)
    ) -> str:
        """Run full CBT analysis and response generation pipeline"""
        detected = classify_patterns_with_probs(user_input)
        if not detected:
            return "No strong cognitive distortion patterns detected. Try rephrasing the input."

        response = generate_llama_response(user_input, detected)
        return response