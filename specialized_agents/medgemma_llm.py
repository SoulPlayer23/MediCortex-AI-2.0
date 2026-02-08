import requests
from typing import Any, List, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

class MedGemmaLLM(LLM):
    """
    Custom LangChain LLM wrapper for the locally hosted MedGemma model.
    """
    
    api_url: str = Field(default="http://100.107.2.102:8000/predict")
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.0) # Not used by current API but good for compatibility

    @property
    def _llm_type(self) -> str:
        return "medgemma_local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the query against the MedGemma API.
        """
        payload = {
            "prompt": prompt,
            "image_base64": None,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            # The API returns {"response": "..."}
            text_output = result.get("response", "")
            
            if stop:
                # Basic client-side stop sequence handling
                for stop_seq in stop:
                    if stop_seq in text_output:
                        text_output = text_output.split(stop_seq)[0]
            
            return text_output

        except requests.exceptions.RequestException as e:
            return f"Error communicating with MedGemma: {str(e)}"
