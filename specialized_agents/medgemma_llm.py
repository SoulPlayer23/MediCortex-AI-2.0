import logging
import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from config import settings

logger = logging.getLogger("MedGemmaLLM")


class MedGemmaLLM(LLM):
    """
    Custom LangChain LLM wrapper for the locally hosted MedGemma model.
    Falls back to OpenAI GPT-4o-mini when the local server is unreachable.
    """

    api_url: str = Field(default_factory=lambda: settings.MEDGEMMA_API_URL)
    max_tokens: int = Field(default=1024)  # Bumped for fuller clinical responses
    temperature: float = Field(default=0.0)
    timeout: int = Field(default=120)  # MedGemma inference takes ~17s; allow headroom

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
        Execute the query against the local MedGemma API.
        Falls back to OpenAI GPT-4o-mini if MedGemma is offline.
        """
        payload = {
            "prompt": prompt,
            "image_base64": None,
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            text_output = response.json().get("response", "")

            if stop:
                for stop_seq in stop:
                    if stop_seq in text_output:
                        text_output = text_output.split(stop_seq)[0]

            return text_output

        except requests.exceptions.RequestException as e:
            # MedGemma is offline — fall back to OpenAI GPT-4o-mini
            logger.warning(f"MedGemma server unreachable ({e}). Falling back to OpenAI GPT-4o-mini.")

            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage

                fallback = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=self.temperature,
                    api_key=settings.OPENAI_API_KEY,
                    max_tokens=self.max_tokens,
                )
                
                # Split prompt into System and Human messages for better instruction following
                if "New input:" in prompt:
                    parts = prompt.split("New input:")
                    sys_msg = parts[0].strip()
                    human_msg = "New input:" + parts[1]
                    messages = [SystemMessage(content=sys_msg), HumanMessage(content=human_msg)]
                else:
                    messages = [HumanMessage(content=prompt)]

                # Pass kwargs explicitly (including stop words for ReAct)
                invoke_kwargs = {}
                if stop:
                     invoke_kwargs["stop"] = stop
                     
                text_output = fallback.invoke(messages, **invoke_kwargs).content

                if stop:
                    for stop_seq in stop:
                        if stop_seq in text_output:
                            text_output = text_output.split(stop_seq)[0]

                return text_output

            except Exception as fallback_error:
                logger.error(f"OpenAI fallback also failed: {fallback_error}")
                return (
                    f"Error: MedGemma is offline and the OpenAI fallback failed.\n"
                    f"MedGemma error: {e}\n"
                    f"OpenAI error: {fallback_error}"
                )
