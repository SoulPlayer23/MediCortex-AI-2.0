import json as _json
import logging
import requests
from typing import Any, Iterator, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from pydantic import Field
from config import settings

logger = logging.getLogger("MedGemmaLLM")


class MedGemmaLLM(LLM):
    """
    Custom LangChain LLM wrapper for the locally hosted MedGemma model.
    Falls back to OpenAI GPT-4o-mini when the local server is unreachable.
    """

    api_url: str = Field(default_factory=lambda: settings.MEDGEMMA_API_URL)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.4)
    top_k: int = Field(default=65)
    top_p: float = Field(default=0.95)
    min_p: float = Field(default=0.0)
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
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
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
            # MedGemma is offline — fall back to Gemma 4 via Ollama Cloud
            logger.warning(f"MedGemma server unreachable ({e}). Falling back to Gemma 4 (Ollama Cloud).")

            try:
                from langchain_ollama import ChatOllama
                from langchain_core.messages import HumanMessage, SystemMessage

                fallback = ChatOllama(
                    model=settings.OLLAMA_CLOUD_MODEL,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    num_predict=self.max_tokens,
                    base_url=settings.OLLAMA_CLOUD_URL.removesuffix("/v1"),
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
                logger.error(f"Gemma 4 fallback also failed: {fallback_error}")
                return (
                    f"Error: MedGemma is offline and the Gemma 4 fallback failed.\n"
                    f"MedGemma error: {e}\n"
                    f"Gemma 4 error: {fallback_error}"
                )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream tokens from MedGemma's /predict/stream SSE endpoint.
        Falls back to a single-chunk yield via _call if the streaming endpoint is
        unreachable or returns an error, so the ReAct loop always gets output.
        """
        stream_url = self.api_url.replace("/predict", "/predict/stream")
        payload = {
            "prompt": prompt,
            "image_base64": None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
        }

        try:
            accumulated = ""
            with requests.post(stream_url, json=payload, timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    event = _json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    token_text = event.get("token", "")
                    if not token_text:
                        continue
                    accumulated += token_text

                    # Honour stop sequences: yield up to the stop word, then halt.
                    if stop:
                        hit = next((s for s in stop if s in accumulated), None)
                        if hit:
                            pre_stop = accumulated.split(hit)[0]
                            already_yielded = accumulated[: len(accumulated) - len(token_text)]
                            new_pre = pre_stop[len(already_yielded):]
                            if new_pre:
                                chunk = GenerationChunk(text=new_pre)
                                if run_manager:
                                    run_manager.on_llm_new_token(new_pre)
                                yield chunk
                            return

                    chunk = GenerationChunk(text=token_text)
                    if run_manager:
                        run_manager.on_llm_new_token(token_text)
                    yield chunk

        except Exception as e:
            logger.warning(f"MedGemma streaming failed ({e}). Falling back to _call.")
            full_text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            yield GenerationChunk(text=full_text)
