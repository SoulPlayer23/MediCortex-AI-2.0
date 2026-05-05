import json as _json
import logging
import requests
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from pydantic import Field
from config import settings

logger = logging.getLogger("MedGemmaLLM")


def _runpod_auth_headers() -> Dict[str, str]:
    """Return Authorization header dict if RUNPOD_API_KEY is configured."""
    if settings.RUNPOD_API_KEY:
        return {"Authorization": f"Bearer {settings.RUNPOD_API_KEY}"}
    return {}


def _looks_like_runpod(url: str) -> bool:
    return "runpod.ai" in url.lower() or "/runsync" in url.lower() or "/run" in url.lower().split("/")[-1:]


def _unwrap_response(payload: Any) -> str:
    """Extract generated text from a heterogeneous JSON response.

    Supports three known shapes:
    - Local medgemma-host:  {"response": "..."}
    - RunPod /runsync:       {"output": {"text": "..."}, "status": "COMPLETED"}
    - RunPod /runsync (alt): {"output": "..."} or {"output": {"response": "..."}}
    """
    if not isinstance(payload, dict):
        return str(payload or "")

    # Local medgemma-host contract
    if "response" in payload and isinstance(payload["response"], str):
        return payload["response"]

    # RunPod
    out = payload.get("output")
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for key in ("text", "response", "output", "generated_text"):
            v = out.get(key)
            if isinstance(v, str):
                return v
    return ""


class MedGemmaLLM(LLM):
    """
    Custom LangChain LLM wrapper for the MedGemma model, served either by the
    local medgemma-host container (returns {"response": "..."}) or by RunPod
    Serverless /runsync (returns {"output": {"text": "..."}}).

    Falls back to Gemma3:1b via Ollama when the MedGemma endpoint is unreachable
    or returns an HTTP error (RunPod cold-start, server down, etc.).
    """

    api_url: str = Field(default_factory=lambda: settings.MEDGEMMA_API_URL)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.4)
    top_k: int = Field(default=65)
    top_p: float = Field(default=0.95)
    min_p: float = Field(default=0.0)
    # DEPLOY-2: shorter default so RunPod cold starts fall back to Gemma3:1b fast
    # rather than blocking the user for the full inference window.
    timeout: int = Field(default_factory=lambda: settings.MEDGEMMA_TIMEOUT_SECONDS)

    @property
    def _llm_type(self) -> str:
        return "medgemma_local"

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """Adapt the request body shape to the endpoint family."""
        body = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
        }
        if _looks_like_runpod(self.api_url):
            # RunPod serverless wraps inputs under "input"
            return {"input": body}
        return body

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._build_payload(prompt)
        headers = _runpod_auth_headers()

        try:
            response = requests.post(
                self.api_url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            text_output = _unwrap_response(response.json())

            if stop:
                for stop_seq in stop:
                    if stop_seq in text_output:
                        text_output = text_output.split(stop_seq)[0]

            return text_output

        except requests.exceptions.RequestException as e:
            # MedGemma is offline / cold-starting / timing out — fall back to Gemma3:1b
            logger.warning(
                f"MedGemma server unreachable ({e}). Falling back to Gemma3:1b (Ollama)."
            )
            return self._gemma3_fallback(prompt, stop=stop, primary_error=str(e))

    def _gemma3_fallback(
        self, prompt: str, stop: Optional[List[str]] = None, primary_error: str = ""
    ) -> str:
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
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
            )

            if "New input:" in prompt:
                parts = prompt.split("New input:")
                sys_msg = parts[0].strip()
                human_msg = "New input:" + parts[1]
                messages = [SystemMessage(content=sys_msg), HumanMessage(content=human_msg)]
            else:
                messages = [HumanMessage(content=prompt)]

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
            logger.error(f"Gemma3:1b fallback also failed: {fallback_error}")
            return (
                f"Error: MedGemma is offline and the Gemma3:1b fallback failed.\n"
                f"MedGemma error: {primary_error}\n"
                f"Gemma3:1b error: {fallback_error}"
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
        unreachable. RunPod /runsync does not stream — for RunPod, _stream
        immediately fans out to _call.
        """
        if _looks_like_runpod(self.api_url):
            # RunPod /runsync is non-streaming. Yield a single chunk.
            full_text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            yield GenerationChunk(text=full_text)
            return

        stream_url = self.api_url.replace("/predict", "/predict/stream")
        payload = self._build_payload(prompt)
        headers = _runpod_auth_headers()

        try:
            accumulated = ""
            with requests.post(
                stream_url, json=payload, headers=headers, timeout=self.timeout, stream=True
            ) as resp:
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
