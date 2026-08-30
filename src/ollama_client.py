"""
Ollama API client for LLM agent communication
"""
import requests
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 200
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_REPEAT_LAST_N = 128
DEFAULT_MIN_P = 0.05
DEFAULT_SEED = None  # None = sampler stays random, as upstream
DEFAULT_KEEP_ALIVE = "4h"  # Keep the model resident; a cold reload breaks twin runs
DEFAULT_THINK = None  # None = leave the model's own thinking default untouched
DEFAULT_TIMEOUT_SECONDS = 60  # Read timeout for one /api/generate call [s]
CONNECTION_CHECK_TIMEOUT = 5


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        repeat_last_n: int = DEFAULT_REPEAT_LAST_N,
        min_p: float = DEFAULT_MIN_P,
        seed: Optional[int] = DEFAULT_SEED,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        think: Optional[bool] = DEFAULT_THINK,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.min_p = min_p
        self.seed = seed
        self.keep_alive = keep_alive
        self.think = think
        self.timeout_seconds = timeout_seconds
        self.api_url = f"{self.base_url}/api/generate"

    def _build_generate_payload(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict:
        """Build the /api/generate request body for a single prompt."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "repeat_penalty": self.repeat_penalty,
                "repeat_last_n": self.repeat_last_n,
                "min_p": self.min_p
            }
        }

        # Counterfactual twin runs are only comparable if the sampler replays
        # identically, so the seed is pinned per call rather than left to the
        # server. Omitted when unset so upstream behaviour is unchanged.
        if self.seed is not None:
            payload["options"]["seed"] = self.seed

        # Pin the model in memory for the length of the run. A model that gets
        # evicted between two calls comes back cold, and a cold model generates
        # different messages and memories from a resident one - which silently
        # splits twin runs apart before the ad ever lands. Relying on the
        # server's environment for this proved unreliable: one of our three
        # nodes had OLLAMA_KEEP_ALIVE set after the server had already started,
        # so it was never picked up.
        payload["keep_alive"] = self.keep_alive

        # Thinking models (qwen3, gpt-oss, ...) return their reasoning in a
        # separate "thinking" field that this client never reads, yet those
        # reasoning tokens still count against num_predict. Left enabled, the
        # whole budget is spent on reasoning and "response" arrives empty.
        # Omitted rather than defaulted when unconfigured, so the request
        # cannot override a model's own thinking default by accident.
        if self.think is not None:
            payload["think"] = self.think

        return payload

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text using Ollama API

        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature (uses instance default if None)
            max_tokens: Maximum tokens to generate (uses instance default if None)

        Returns:
            Generated text response
        """
        # Use instance defaults if not specified
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            payload = self._build_generate_payload(prompt, temperature, max_tokens)

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in Ollama client: {e}")
            return ""
    
    def fetch_models(self) -> Optional[List[str]]:
        """
        Fetch the model names available on the Ollama server.

        Reachability and the model list come from the same endpoint, so both are
        answered by a single request. The return value distinguishes the two
        failure modes a caller needs to report differently:

        Returns:
            List of model names, [] if the server has none,
            or None if the server could not be reached
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=CONNECTION_CHECK_TIMEOUT
            )
            response.raise_for_status()
            return [model['name'] for model in response.json().get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return None


