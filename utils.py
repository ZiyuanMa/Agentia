import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI, AsyncOpenAI
from config import OPENAI_API_KEY, MODEL_NAME, OPENAI_BASE_URL

logger = logging.getLogger("Simworld")


def _record_api_call():
    """Record an API call to stats if available."""
    try:
        from logger_config import get_stats
        stats = get_stats()
        if stats:
            stats.record_api_call()
    except ImportError:
        pass


def _record_error():
    """Record an error to stats if available."""
    try:
        from logger_config import get_stats
        stats = get_stats()
        if stats:
            stats.record_error()
    except ImportError:
        pass


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if not api_key:
            api_key = OPENAI_API_KEY
        
        if not base_url:
            base_url = OPENAI_BASE_URL
        
        if not api_key:
            logger.warning("No OpenAI API Key found. LLM calls will fail.")
            api_key = "dummy-key-for-init"
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = MODEL_NAME

    def chat_completion(self, messages: List[Dict[str, str]], 
                        tools: Optional[List[Dict]] = None,
                        response_format: Optional[Dict] = None) -> Any:
        """Sync chat completion for EnvAgent with optional JSON output mode."""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.95,
                "extra_body": {
                    "thinking": {"type": "disabled"}
                }
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            if response_format:
                params["response_format"] = response_format

            response = self.client.chat.completions.create(**params)
            _record_api_call()
            return response.choices[0].message
        except Exception as e:
            logger.error(f"LLM API Call Error: {e}")
            _record_error()
            return None

    async def async_chat_completion(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Any:
        """Async chat completion with optional function calling."""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "top_p": 0.95,
                "extra_body": {
                    "thinking": {"type": "disabled"}
                }
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            response = await self.async_client.chat.completions.create(**params)
            _record_api_call()
            return response.choices[0].message
        except Exception as e:
            logger.error(f"Async LLM Chat Completion Error: {e}")
            _record_error()
            return None
