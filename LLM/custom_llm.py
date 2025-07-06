import os
import logging
from typing import Optional
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI

# Optional: import tiktoken for OpenAI token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Optional: for more accurate tokenization in other models
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

load_dotenv()

# Setup logger
logger = logging.getLogger("CustomLLMRouter")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class CustomLLMRouter:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.provider, self.model = self._parse_model_name()
        logger.info(f"Initializing model: provider={self.provider}, model={self.model}")
        self.llm = self._load_llm()

    def _parse_model_name(self):
        if ":" in self.model_name:
            return self.model_name.split(":", 1)
        else:
            raise ValueError("Model name must include provider, e.g., 'openai:gpt-4'")

    def _load_llm(self) -> Optional[BaseChatModel]:
        if self.provider == "openai":
            logger.info("Loading OpenAI model...")
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.provider == "ollama":
            logger.info("Loading Ollama model...")
            return ChatOllama(
                model=os.getenv('OLLAMA_CHAT_MODEL', self.model),
                temperature=self.temperature,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
        elif self.provider == "gemini":
            logger.info("Loading Gemini model...")
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif self.provider == "xai":
            logger.info("Loading Grok (xAI) model...")
            return ChatXAI(
                model=self.model,
                temperature=self.temperature,
                xai_api_key=os.getenv("XAI_API_KEY"),
            )
        else:
            logger.error(f"Unsupported provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _count_tokens(self, text: str) -> int:
        """Estimate token count based on provider."""
        try:
            if self.provider == "openai" and tiktoken:
                enc = tiktoken.encoding_for_model(self.model)
                return len(enc.encode(text))

            elif self.provider == "ollama" and AutoTokenizer:
                tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
                return len(tokenizer.encode(text))

            elif self.provider == "gemini":
                # Gemini tokenization is proprietary; use word count estimate
                return len(text.split())

            elif self.provider == "xai":
                # Grok API does not expose tokenizer, use approximation
                return len(text.split())

        except Exception as e:
            logger.warning(f"Token counting failed: {e}")

        # Fallback: rough estimate
        return len(text.split())

    def invoke(self, prompt: str) -> str:
        logger.info(f"Invoking model '{self.model_name}' with prompt: {prompt}")
        try:
            if not self.llm:
                raise RuntimeError("LLM backend not initialized.")

            human_msg = HumanMessage(content=prompt)
            response = self.llm.invoke([human_msg])

            # Token counting
            input_tokens = self._count_tokens(prompt)
            output_tokens = self._count_tokens(response.content)
            total_tokens = input_tokens + output_tokens

            logger.info(f"Prompt tokens: {input_tokens}, Response tokens: {output_tokens}, Total: {total_tokens}")
            logger.info(f"Response: {response.content}")

            return response.content

        except Exception as e:
            logger.exception(f"Error invoking model '{self.model_name}': {e}")
            return f"[ERROR] {str(e)}"
