import os
import sys
import json
from dotenv import load_dotenv
from logger.custom_logger import CustomLogger

log = CustomLogger().get_logger(__name__)
from exception.custom_exception import DocumentPortalException
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings



class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))


    def load_embeddings(self):
        """
        load and return the embedding model.
        """

        try:
            log.info("Loading embedding model...")
            model_name = self.config["embedding_model"]["model_name"]
            hf_token = self.api_key_mgr.get("HUGGINGFACEHUB_API_TOKEN")
            return HuggingFaceEndpointEmbeddings(
                model=model_name,
                huggingfacehub_api_token=hf_token
            )
        except Exception as e:
            log.error("Failed to load embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)


    def _base_llm_config(self):
        """Return (provider, model_name, temperature, max_tokens, api_key) from config."""
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "groq")
        if provider_key not in llm_block:
            raise ValueError(f"Provider '{provider_key}' not found in configuration")
        cfg = llm_block[provider_key]
        return (
            cfg.get("provider"),
            cfg.get("model_name"),
            cfg.get("temperature", 0),
            cfg.get("max_tokens", 2048),
            self.api_key_mgr.get("GROQ_API_KEY"),
        )

    def load_llm(self):
        """
        Load LLM with JSON object mode forced.
        Use for analysis tasks where the response MUST be a JSON object.
        """
        provider, model_name, temp, max_tok, api_key = self._base_llm_config()
        log.info("Loading LLM (JSON mode)", provider=provider, model=model_name, temperature=temp)
        if provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temp,
                max_tokens=max_tok,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def load_llm_text(self):
        """
        Load LLM without JSON mode — for comparison and chat where the
        response may be a JSON array or free text.
        """
        provider, model_name, temp, max_tok, api_key = self._base_llm_config()
        log.info("Loading LLM (text mode)", provider=provider, model=model_name, temperature=temp)
        if provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=temp,
                max_tokens=max_tok,
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")







if __name__ == "__main__":
    loader= ModelLoader()

    embeddings = loader.load_embeddings()
    print("Embeddings loaded successfully.")

    llm = loader.load_llm()
    print("LLM loaded successfully.")

    result = llm.invoke("Hello, how are you?")
    print(f"LLM response: {result.content}")
