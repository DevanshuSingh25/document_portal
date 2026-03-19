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


    def load_llm(self):
        """
        load and return the LLM.
        """

        llm_block = self.config["llm"]
        log.info("Loading LLM model...")
        
        provider_key = os.getenv("LLM_PROVIDER", "groq")
        if provider_key not in llm_block:
            log.error(f"LLM provider '{provider_key}' not found in configuration", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' not found in configuration")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        tempreture = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 2048)

        log.info("Loading LLM", provider=provider, model_name=model_name, temperature=tempreture, max_tokens=max_tokens)

        if provider == "groq":
            llm=ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                temperature=tempreture
            )

            return llm
        
        else:
            log.error(f"Unsupported LLM provider: {provider}", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")





if __name__ == "__main__":
    loader= ModelLoader()

    embeddings = loader.load_embeddings()
    print("Embeddings loaded successfully.")

    llm = loader.load_llm()
    print("LLM loaded successfully.")

    result = llm.invoke("Hello, how are you?")
    print(f"LLM response: {result.content}")
