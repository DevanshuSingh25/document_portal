import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings



log = CustomLogger().get_logger(__name__)

class ModelLoader:
    def __init__(self):
        """
        load environment variables.
        """
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded successfully.", config_keys=list(self.config.keys()))

    def _validate_env(self):
        """
        Validate necessary environment variables.
        Ensure API keys exists
        """

        required_vars = ['HUGGINGFACEHUB_API_TOKEN', 'GROQ_API_KEY']
        self.api_keys = {key: os.getenv(key) for key in required_vars}

        missing = [k for k, v in self.api_keys.items() if not v]

        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise DocumentPortalException("Missing environment variables", sys)

        log.info("Environment variables validated.", available_keys=[k for k in self.api_keys if self.api_keys[k]])


    def load_embeddings(self):
        """
        load and return the embedding model.
        """

        try:
            log.info("Loading embedding model...")
            model_name = self.config["embedding_model"]["model_name"]
            return HuggingFaceEndpointEmbeddings(model=model_name)
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
                api_key=self.api_keys["GROQ_API_KEY"],
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
