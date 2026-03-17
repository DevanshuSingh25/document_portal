import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse, PromptType

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.fixing_parser
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=500,
        )
        self.log.info("DocumentComparatorLLM initialized", model=self.llm)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            # Split the combined text into chunks to respect token limits
            chunks = self.splitter.split_text(combined_docs)
            self.log.info("Document chunked for comparison", total_chunks=len(chunks))

            all_rows: list[dict] = []

            for i, chunk in enumerate(chunks[:3]):  # limit to first 3 chunks to avoid rate limits
                self.log.info("Processing comparison chunk", chunk_number=i + 1)
                inputs = {
                    "combined_docs": chunk,
                    "format_instruction": self.parser.get_format_instructions(),
                }
                response = self.chain.invoke(inputs)
                self.log.info("Chunk compared successfully", chunk_number=i + 1)
                if isinstance(response, list):
                    all_rows.extend(response)
                elif isinstance(response, dict):
                    all_rows.append(response)

            self.log.info("Comparison completed", total_rows=len(all_rows))
            return self._format_response(all_rows)

        except Exception as e:
            self.log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:  # type: ignore
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            self.log.error("Error formatting response into DataFrame", error=str(e))
            DocumentPortalException("Error formatting response", sys)

