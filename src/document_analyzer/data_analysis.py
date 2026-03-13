import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            
            self.splitter = RecursiveCharacterTextSplitter(
                            chunk_size=4000,
                            chunk_overlap=500
                        )

            self.log.info("DocumentAnalyzer initialized successfully")
            
            
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)
        
        
    
    # def analyze_document(self, document_text:str)-> dict:
    #     """
    #     Analyze a document's text and extract structured metadata & summary.
    #     """
    #     try:
    #         chain = self.prompt | self.llm | self.fixing_parser
            
    #         self.log.info("Meta-data analysis chain initialized")

    #         response = chain.invoke({
    #             "format_instructions": self.parser.get_format_instructions(),
    #             "document_text": document_text
    #         })

    #         self.log.info("Metadata extraction successful", keys=list(response.keys()))
            
    #         return response

    #     except Exception as e:
    #         self.log.error("Metadata analysis failed", error=str(e))
    #         raise DocumentPortalException("Metadata extraction failed",sys)
        
    def analyze_document(self, document_text: str) -> dict:

        try:

            chain = self.prompt | self.llm | self.fixing_parser

            self.log.info("Metadata analysis chain initialized")

            # Split text into chunks
            chunks = self.splitter.split_text(document_text)

            self.log.info("Document chunked", total_chunks=len(chunks))

            results = []

            for i, chunk in enumerate(chunks[:3]):  # limit chunks to avoid high cost

                self.log.info("Processing chunk", chunk_number=i+1)

                response = chain.invoke({
                    "format_instructions": self.parser.get_format_instructions(),
                    "document_text": chunk
                })

                results.append(response)

            # Combine results (simple merge strategy)
            final_result = results[0] if results else {}

            self.log.info("Metadata extraction successful")

            return final_result

        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed") from e
    
