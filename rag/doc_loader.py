from langchain_docling.loader import DoclingLoader
from pathlib import Path
from docling.chunking import HybridChunker
from config import EMBED_MODEL_ID
from docling.datamodel.base_models import PipelineOptions
from docling.datamodel.pipeline_options import ApiVlmOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from dotenv import load_dotenv

load_dotenv()

def load_files(files): 
    files_input = Path(files)
    pdf_files = list(files_input.glob("*.pdf"))

    for file in pdf_files:
        print(f"Found : {file}")
    converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                generate_page_images=True,
                images_scale=2.0,
            ),
        )
    }
)
    loader = DoclingLoader(file_path=pdf_files,chunker=HybridChunker(tokenizer=EMBED_MODEL_ID), converter=converter)
    print(f"{type(loader)}")
    return loader.load()

def main():
    input_path="./data"
    docs = load_files(input_path)
    for d in docs[:3]:
        print(f"- {d.page_content=}")

if __name__ == "__main__":
    main()