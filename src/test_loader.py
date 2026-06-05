from langchain_community.document_loaders import PyPDFLoader
import os

pdf_path = os.path.join("data", "raw", "rag_original_paper.pdf")

print(f"Loading: {pdf_path}")
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"Pages loaded: {len(pages)}")
print(f"\n--- First 500 characters of page 1 ---\n")
print(pages[0].page_content[:500])
print(f"\n--- Metadata ---")
print(pages[0].metadata)