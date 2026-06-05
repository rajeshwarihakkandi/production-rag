import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── SETTINGS ──────────────────────────────────────────────
CHUNK_SIZE = 700    # max tokens per chunk
CHUNK_OVERLAP = 100    # overlap between chunks
RAW_DIR = os.path.join("data", "raw")
CHUNKS_DIR = os.path.join("data", "chunks")

os.makedirs(CHUNKS_DIR, exist_ok=True)

# ── SPLITTER ───────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
# separators tells the splitter WHERE it is allowed to cut.
# It tries "\n\n" (paragraph break) first — cleanest cut.
# If the chunk is still too big, it tries "\n" (line break).
# Then ". " (end of sentence).
# Then " " (word boundary — never cuts mid-word).
# "" is the last resort — character level.

# ── LOAD + CHUNK ONE PDF ──────────────────────────────────


def chunk_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"\nProcessing: {filename}")

    # Step 1: load all pages from the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Pages loaded: {len(pages)}")

    all_chunks = []

    for page in pages:
        # Step 2: skip empty pages (common in PDFs)
        if not page.page_content.strip():
            continue

        # Step 3: split this page's text into chunks
        raw_chunks = splitter.split_text(page.page_content)

        # Step 4: wrap each chunk with rich metadata
        for i, chunk_text in enumerate(raw_chunks):
            chunk = {
                "text":        chunk_text,
                "source_file": filename,
                "page_number": page.metadata.get("page", 0) + 1,
                # +1 because LangChain uses 0-indexed pages
                # but humans think of page 1 as the first page
                "chunk_index": i,
                "chunk_size":  len(chunk_text),
                "total_pages": page.metadata.get("total_pages", 0),
            }
            all_chunks.append(chunk)

    print(f"  Chunks created: {len(all_chunks)}")
    return all_chunks


# ── PROCESS ALL PDFs IN data/raw ─────────────────────────
def chunk_all_pdfs():
    pdf_files = [
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".pdf")
    ]

    print(f"Found {len(pdf_files)} PDFs to process")

    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAW_DIR, pdf_file)
        chunks = chunk_pdf(pdf_path)
        all_chunks.extend(chunks)

    # Step 5: save all chunks to a JSON file
    output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Total chunks created : {len(all_chunks)}")
    print(f"Saved to             : {output_path}")
    print(f"{'='*50}")

    return all_chunks


# ── SMOKE TEST ────────────────────────────────────────────
def inspect_chunks(chunks, n=3):
    print(f"\n── Inspecting first {n} chunks ──")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk #{i+1}")
        print(f"  Source : {chunk['source_file']}")
        print(f"  Page   : {chunk['page_number']}")
        print(f"  Size   : {chunk['chunk_size']} chars")
        print(f"  Text preview:")
        print(f"    {chunk['text'][:200]}...")
        print()


if __name__ == "__main__":
    chunks = chunk_all_pdfs()
    inspect_chunks(chunks, n=3)
