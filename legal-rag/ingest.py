"""
Ingesta documental: carga PDFs, TXT y MD, los divide en chunks,
genera embeddings y los almacena en ChromaDB.
"""
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(data_dir: str) -> list:
    docs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt" or ext == ".md":
            loader = TextLoader(filepath, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(filepath)
        else:
            print(f"  [SKIP] Formato no soportado: {filename}")
            continue

        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = filename
        docs.extend(loaded)
        print(f"  [OK] Cargado: {filename} ({len(loaded)} paginas/secciones)")

    return docs


def main():
    print("=" * 60)
    print("  ASISTENTE LEGAL RAG - Ingesta Documental")
    print("=" * 60)
    print(f"  Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print(f"  Embeddings: {EMBEDDING_MODEL}")
    print()

    print("[1/3] Cargando documentos...")
    documents = load_documents(DATA_DIR)
    print(f"  Total documentos cargados: {len(documents)}")

    print(f"\n[2/3] Dividiendo en chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Total chunks generados: {len(chunks)}")

    print(f"\n[3/3] Generando embeddings y guardando en ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DIR):
        import chromadb
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            for col in client.list_collections():
                try:
                    client.delete_collection(col.name)
                except Exception:
                    pass
            print(f"  Colecciones anteriores eliminadas.")
        except Exception:
            pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Base vectorial creada en: {CHROMA_DIR}")
    print(f"  Total vectores: {vectorstore._collection.count()}")

    print("\n" + "=" * 60)
    print("  Ingesta completada exitosamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
