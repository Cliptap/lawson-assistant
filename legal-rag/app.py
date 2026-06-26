"""
Asistente Legal RAG - Interfaz Web con Streamlit
Caso 3: Asistente Legal - Evaluacion Sumativa
"""
import streamlit as st
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="Asistente Legal RAG",
    page_icon="⚖️",
    layout="wide",
)

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
SIMILARITY_THRESHOLD = 0.3

PROMPT_TEMPLATE = """Eres un asistente legal especializado. Responde la pregunta basandote UNICAMENTE en el contexto legal proporcionado a continuacion.

Si el contexto no contiene informacion suficiente para responder, indica claramente: "No encontre informacion suficiente en los documentos disponibles para responder esta consulta."

Responde en espanol, de forma clara y precisa. Cita los articulos y leyes mencionados en el contexto cuando corresponda.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


def format_docs(docs):
    return "\n\n".join(
        f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}"
        for doc in docs
    )


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.1)


def load_and_ingest(data_dir: str = DATA_DIR) -> tuple[int, int]:
    """Carga documentos, genera chunks y los indexa en ChromaDB."""
    documents = []
    supported = {".txt", ".md", ".pdf"}

    files = [f for f in os.listdir(data_dir)
             if os.path.splitext(f)[1].lower() in supported]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".txt", ".md"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            loader = PyPDFLoader(filepath)

        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = filename
        documents.extend(loaded)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return len(documents), len(chunks)


def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def query_rag(question: str):
    """Ejecuta el pipeline RAG completo y retorna respuesta + fuentes."""
    vectorstore = load_vectorstore()

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=TOP_K
    )
    relevant = [(doc, score) for doc, score in docs_with_scores
                if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        return ("No encontre informacion suficiente en los documentos "
                "disponibles para responder esta consulta."), [], []

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)

    sources = []
    seen = set()
    for doc, score in relevant:
        source = doc.metadata.get("source", "desconocida")
        if source not in seen:
            seen.add(source)
            sources.append(source)

    fragments = [(doc.page_content[:300] + "...", score, doc.metadata.get("source", ""))
                 for doc, score in relevant]

    return response, sources, fragments


def main():
    st.title("⚖️ Asistente Legal RAG")
    st.caption("Caso 3: Asistente Legal — Evaluación Sumativa · IA Embebida")

    with st.sidebar:
        st.header("📁 Configuración")

        if st.button("🔄 Indexar Documentos", use_container_width=True):
            with st.spinner("Cargando y procesando documentos..."):
                try:
                    ndocs, nchunks = load_and_ingest(DATA_DIR)
                    st.success(f"{ndocs} documentos → {nchunks} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        st.markdown("**Parámetros RAG**")
        st.markdown(f"- Embedding: `{EMBEDDING_MODEL}`")
        st.markdown(f"- LLM: `{LLM_MODEL}`")
        st.markdown(f"- Chunk size: `{CHUNK_SIZE}`")
        st.markdown(f"- Overlap: `{CHUNK_OVERLAP}`")
        st.markdown(f"- Top-K: `{TOP_K}`")
        st.markdown(f"- Threshold: `{SIMILARITY_THRESHOLD}`")

        st.divider()

        st.markdown("**📂 Documentos cargados**")
        supported = {".txt", ".md", ".pdf"}
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR)
                     if os.path.splitext(f)[1].lower() in supported]
            for f in files:
                st.markdown(f"- {f}")
        else:
            st.markdown("*No hay documentos.*")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("💬 Consulta")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Escribe tu pregunta legal..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Buscando en documentos..."):
                    response, sources, fragments = query_rag(prompt)

                st.markdown(response)

                if fragments:
                    with st.expander("📎 Fragmentos recuperados"):
                        for i, (text, score, src) in enumerate(fragments, 1):
                            st.markdown(f"**Fragmento {i}** — `{src}`  "
                                        f"(similitud: {score:.2f})")
                            st.markdown(f"> {text}")
                            st.divider()

    with col2:
        st.subheader("📋 Preguntas de ejemplo")

        ejemplos = {
            "Simple": [
                "¿Qué establece la Ley 19.628 sobre el consentimiento?",
                "¿Cuáles son los derechos del consumidor?",
                "¿Cuántas horas máximas dura la jornada laboral?",
            ],
            "Compleja": [
                "¿Qué derechos tiene un consumidor con un producto defectuoso y qué obligaciones tiene el empleador con los datos personales de sus trabajadores?",
                "¿En qué casos termina el contrato de trabajo sin indemnización y cuánto debe pagarse cuando sí corresponde?",
            ],
            "Sin respuesta": [
                "¿Cuál es el procedimiento para registrar una patente en Chile?",
                "¿Cómo se tramita una visa de trabajo?",
            ],
        }

        for categoria, preguntas in ejemplos.items():
            st.markdown(f"**{categoria}**")
            for p in preguntas:
                if st.button(p, key=p[:50], use_container_width=True):
                    st.session_state.messages.append(
                        {"role": "user", "content": p}
                    )
                    with st.chat_message("user"):
                        st.markdown(p)
                    with st.chat_message("assistant"):
                        with st.spinner("Buscando..."):
                            resp, srcs, frags = query_rag(p)
                        st.markdown(resp)
                        if frags:
                            with st.expander("📎 Fragmentos recuperados"):
                                for i, (text, score, src) in enumerate(frags, 1):
                                    st.markdown(f"**Fragmento {i}** — `{src}`  "
                                                f"(similitud: {score:.2f})")
                                    st.markdown(f"> {text}")
                                    st.divider()
            st.divider()

        st.markdown("---")
        st.caption(
            "Arquitectura: Usuario → Embedding → ChromaDB → "
            "Recuperación → LLM (Mistral 7B) → Respuesta + Fuentes"
        )


if __name__ == "__main__":
    main()
