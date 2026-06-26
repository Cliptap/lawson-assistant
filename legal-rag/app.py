"""
Asistente Legal RAG - Interfaz Web con Streamlit
Caso 3: Asistente Legal - Evaluacion Sumativa
"""
import streamlit as st
import os
import json
import gc
import time
import chromadb

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
HISTORY_FILE = "chat_history.json"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_PDF_PAGES = 15
TOP_K = 4
SIMILARITY_THRESHOLD = 0.30

PROMPT_TEMPLATE = """Eres un asistente legal especializado. Responde la pregunta basandote UNICAMENTE en el contexto legal proporcionado a continuacion.

REGLAS:
1. Si el contexto no contiene informacion suficiente, indica: "No encontre informacion suficiente en los documentos disponibles para responder esta consulta."
2. ANTES de usar un fragmento, verifica que el area legal del documento coincida con el tema de la pregunta. Ejemplo: si preguntan sobre derechos del consumidor, ignora fragmentos de leyes de proteccion de datos personales o normativa laboral.
3. Cita SIEMPRE el nombre completo de la ley y el numero de articulo (ej: "Articulo 20 de la Ley 19.496 de Proteccion de los Derechos de los Consumidores").
4. Si un fragmento no menciona explicitamente un numero de articulo o el nombre de una ley, no lo uses.
5. Responde en espanol, de forma clara y precisa.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

EJEMPLOS = {
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

HIDE_ANCHORS_CSS = """
<style>
    .stDeployButton { display: none !important; }
    .stAppDeployButton { display: none !important; }
    h1 a, h2 a, h3 a, h4 a { display: none !important; }
    /* Boton eliminar: esquina superior derecha, invisible hasta hover */
    [data-testid="stChatMessage"] {
        position: relative !important;
    }
    [data-testid="stChatMessage"] .stButton {
        opacity: 0;
        transition: opacity 0.15s;
        position: absolute !important;
        right: 4px !important;
        top: 4px !important;
        z-index: 5;
    }
    [data-testid="stChatMessage"]:hover .stButton {
        opacity: 1;
    }
    [data-testid="stChatMessage"] button {
        padding: 0px !important;
        min-width: 16px !important;
        width: 16px !important;
        height: 16px !important;
        min-height: 16px !important;
        line-height: 16px !important;
        font-size: 10px !important;
        background: transparent !important;
        border: none !important;
        border-radius: 2px !important;
        color: #bbb !important;
    }
    [data-testid="stChatMessage"] button:hover {
        color: #e74c3c !important;
        background: rgba(231,76,60,0.08) !important;
    }
</style>
"""


def format_docs(docs):
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "desconocida")
        page = doc.metadata.get("page", None)
        ref = src
        if page is not None:
            ref += f", p. {page}"
        parts.append(f"[Documento: {ref}]\n{doc.page_content}")
    return "\n\n".join(parts)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.1)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def delete_message_pair(idx):
    if 0 <= idx < len(st.session_state.messages):
        del st.session_state.messages[idx]
        if idx < len(st.session_state.messages):
            del st.session_state.messages[idx]
        save_history(st.session_state.messages)
        st.rerun()


def load_and_ingest(data_dir: str = DATA_DIR) -> tuple[int, int]:
    if os.path.exists(CHROMA_DIR):
        st.cache_resource.clear()
        gc.collect()
        time.sleep(0.5)

        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            for col in client.list_collections():
                try:
                    client.delete_collection(col.name)
                except Exception:
                    pass
            del client
            gc.collect()
        except Exception:
            pass

    documents = []
    supported = {".txt", ".md", ".pdf"}

    files = [f for f in os.listdir(data_dir)
             if os.path.splitext(f)[1].lower() in supported]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".txt", ".md"):
            loader = TextLoader(filepath, encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = filename
            documents.extend(loaded)
        else:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            pages_to_load = min(len(reader.pages), MAX_PDF_PAGES)
            loader = PyPDFLoader(filepath)
            loaded = loader.load()
            for doc in loaded[:pages_to_load]:
                doc.metadata["source"] = filename
            documents.extend(loaded[:pages_to_load])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()

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
    vectorstore = load_vectorstore()

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=TOP_K
    )
    relevant = [(doc, score) for doc, score in docs_with_scores
                if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        return ("No encontre informacion suficiente en los documentos "
                "disponibles para responder esta consulta."), []

    filtered_docs = [doc for doc, _ in relevant]
    context = format_docs(filtered_docs)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})

    fragments = [
        (doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
         score, doc.metadata.get("source", ""))
        for doc, score in relevant
    ]

    return response, fragments


def mostrar_chat():
    for i in range(0, len(st.session_state.messages), 2):
        q_msg = st.session_state.messages[i]

        with st.chat_message("user"):
            if st.button("✕", key=f"del_{i}"):
                st.session_state.confirm_delete = i
                st.rerun()
            st.markdown(q_msg["content"])

        if st.session_state.get("confirm_delete") == i:
            with st.container():
                st.warning("¿Eliminar esta conversación?")
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("✓ Confirmar", key=f"confirm_{i}"):
                        delete_message_pair(i)
                with c2:
                    if st.button("Cancelar", key=f"cancel_{i}"):
                        st.session_state.confirm_delete = None
                        st.rerun()

        if i + 1 < len(st.session_state.messages):
            a_msg = st.session_state.messages[i + 1]
            with st.chat_message("assistant"):
                st.markdown(a_msg["content"])

                if a_msg.get("fragments"):
                    with st.expander("📎 Fragmentos recuperados"):
                        for j, (text, score, src) in enumerate(a_msg["fragments"], 1):
                            st.markdown(
                                f"**Fragmento {j}** — `{src}`  "
                                f"(similitud: {score:.2f})"
                            )
                            st.markdown(f"> {text}")
                            st.divider()


def enviar_pregunta(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Buscando en documentos..."):
        response, fragments = query_rag(question)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "fragments": fragments,
    })
    save_history(st.session_state.messages)


def main():
    st.markdown(HIDE_ANCHORS_CSS, unsafe_allow_html=True)
    st.markdown("# ⚖️ Asistente Legal RAG")
    st.caption("Caso 3: Asistente Legal — Evaluación Sumativa · IA Embebida")

    if "messages" not in st.session_state:
        st.session_state.messages = load_history()
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = None

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

    col_chat, col_ejemplos = st.columns([3, 2])

    with col_chat:
        st.markdown("### 💬 Consulta")
        mostrar_chat()

        if prompt := st.chat_input("Escribe tu pregunta legal..."):
            enviar_pregunta(prompt)
            st.rerun()

    with col_ejemplos:
        st.markdown("### 📋 Preguntas de ejemplo")

        for categoria, preguntas in EJEMPLOS.items():
            st.markdown(f"**{categoria}**")
            for p in preguntas:
                if st.button(p, key=p[:50], use_container_width=True):
                    enviar_pregunta(p)
                    st.rerun()
            st.divider()

        st.markdown(
            "<small style='color: gray;'>Arquitectura: Usuario → Embedding → "
            "ChromaDB → Recuperación → LLM (Mistral 7B) → Respuesta + Fuentes</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
