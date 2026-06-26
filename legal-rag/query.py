"""
Sistema de consultas RAG: recibe pregunta del usuario, busca en ChromaDB
los fragmentos mas relevantes y genera respuesta con Mistral via Ollama.
"""
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.64

PROMPT_TEMPLATE = """Eres un asistente legal especializado. Responde la pregunta basandote UNICAMENTE en el contexto legal proporcionado a continuacion.

REGLAS:
1. Si el contexto no contiene informacion suficiente para responder, indica: "No encontre informacion suficiente en los documentos disponibles para responder esta consulta."
2. Descarta fragmentos de leyes o documentos que no guarden relacion directa con la pregunta.
3. Cita SIEMPRE la ley especifica y el numero de articulo correspondiente (ej: "Articulo 20 de la Ley 19.496 de Proteccion de los Derechos de los Consumidores").
4. Si un fragmento no menciona un numero de articulo o ley, no lo uses para responder.
5. Responde en espanol, de forma clara y precisa.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


def format_docs(docs):
    return "\n\n".join(
        f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}"
        for doc in docs
    )


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def main():
    print("=" * 60)
    print("  ASISTENTE LEGAL RAG - Sistema de Consultas")
    print("=" * 60)
    print(f"  LLM: {LLM_MODEL} | Top-K: {TOP_K} | Threshold: {SIMILARITY_THRESHOLD}")
    print("  Escribe 'salir' para terminar.")
    print("=" * 60)

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nAsistente listo.\n")

    while True:
        try:
            question = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if question.strip().lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        if not question.strip():
            continue

        print("\nBuscando en documentos...")

        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            question, k=TOP_K
        )

        relevant_docs = [
            doc for doc, score in docs_with_scores if score >= SIMILARITY_THRESHOLD
        ]

        if not relevant_docs:
            print(
                "\n[!] No encontre informacion suficiente en los documentos "
                "disponibles para responder esta consulta.\n"
            )
            continue

        print(f"  {len(relevant_docs)} fragmentos relevantes encontrados.\n")
        print("Respuesta:")

        try:
            response = chain.invoke(question)
            print(f"\n{response}\n")
        except Exception as e:
            print(f"\n[ERROR] Fallo al generar respuesta: {e}\n")

        print("-" * 60)
        print("Fragmentos recuperados:")
        seen = set()
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            if score < SIMILARITY_THRESHOLD:
                continue
            source = doc.metadata.get("source", "desconocida")
            print(f"\n  [{i}] {source}  (similitud: {score:.2f})")
            print(f"      \"{doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}\"")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
