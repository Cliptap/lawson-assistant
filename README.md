# ⚖️ Asistente Legal RAG

**Evaluación Sumativa — IA Embebida en Sistemas Computacionales**  
Caso 3: Asistente Legal · Ingeniería Informática

Asistente basado en arquitectura **RAG** (Retrieval Augmented Generation) que permite consultar normativa legal chilena utilizando documentos propios del dominio jurídico. Corre 100% local con Ollama.

## 🧱 Arquitectura

```
Usuario → Embedding → ChromaDB → Recuperación → LLM (Mistral 7B) → Respuesta + Fuentes
```

| Componente | Tecnología |
|------------|------------|
| Embeddings | nomic-embed-text (Ollama) |
| Base Vectorial | ChromaDB |
| Chunking | RecursiveCharacterTextSplitter (500 chars, overlap 100) |
| LLM | Mistral 7B (Ollama) |
| Interfaz | Streamlit |

## 📁 Corpus Documental

| Documento | Formato | Fuente |
|-----------|---------|--------|
| Ley 19.628 — Protección de Datos Personales | PDF | leychile.cl |
| Ley 21.719 — Nueva Ley de Datos Personales | PDF | leychile.cl |
| Ley 19.496 — Derechos del Consumidor | TXT | — |
| Normativa Laboral — Código del Trabajo | TXT | — |
| Reglamento Interno del Estudio Jurídico | MD | — |

## 🚀 Ejecución

### Requisitos

- [Ollama](https://ollama.com) + modelos `mistral` y `nomic-embed-text`
- Python 3.10+

```bash
# Instalar dependencias
pip install -r requirements.txt

# Indexar documentos (una vez)
python ingest.py

# Iniciar interfaz web
streamlit run app.py
```

Abrir http://localhost:8501

### Docker

```bash
docker compose up --build
```

## 📂 Estructura del proyecto

```
legal-rag/
├── app.py                 # Interfaz web Streamlit
├── ingest.py              # Carga documental + chunking + embeddings
├── query.py               # CLI interactiva
├── data/                  # Corpus documental (PDF, TXT, MD)
├── chroma_db/             # Base vectorial (generada por ingest.py)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🎯 Escenarios de demostración

| Escenario | Ejemplo |
|-----------|---------|
| Simple | ¿Qué derechos tiene el consumidor? |
| Complejo | ¿En qué casos termina el contrato sin indemnización y cuánto se paga? |
| Sin respuesta | ¿Cómo se tramita una visa de trabajo? |
