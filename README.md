# ⚖️ Asistente Legal RAG

**Evaluación Sumativa — IA Embebida en Sistemas Computacionales**  
Caso 3: Asistente Legal · Ingeniería Informática

Asistente basado en arquitectura **RAG** (Retrieval Augmented Generation) que permite consultar normativa legal chilena utilizando documentos propios del dominio jurídico. Corre 100% local con Ollama.

## 🧱 Arquitectura y decisiones de diseño

```
Usuario ──► Embedding ──► ChromaDB ──► Recuperación ──► Mistral 7B ──► Respuesta + Fuentes
                 ▲                               │
                 └─── mismo modelo ──────────────┘
```

### 1. Ingesta documental — `PyPDF` + `TextLoader`

Se eligió `langchain_community.document_loaders` porque unifica la carga de PDF, TXT y Markdown bajo una misma interfaz, evitando escribir parsers separados para cada formato. Los PDFs se limitan a 15 páginas para no saturar el embedding con contenido redundante (preámbulos legales extensos).

### 2. Fragmentación (chunking) — `RecursiveCharacterTextSplitter`

- **Tamaño: 500 caracteres** — suficiente para contener un artículo legal completo con sus incisos. Fragmentos más grandes diluyen la precisión semántica; más chicos pierden contexto jurídico.
- **Solapamiento: 100 caracteres** — evita que un artículo cortado en el borde de un chunk pierda la conexión con sus incisos. El 20% de overlap es un balance estándar entre redundancia y coherencia.
- **Separadores**: `\n\n` → `\n` → `. ` → `espacio` — prioriza cortes en párrafos y oraciones antes que partir palabras, respetando la estructura natural del texto legal.

### 3. Embeddings — `nomic-embed-text` (Ollama)

Se eligió un modelo local en vez de APIs externas (OpenAI, Cohere) por tres razones: **(a)** los documentos legales pueden contener información sensible que no debe salir del entorno local, **(b)** elimina la dependencia de conexión a internet y costos por token, y **(c)** `nomic-embed-text` genera vectores de 768 dimensiones con buen rendimiento en español jurídico, validado empíricamente con los scores de similitud obtenidos (0.64–0.72 en fragmentos relevantes).

### 4. Base vectorial — `ChromaDB`

ChromaDB se prefirió sobre FAISS porque: **(a)** es persistente por defecto — los embeddings sobreviven a reinicios sin configuración adicional, **(b)** almacena metadata (fuente, página) junto a cada vector, permitiendo mostrar las fuentes en la respuesta, y **(c)** se integra nativamente con LangChain mediante `langchain-chroma`, simplificando el pipeline.

### 5. Recuperación — búsqueda por similitud de coseno

- **Top-K = 3** — recuperar solo 3 fragmentos reduce ruido. Con K=4 aparecían fragmentos de leyes no relacionadas (ej: Ley 21.719 en consultas laborales).
- **Threshold = 0.64** — filtro empírico: los fragmentos relevantes puntúan ≥0.64, mientras que los de otras leyes quedan en ~0.63 o menos. Este umbral se ajustó iterativamente probando consultas de los 3 escenarios.

### 6. Generación — `Mistral 7B` (Ollama)

Mistral 7B se eligió porque: **(a)** cumple el requisito de ≤7B parámetros, **(b)** tiene buen desempeño en español a pesar de ser un modelo multilingual, y **(c)** corre localmente en Ollama sin GPU gracias a su tamaño contenido (~4 GB). La temperatura se fijó en 0.1 para minimizar alucinaciones y mantener respuestas deterministas basadas estrictamente en el contexto recuperado.

### 7. Interfaz — `Streamlit`

Streamlit permite construir una UI web funcional en un solo archivo Python, con componentes nativos de chat (`st.chat_message`, `st.chat_input`) que reflejan el flujo conversacional del asistente sin necesidad de escribir HTML/JS.

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
