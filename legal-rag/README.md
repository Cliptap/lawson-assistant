# ⚖️ Asistente Legal RAG

**Caso 3: Asistente Legal** — Evaluación Sumativa · IA Embebida en Sistemas Computacionales

Asistente basado en arquitectura **RAG** (Retrieval Augmented Generation) para consultar normativa legal utilizando documentos propios del dominio jurídico.

## 📁 Corpus Documental

- Ley de Protección de Datos Personales (Ley 19.628)
- Ley de Protección de los Derechos de los Consumidores (Ley 19.496)
- Normativa Laboral — Código del Trabajo
- Reglamento Interno del Estudio Jurídico

## 🧱 Arquitectura

```
Usuario → Embedding → ChromaDB → Recuperación → LLM (Mistral 7B) → Respuesta + Fuentes
```

| Componente | Tecnología |
|------------|------------|
| Embeddings | nomic-embed-text (Ollama) |
| Base Vectorial | ChromaDB |
| Chunking | RecursiveCharacterTextSplitter (500 tokens, overlap 100) |
| LLM | Mistral 7B (Ollama) |
| Interfaz | Streamlit |

## 🚀 Ejecución

### Requisitos previos

1. Instalar [Ollama](https://ollama.com/download/windows)
2. Descargar los modelos:
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral
   ```
3. Python 3.10+ y dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Opción 1: Interfaz Web (recomendada)

```bash
python ingest.py          # Indexar documentos (una sola vez)
streamlit run app.py      # Iniciar aplicación web
```

Abrir http://localhost:8501

### Opción 2: CLI

```bash
python ingest.py          # Indexar documentos
python query.py           # Consola interactiva
```

### Opción 3: Docker

```bash
docker compose up --build
```

## 🎯 Escenarios de demostración

| Escenario | Ejemplo |
|-----------|---------|
| Consulta simple | ¿Qué derechos tiene el consumidor? |
| Consulta compleja | ¿Qué obligaciones tiene el empleador con datos personales y qué puede hacer un consumidor con un producto defectuoso? |
| Sin respuesta | ¿Cómo se registra una patente en Chile? |

## 📊 Parámetros del chunking

- **Tamaño**: 500 caracteres — suficiente para contener artículos completos
- **Solapamiento**: 100 caracteres — mantiene coherencia entre fragmentos adyacentes
