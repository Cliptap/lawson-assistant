# Informe — Asistente Legal RAG

**Evaluación Sumativa · IA Embebida en Sistemas Computacionales**  
Caso 3: Asistente Legal · Equipo: [COMPLETAR]

---

## 1. Descripción del problema abordado

Un estudio jurídico necesita consultar normativa legal específica de forma rápida y fundamentada. Los modelos de lenguaje tradicionales (LLM) no conocen documentos internos ni leyes locales chilenas, ya que su conocimiento se limita a los datos de entrenamiento. Si se les pregunta por la Ley 19.628 o por reglamentos internos del estudio, alucinan o dan respuestas genéricas sin sustento.

La solución implementada utiliza **RAG** (*Retrieval Augmented Generation*): antes de generar una respuesta, el sistema recupera fragmentos relevantes desde un corpus documental propio —leyes, normativas y reglamentos— y los entrega como contexto al LLM. El corpus incluye documentos en tres formatos: PDF (leyes oficiales de leychile.cl), TXT y Markdown.

---

## 2. Tecnologías utilizadas

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **Carga documental** | PyPDF, TextLoader (LangChain) | Interfaz unificada para PDF, TXT y MD |
| **Fragmentación** | RecursiveCharacterTextSplitter | Chunks de 500 chars con 100 de overlap; preserva artículos completos |
| **Embeddings** | nomic-embed-text (Ollama) | Modelo local de 768 dimensiones; no envía datos a servicios externos |
| **Base vectorial** | ChromaDB | Persistente, almacena metadata (fuente, página) junto al vector |
| **LLM** | Mistral 7B (Ollama) | ≤7B parámetros, corre local sin GPU, buen desempeño en español |
| **Interfaz** | Streamlit | Chat conversacional con componentes nativos |

---

## 3. Flujo de funcionamiento

```
1. INGESTA: PDF/TXT/MD → PyPDF/TextLoader → texto plano
2. CHUNKING: texto → RecursiveCharacterTextSplitter → 247 fragmentos
3. EMBEDDING: cada fragmento → nomic-embed-text → vector 768d
4. ALMACENAMIENTO: vectores + metadata → ChromaDB
5. CONSULTA: pregunta del usuario → embedding → búsqueda por similitud
6. RECUPERACIÓN: top-4 fragmentos filtrados por threshold 0.30
7. GENERACIÓN: prompt con contexto + pregunta → Mistral 7B → respuesta
8. TRAZABILIDAD: respuesta + fragmentos fuente + puntajes de similitud
```

El LLM recibe reglas estrictas en el prompt: descartar fragmentos de áreas legales no relacionadas con la pregunta, citar siempre la ley y el número de artículo, e ignorar fragmentos sin referencia legal explícita.

---

## 4. Capturas de pantalla

> [INSERTAR CAPTURAS AQUÍ]
>
> 1. Pantalla principal con la interfaz Streamlit (chat + preguntas de ejemplo)
> 2. Consulta simple respondida con fragmentos recuperados visibles
> 3. Consulta compleja con citas a múltiples artículos
> 4. Consulta sin respuesta documental (sistema indica que no hay información)

---

## 5. Dificultades encontradas

- **Ajuste del threshold de similitud**: un umbral muy alto (0.64) filtraba ruido entre leyes pero rompía consultas legítimas de consumidor. Un umbral muy bajo (0.30) dejaba pasar fragmentos de leyes no relacionadas. La solución fue mantener 0.30 y reforzar el prompt con reglas de área legal para que el LLM descarte contexto irrelevante.

- **Bloqueo de archivos en Windows**: al reindexar documentos, `shutil.rmtree` fallaba con error `WinError 32` porque ChromaDB mantenía locks sobre los archivos SQLite. Se resolvió usando la API de ChromaDB (`delete_collection`) en vez de borrar archivos directamente.

- **Cruce semántico entre leyes**: el embedding `nomic-embed-text` es un modelo genérico. Las palabras "derechos del consumidor" y "derechos del titular de datos" son semánticamente cercanas, provocando que fragmentos de la Ley 19.628 aparecieran en consultas de la Ley 19.496. Se mitigó con reglas explícitas en el prompt del LLM.

- **Procesamiento de PDFs extensos**: las leyes oficiales de leychile.cl tienen 56-57 páginas con preámbulos y disposiciones transitorias. Indexarlas completas saturaba el embedding (886 chunks). Se limitó la carga a 15 páginas por PDF, suficientes para cubrir el articulado principal.

---

## 6. Reflexión sobre ventajas y limitaciones de RAG

### Ventajas

- **Trazabilidad**: cada afirmación del asistente está respaldada por un fragmento documental recuperado, con su fuente y puntaje de similitud visibles para el usuario.
- **Conocimiento actualizable**: agregar un documento nuevo al corpus solo requiere reindexar; no es necesario reentrenar el modelo.
- **Privacidad**: toda la pipeline corre localmente con Ollama. Los documentos legales del estudio nunca salen a servidores externos.
- **Control de alucinaciones**: el prompt instruye al LLM a responder únicamente con el contexto proporcionado. Si no hay información suficiente, el sistema lo indica explícitamente en vez de inventar.

### Limitaciones

- **Calidad del embedding**: un modelo genérico como `nomic-embed-text` no distingue matices entre dominios legales cercanos. Un embedding especializado en texto jurídico mejoraría la precisión de la recuperación.
- **Chunking estático**: el tamaño fijo de 500 caracteres puede cortar artículos largos o agrupar varios cortos en un mismo fragmento, afectando la granularidad de la búsqueda.
- **Latencia**: la generación de embeddings para 247 fragmentos toma varios segundos. En un corpus más grande, este tiempo crece linealmente.
- **Dependencia del prompt**: la calidad de la respuesta depende críticamente de las reglas escritas en el prompt. Cambios en la redacción de la pregunta pueden producir resultados inconsistentes.
- **Sin memoria conversacional**: el sistema trata cada consulta de forma independiente. No mantiene contexto entre preguntas consecutivas del usuario.
