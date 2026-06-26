Evaluaci¾n Sumativa

Laboratorio Evaluado: Implementaci¾n de un Asistente RAG

Asignatura: IA Embebida en Sistemas Computacionales
Carrera: IngenierÝa Informßtica
Modalidad: Equipos de 2 a 3 estudiantes
Presentaci¾n: Clase siguiente (5 a 7 minutos por equipo)

1. Contexto

Los modelos de lenguaje (LLM) son capaces de responder preguntas complejas, pero
presentan una limitaci¾n importante: su conocimiento depende de la informaci¾n
disponible durante su entrenamiento.

Para resolver este problema, las organizaciones modernas utilizan arquitecturas RAG
(Retrieval Augmented Generation), las cuales permiten recuperar informaci¾n
desde documentos propios antes de generar una respuesta.

En esta evaluaci¾n deberßn implementar un asistente basado en RAG capaz de
responder preguntas utilizando documentos especÝficos del dominio seleccionado.

El objetivo no es construir una aplicaci¾n compleja, sino demostrar comprensi¾n
prßctica de los siguientes conceptos:

ò  Embeddings.
ò  B·squeda semßntica.
ò  Bases vectoriales.
ò  Recuperaci¾n de contexto.
ò

Integraci¾n con modelos de lenguaje.

2. Objetivo de Aprendizaje

Implementar una soluci¾n funcional basada en RAG capaz de recuperar informaci¾n
relevante desde un conjunto de documentos y utilizar dicha informaci¾n para generar
respuestas fundamentadas mediante un modelo de lenguaje.

3. Arquitectura Base

Todos los equipos deberßn implementar la siguiente arquitectura:

Usuario
?
Pregunta
?
Embedding
?
Base Vectorial
?
Recuperaci¾n de Contexto
?
LLM
?
Respuesta + Fuentes

No se evaluarß el dise±o arquitect¾nico, ya que esta arquitectura serß com·n para
todos los equipos.

La evaluaci¾n se centrarß en la correcta implementaci¾n y funcionamiento de cada
componente.

4. Casos de Uso Disponibles

Cada equipo deberß seleccionar uno de los siguientes escenarios.

Caso 1: Asistente AcadÚmico Universitario

Descripci¾n

La universidad desea implementar un asistente capaz de responder consultas
frecuentes de estudiantes utilizando documentaci¾n institucional.

Corpus documental sugerido
ò  Reglamento de asistencia.
ò  Reglamento de evaluaciones.
ò  Reglamento de prßcticas.
ò  Reglamento de titulaci¾n.
ò  Calendario acadÚmico.

Preguntas de ejemplo

ò  ┐Cußl es el porcentaje mÝnimo de asistencia?
ò  ┐QuÚ ocurre si repruebo una asignatura?
ò  ┐Cußles son los requisitos para titularme?
ò  ┐Cußndo finaliza el semestre?

Resultado esperado

El sistema debe responder utilizando ·nicamente informaci¾n recuperada desde los
documentos institucionales.

Caso 2: Asistente de Soporte TÚcnico

Descripci¾n

Una empresa tecnol¾gica desea implementar un asistente capaz de ayudar a
usuarios y tÚcnicos utilizando documentaci¾n interna.

Corpus documental sugerido

ò  Manual Docker.
ò  Manual Linux.
ò  Manual Git.
ò  Procedimientos de soporte.
ò  Preguntas frecuentes.

Preguntas de ejemplo

ò  ┐C¾mo crear una imagen Docker?
ò  ┐C¾mo actualizar un repositorio Git?
ò  ┐C¾mo visualizar procesos en Linux?
ò  ┐C¾mo abrir un ticket de soporte?

Resultado esperado

Las respuestas deben fundamentarse en la documentaci¾n tÚcnica disponible.

Caso 3: Asistente Legal

Descripci¾n

Un estudio jurÝdico desea construir un asistente que permita consultar normativa
especÝfica.

Corpus documental sugerido

ò  Ley de Protecci¾n de Datos Personales.
ò  Ley del Consumidor.
ò  Normativas laborales.
ò  Reglamentos internos.

Preguntas de ejemplo

ò  ┐QuÚ establece la Ley 21.719?
ò  ┐Cußles son los derechos del consumidor?
ò  ┐QuÚ obligaciones tiene un empleador respecto a los datos personales?

Resultado esperado

El sistema debe citar correctamente los documentos utilizados para responder.

Caso 4: Asistente MÚdico

Descripci¾n

Un centro mÚdico requiere consultar protocolos clÝnicos y procedimientos internos.

Corpus documental sugerido

ò  Protocolos clÝnicos.
ò  Procedimientos mÚdicos.
ò  GuÝas de atenci¾n.
ò  Manuales de operaci¾n.

Preguntas de ejemplo

ò  ┐Cußl es el protocolo para atenci¾n inicial?
ò  ┐QuÚ exßmenes deben realizarse?
ò  ┐Cußles son los pasos del procedimiento?

Resultado esperado

Las respuestas deben estar sustentadas por los documentos proporcionados.

5. Requisitos TÚcnicos Obligatorios

La soluci¾n deberß incluir obligatoriamente los siguientes componentes.

5.1 Ingesta Documental

El sistema debe cargar documentos desde archivos.

Formatos permitidos:

ò  PDF
ò  TXT
ò  Markdown (.md)

5.2 Fragmentaci¾n (Chunking)

Los documentos deberßn dividirse en fragmentos adecuados para la recuperaci¾n
semßntica.

El equipo deberß justificar brevemente:

ò  Tama±o de chunk utilizado.
ò  Solapamiento (overlap) utilizado.

5.3 Embeddings

La soluci¾n deberß generar embeddings para cada fragmento documental.

TecnologÝas sugeridas:

ò  Ollama Embeddings
ò  Nomic Embed
ò  Sentence Transformers

5.4 Base Vectorial

Debe utilizarse una base vectorial para almacenar los embeddings.

Opciones sugeridas:

ò  ChromaDB
ò  FAISS

5.5 Recuperaci¾n de Informaci¾n

La consulta del usuario debe generar un embedding y recuperar los fragmentos mßs
relevantes.

El sistema deberß mostrar al menos uno de los siguientes elementos:

ò  Fragmentos recuperados.
ò  Fuente documental utilizada.

5.6 Generaci¾n de Respuesta

La respuesta deberß ser generada por un modelo de lenguaje utilizando:

Pregunta del usuario + Contexto recuperado

Opciones sugeridas, no mayor a 7B:

ò  Llama 3
ò  Gemma
ò  Mistral

6. Funcionalidades MÝnimas

La aplicaci¾n deberß permitir:

? Realizar preguntas.

? Recuperar documentos relevantes.

? Generar respuestas utilizando el contexto recuperado.

? Mostrar las fuentes utilizadas.

? Manejar preguntas cuya respuesta no exista en los documentos.

7. Demostraci¾n Obligatoria

Durante la presentaci¾n cada equipo deberß demostrar los siguientes escenarios.

Escenario 1: Consulta Simple

Ejemplo:

┐Cußl es la asistencia mÝnima requerida?

Escenario 2: Consulta Compleja

Ejemplo:

┐QuÚ requisitos debo cumplir para titularme y quÚ sucede si no los cumplo?

Escenario 3: Consulta Sin Respuesta Documental

Ejemplo:

Una pregunta cuya respuesta no exista dentro del corpus documental.

El sistema deberß reconocer la ausencia de informaci¾n o indicar incertidumbre de
manera razonable.

8. Entregables

Cada equipo deberß entregar lo siguiente.

Repositorio GitHub

Debe contener:

ò  C¾digo fuente.
ò
Instrucciones de ejecuci¾n.
ò  Dependencias necesarias.
ò  Archivos de configuraci¾n requeridos para ejecutar la soluci¾n.

Documento Breve (Mßximo 3 Pßginas)

Debe incluir:

Descripci¾n del problema abordado.

TecnologÝas utilizadas.

Flujo de funcionamiento de la soluci¾n.

Capturas de pantalla.

Dificultades encontradas.

Reflexi¾n sobre ventajas y limitaciones de RAG. (No IA)

9. Presentaci¾n

Duraci¾n mßxima:

7 minutos por equipo

Distribuci¾n sugerida:

ò  Problema abordado (1 minuto).
ò  Arquitectura implementada (2 minutos).
ò  Demostraci¾n funcional (3 minutos).
ò  Reflexi¾n y conclusiones (1 minuto).
ò  Video grabado subido en canvas.

10. R·brica
Criterio

Carga e indexaci¾n documental

Generaci¾n de embeddings

Implementaci¾n de recuperaci¾n semßntica

Integraci¾n con LLM

Calidad de las respuestas generadas

Presentaci¾n de fuentes recuperadas

Puntaje

15

15

20

15

10

10

Manejo de preguntas fuera del contexto documental  5

Calidad de la demostraci¾n

Total

10

100

Consideraciones Finales

Esta evaluaci¾n busca demostrar la comprensi¾n prßctica del paradigma RAG y su
integraci¾n dentro de una arquitectura moderna de inteligencia artificial.

Se evaluarß principalmente:

ò  Comprensi¾n tÚcnica.
ò  Correcta implementaci¾n.
ò  Capacidad de explicar el flujo de recuperaci¾n y generaci¾n de informaci¾n.

No se evaluarß el dise±o visual avanzado ni funcionalidades adicionales fuera del
alcance definido para este laboratorio.


