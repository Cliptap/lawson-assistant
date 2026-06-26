#!/bin/bash
# Script de configuracion inicial: descarga modelos y construye la base vectorial
# Ejecutar antes de iniciar la aplicacion

set -e

echo "=== Asistente Legal RAG - Configuracion Inicial ==="

echo "[1/3] Esperando a Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
  echo "  Esperando... (asegurate que Ollama este corriendo)"
  sleep 2
done
echo "  Ollama listo."

echo "[2/3] Descargando modelos..."
ollama pull nomic-embed-text
ollama pull mistral
echo "  Modelos listos."

echo "[3/3] Indexando documentos..."
python ingest.py
echo "  Base vectorial creada."

echo "=== Listo. Ejecuta: python app.py  o  streamlit run app.py ==="
