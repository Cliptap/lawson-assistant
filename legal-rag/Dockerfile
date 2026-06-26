FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    chromadb \
    langchain \
    langchain-chroma \
    langchain-community \
    langchain-ollama \
    pypdf \
    streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
