FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai07_chunker.py .
COPY ai08_vector_retrieval.py .
COPY ai10_ai11_rag_prompt.py .
COPY ai12_ai13_agent_tool.py .
COPY ai14_agent_api.py .

EXPOSE 8000

CMD ["uvicorn", "ai14_agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
