FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt /app/
COPY src /app/src
COPY streamlit_app.py /app/streamlit_app.py

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "pif_research_platform.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
