FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src /app/src
COPY ./templates /app/templates
COPY ./data/processed/test.csv /app/data/processed/test.csv


WORKDIR /app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]