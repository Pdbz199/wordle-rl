FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/webapp

COPY webapp/requirements-runtime.txt /tmp/requirements-runtime.txt
COPY webapp/requirements-web.txt /tmp/requirements-web.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt -r /tmp/requirements-web.txt

COPY webapp /opt/webapp

ENV PYTHONPATH=/opt/webapp:/opt/wordle

CMD ["python", "-m", "worker.worker"]
