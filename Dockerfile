FROM python:3.11-slim

WORKDIR /app

COPY App/requirements-docker.txt ./requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "src/agent/agent.py"]