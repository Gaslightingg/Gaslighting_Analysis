FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .[dev]
COPY . /app
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.bot.main"]
