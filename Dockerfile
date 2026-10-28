FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY onboarding ./onboarding
RUN pip install --no-cache-dir '.[test]' && useradd --create-home --uid 10001 agent
COPY tests ./tests
USER agent
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
