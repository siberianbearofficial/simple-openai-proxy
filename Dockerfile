FROM sunpeek/poetry:py3.12-slim

ARG ENVIRONMENT=production
ENV ENVIRONMENT=${ENVIRONMENT}

COPY pyproject.toml .
COPY poetry.lock .
COPY ./openai_proxy ./openai_proxy
COPY ./envs ./envs

RUN poetry install $(test "$ENVIRONMENT" != local && echo "--only main") --no-interaction --no-ansi
