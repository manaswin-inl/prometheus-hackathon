# syntax=docker/dockerfile:1
# The base image bundles Python 3.12, prometheus_api, and all agent
# infrastructure (a2a, starlette, uvicorn).  Its hash is pinned in the
# platform's Makefile / your .env and passed in as a build argument — you never
# edit the FROM line directly; you change PROMETHEUS_API_BASE_IMAGE_HASH.
ARG PROMETHEUS_API_BASE_IMAGE_REPO=containers.github.inl.gov/genesis/prometheus_api
ARG PROMETHEUS_API_BASE_IMAGE_HASH
FROM ${PROMETHEUS_API_BASE_IMAGE_REPO}:${PROMETHEUS_API_BASE_IMAGE_HASH}

# Accept corporate proxy settings as build arguments (optional).
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

WORKDIR /app

# Tell uv (the package manager) to install into the system Python — where
# prometheus_api already lives — rather than creating a separate virtualenv.
ENV UV_PROJECT_ENVIRONMENT="/usr/local" \
    UV_LINK_MODE=copy

# Copy the dependency manifest first so Docker can cache this layer.
# If pyproject.toml hasn't changed, the next RUN is skipped on rebuild.
COPY pyproject.toml /app/

# Install your agent's extra dependencies (not prometheus_api itself — it's in
# the base image).  The cache mount speeds up repeated builds on the same host.
RUN --mount=type=cache,id=prometheus-uv-cache,target=/root/.cache/uv,sharing=locked \
    uv sync --no-dev --no-install-project --inexact

# Copy agent source last so code changes don't invalidate the dependency cache.
COPY agent.py /app/agent.py
COPY agent_card.json /app/agent_card.json
COPY prompts /app/prompts
COPY workflows /app/workflows

# Port is configured via the AGENT_PORT environment variable (default 8000).
EXPOSE ${AGENT_PORT}

CMD ["python", "agent.py"]
