# Runtime image of a released Eris version, installed from PyPI (no source copied), so any past
# release can be rebuilt. Built and pushed by .github/workflows/publish-container.yml.
#   docker build --build-arg ERIS_VERSION=0.2.0 -t eris:0.2.0 .
#   docker build --build-arg ERIS_VERSION=0.2.0 --build-arg EXTRAS=[gearnet,pyg] -t eris:0.2.0-full .
# 3.13: biotraj has no cp314 wheel, so 3.14 would need a compiler.
FROM python:3.13-slim

ARG ERIS_VERSION
ARG EXTRAS=""
LABEL org.opencontainers.image.source="https://github.com/kren-ai-lab/eris" \
      org.opencontainers.image.description="Eris: protein structures to numerical representations and GearNet-Edge embeddings" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${ERIS_VERSION}"

# procps: Nextflow reads task metrics with `ps` inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${ERIS_VERSION}" \
    && pip install --no-cache-dir "erislib${EXTRAS}==${ERIS_VERSION}"
