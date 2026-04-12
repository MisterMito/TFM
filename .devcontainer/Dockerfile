FROM mcr.microsoft.com/devcontainers/python:3.12
RUN rm -f /etc/apt/sources.list.d/yarn.list /usr/share/keyrings/yarnkey.gpg 2>/dev/null || true
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*
