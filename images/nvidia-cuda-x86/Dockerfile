# Common device tools. Phase Dockerfiles append their own user/workspace tail.
FROM ghcr.io/astral-sh/uv:0.12.17 AS uv
FROM golang:1.26 AS go
FROM alpine:3.23 AS certificates
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS toolchain
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PYTHON_PREFERENCE=only-managed UV_NO_CACHE=1 \
    PATH=/opt/python/bin:/usr/local/go/bin:/usr/local/bin:$PATH
# Bootstrap HTTPS before the Ubuntu ca-certificates package is installed.
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git cmake ninja-build build-essential pkg-config \
        libssl-dev libcurl4-openssl-dev libhwloc-dev libnuma-dev python3 \
        procps iproute2 jq less file tar unzip xz-utils nodejs npm \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=go /usr/local/go /usr/local/go
RUN uv python install 3.12 && chmod -R a+rX,go-w /opt/uv-python \
    && chmod 0700 /root \
    && install -d -o root -g root -m 0700 /root/tmp /root/.cache
ENTRYPOINT []
CMD ["sleep", "infinity"]
