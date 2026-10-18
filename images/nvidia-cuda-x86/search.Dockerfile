# Appended to the common Dockerfile before the provider builds from source.
FROM toolchain AS search
RUN groupadd --gid 1002 search \
    && useradd --uid 1002 --gid search --create-home --shell /bin/bash search \
    && chmod 0700 /home/search \
    && install -d -o root -g search -m 0750 /var/lib/local-eval \
    && install -d -o root -g root -m 0700 /var/lib/local-eval/reference \
    && install -d -o root -g root -m 0755 /workspace /workspace/data \
    && install -d -o search -g search -m 0755 /workspace/baselines /workspace/bench \
        /workspace/output /workspace/runtime/search \
    && install -d -o search -g search -m 0700 /home/search/tmp /home/search/.cache
WORKDIR /workspace
# Trusted startup needs root; native task.toml selects the workload user.
USER root
