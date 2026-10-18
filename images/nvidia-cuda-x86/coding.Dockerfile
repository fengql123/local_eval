# Appended to the common Dockerfile before the provider builds from source.
FROM toolchain AS coding
RUN groupadd --gid 1001 agent \
    && install -d -o root -g root -m 0755 /workspace /workspace/.harbor \
    && useradd --uid 1001 --gid agent --home-dir /workspace/.harbor/agent --create-home --shell /bin/bash agent \
    && chmod 0700 /workspace/.harbor/agent \
    && chown root:agent /workspace && chmod 1775 /workspace \
    && install -d -o root -g agent -m 0750 /var/lib/local-eval \
    && install -d -o root -g root -m 0755 /inputs /models /models/target /models/tokenizer
WORKDIR /workspace
# Trusted startup needs root; native task.toml selects the workload user.
USER root
