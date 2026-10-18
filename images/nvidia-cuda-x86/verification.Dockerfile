# Appended to the common Dockerfile before the provider builds from source.
FROM toolchain AS verification
RUN apt-get update && apt-get install -y --no-install-recommends libseccomp2 \
    && rm -rf /var/lib/apt/lists/*
RUN install -d -o root -g root -m 0755 /workspace /workspace/.harbor /workspace/.harbor/scratch \
    && groupadd --gid 1004 verifier \
    && useradd --uid 1004 --gid verifier --home-dir /workspace/.harbor/scratch/verifier --create-home --shell /bin/bash verifier \
    && groupadd --gid 1003 judge \
    && useradd --uid 1003 --gid judge --home-dir /workspace/.harbor/scratch/judge --create-home --shell /bin/bash judge \
    && groupadd --gid 1005 reference \
    && useradd --uid 1005 --gid reference --home-dir /workspace/.harbor/scratch/reference --create-home --shell /bin/bash reference \
    && chmod 0700 /workspace/.harbor/scratch/* \
    && chown root:verifier /workspace && chmod 1775 /workspace \
    && install -d -o root -g root -m 0700 /var/lib/local-eval /var/lib/local-eval/verification \
    && install -d -o root -g root -m 0755 /inputs /models /models/target /models/tokenizer \
    && install -d -o judge -g root -m 2750 /workspace/output/review \
    && install -d -o root -g root -m 0755 /workspace/.harbor/review-rubric \
    && install -d -o root -g judge -m 0770 /logs/verifier
WORKDIR /workspace
# Trusted startup needs root; native task.toml selects the workload user.
USER root
