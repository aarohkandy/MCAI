# MCAI headless training image.
# Contains ONLY toolchain + Python deps. The repo is bind-mounted at /app and all
# Minecraft/Eaglercraft binaries are fetched at runtime into the /data volume, so this
# image redistributes no game code or assets.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# JDK 17 runs BOTH Paper 1.12.2 and EaglerXServer in a single JVM.
# Conventional wisdom says 1.12.2 needs Java 8, so this was verified directly rather than assumed:
# Paper 1.12.2 build 1620 with the MCAIArena plugin boots on Temurin 17 in ~8s, control socket up,
# no exceptions. That matters because EaglerXServer REQUIRES Java 17+ and its native Spigot API
# target is Paper 1.12.2 - so one JVM serves both, with no BungeeCord/Velocity proxy needed.
# Do not "fix" this down to Java 8/11: that would break EaglerXServer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless maven git curl ca-certificates procps xz-utils gnupg \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Bake the heavy, stable trainer dependencies (CPU-only PyTorch — no CUDA needed for a 296k model).
# Versions mirror trainer/pyproject.toml. The local combat_ai package is installed --no-deps at
# container start (entrypoint) so code edits take effect on a plain restart.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<2.8" \
 && pip install \
        "numpy>=1.26,<3" "msgpack>=1.0.8,<2" "websockets>=12,<16" \
        "pytest>=8,<9" "pytest-asyncio>=0.23,<1"

ENV MCAI_RUNTIME=/data/server-runtime \
    MCAI_CHECKPOINT_DIR=/data/checkpoints \
    MCAI_RUN_DIR_BASE=/data/runs \
    MCAI_VENV_PYTHON=python3 \
    MCAI_BOT_COUNT=4 \
    MCAI_MODE=sword \
    MCAI_JAVA_MEMORY=3G

WORKDIR /app
ENTRYPOINT ["/app/deploy/azure/entrypoint.sh"]
