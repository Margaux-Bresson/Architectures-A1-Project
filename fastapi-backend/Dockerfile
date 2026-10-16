FROM ghcr.io/astral-sh/uv:python3.12-trixie

WORKDIR /app

# Utilise le Python de l'image (pas de téléchargement), compile le bytecode
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Dépendances d'abord : cette couche est mise en cache tant que uv.lock ne change pas
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Code de l'application
COPY *.py ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

EXPOSE 8000

ENTRYPOINT []
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]