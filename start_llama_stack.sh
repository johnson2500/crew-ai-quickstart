# Start ollama in the background
ollama run llama3.2:3b --keepalive 60m &

# Sync project dependencies (installs everything from pyproject.toml)
uv sync

# Run llama stack with all dependencies available
OLLAMA_URL=http://localhost:11434 uv run llama stack run starter
