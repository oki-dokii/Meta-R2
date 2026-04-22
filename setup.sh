#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' was not found" >&2
    exit 1
  fi
}

require_command "${PYTHON_BIN}"

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("error: LifeStack requires Python 3.9 or newer")
PY

echo "==> Using Python ${PYTHON_VERSION}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "==> Upgrading packaging tools"
"${VENV_PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${VENV_PIP}" install --upgrade pip setuptools wheel

echo "==> Installing project dependencies"
"${VENV_PIP}" install -r "${ROOT_DIR}/requirements.txt"

echo "==> Verifying core runtime imports"
"${VENV_PYTHON}" - <<'PY'
import uvicorn
import openenv
print(f"uvicorn ok: {uvicorn.__version__}")
print(f"openenv ok: {getattr(openenv, '__file__', 'module import succeeded')}")
PY

if [[ ! -f "${ROOT_DIR}/.env.example" ]]; then
  cat > "${ROOT_DIR}/.env.example" <<'EOF'
GROQ_API_KEY=your_groq_api_key_here
# Optional: path to your Google OAuth desktop client credentials JSON for Gmail intake
# GOOGLE_CLIENT_SECRET_FILE=/absolute/path/to/client_secret.json
EOF
fi

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "==> Created .env from .env.example"
fi

echo "==> Running smoke test"
"${VENV_PYTHON}" "${ROOT_DIR}/scripts/test_lifestack.py"

cat <<EOF

LifeStack is set up.

Activate the environment:
  source "${VENV_DIR}/bin/activate"

Run the app:
  ./run-app.sh
  # or:
  "${VENV_PYTHON}" "${ROOT_DIR}/app.py"

Run the OpenEnv server:
  ./run-server.sh
  # or:
  "${VENV_PYTHON}" "${ROOT_DIR}/server.py"

If you want live Groq-powered actions, set GROQ_API_KEY in:
  ${ROOT_DIR}/.env
EOF
