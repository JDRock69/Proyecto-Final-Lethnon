#!/usr/bin/env bash
set -e
source .venv/bin/activate
python validar_datos.py
python app.py
