#!/bin/bash
set -e  # Detener en caso de error

# Instalar dependencias con pip
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
