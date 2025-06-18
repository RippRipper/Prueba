#!/bin/bash
set -e  # Detener en caso de error

# Instalar Python 3.9 si es necesario
pyenv install 3.9.16 -s
pyenv global 3.9.16

# Configurar entorno
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
