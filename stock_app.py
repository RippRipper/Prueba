import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Stock Forecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Descargar recursos de NLP (con verificación)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Estilos CSS personalizados
st.markdown("""
    <style>
    /* ... (los estilos se mantienen igual) ... */
    </style>
""", unsafe_allow_html=True)

# Función para calcular RSI (igual)

# Función para obtener datos bursátiles (con manejo de error mejorado)
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            # Intentar con un período más largo
            data = yf.download(ticker, period="max")
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return pd.DataFrame()

# Función para crear características técnicas (igual)

# Función para entrenar el modelo (igual)

# Función para obtener noticias financieras (con selectores actualizados)
def get_financial_news(ticker):
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_items = []
        
        # Buscar noticias en el nuevo formato
        news_containers = soup.find_all('div', class_='article__content', limit=5)
        if not news_containers:
            news_containers = soup.find_all('li', class_='article', limit=5)
        
        for container in news_containers:
            try:
                headline_elem = container.find('a', class_='link')
                if not headline_elem:
                    headline_elem = container.find('h3', class_='article__headline')
                    if headline_elem:
                        headline_elem = headline_elem.find('a')
                
                if headline_elem:
                    headline = headline_elem.text.strip()
                    link = headline_elem['href']
                    if not link.startswith('http'):
                        link = 'https://www.marketwatch.com' + link
                else:
                    continue
                
                time_elem = container.find('span', class_='article__timestamp')
                if not time_elem:
                    time_elem = container.find('span', class_='article__details')
                time_text = time_elem.text.strip() if time_elem else 'Hace un momento'
                
                news_items.append({
                    'headline': headline,
                    'link': link,
                    'time': time_text
                })
            except Exception as e:
                continue
        
        return news_items
    
    except Exception as e:
        st.warning(f"No se pudieron obtener noticias: {str(e)}")
        return []

# Función para analizar sentimiento de noticias (igual)

# Función para generar recomendación (igual)

# Interfaz principal de la aplicación
def main():
    # ... (código anterior de la interfaz) ...

    # Obtener datos y entrenar modelo
    with st.spinner('Obteniendo datos y entrenando modelo...'):
        try:
            # ... (código anterior) ...
        except Exception as e:
            st.error(f"Error crítico: {str(e)}")
            st.stop()  # Detener la ejecución de la app si hay un error grave

# ... (resto del código) ...
