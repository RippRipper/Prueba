
import sys
st.write("Versión de Python:", sys.version)
st.write("Dependencias instaladas:")
dependencies = ['yfinance', 'pandas', 'numpy', 'plotly', 'scikit-learn', 'nltk', 'beautifulsoup4', 'requests', 'lxml', 'html5lib']
for dep in dependencies:
    try:
        module = __import__(dep)
        st.write(f"{dep}: {module.__version__}")
    except ImportError:
        st.error(f"{dep} NO INSTALADO")
    except AttributeError:
        st.write(f"{dep}: versión no disponible")


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
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

# Descargar recursos de NLP con manejo de errores
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #1F77B4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stSelectbox, .stTextInput, .stDateInput {
        background-color: #192841;
        border-radius: 5px;
    }
    .stAlert {
        border-left: 4px solid #FF4B4B;
        padding: 1rem;
        background-color: #1A1F2C;
    }
    .positive {
        color: #00CC96;
    }
    .negative {
        color: #EF553B;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1F77B4, #FF7F0E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2A4B8D;
    }
    .feature-box {
        background-color: #1A1F2C;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ticker-card {
        background-color: #1A2A4C;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Función para calcular RSI con manejo de división por cero
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calcular ganancia y pérdida promedio
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Evitar división por cero
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Evita división por cero
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función para obtener datos bursátiles con reintentos
def get_stock_data(ticker, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"Intento {attempt+1}/{retries} fallido para {ticker}. Reintentando...")
            if attempt == retries - 1:
                st.error(f"No se pudieron obtener datos para {ticker}: {str(e)}")
                return pd.DataFrame()
    
    return pd.DataFrame()

# Función para crear características técnicas con manejo de errores
def create_features(data):
    if data.empty:
        return pd.DataFrame()
    
    try:
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
        data['RSI'] = compute_rsi(data['Close'])
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(30).std() * np.sqrt(252)
        
        # Crear variable objetivo
        data['Tomorrow'] = data['Close'].shift(-1)
        data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
        
        return data.dropna()
    
    except Exception as e:
        st.error(f"Error creando características: {str(e)}")
        return pd.DataFrame()

# Función para entrenar el modelo con verificación de datos
def train_model(data):
    if data.empty or len(data) < 100:
        st.warning("Datos insuficientes para entrenar el modelo")
        return None, 0.0, []
    
    features = ['MA20', 'MA50', 'MA200', 'RSI', 'Volatility']
    target = 'Target'
    
    # Verificar que las características existan
    for feature in features:
        if feature not in data.columns:
            st.error(f"Característica faltante: {feature}")
            return None, 0.0, []
    
    # Dividir datos
    train = data.iloc[:-30]
    test = data.iloc[-30:]
    
    # Verificar que hay suficientes datos para entrenamiento y prueba
    if len(train) < 50 or len(test) < 10:
        st.warning("Datos insuficientes para división entrenamiento/prueba")
        return None, 0.0, []
    
    try:
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=50)
        model.fit(train[features], train[target])
        
        # Evaluar modelo
        accuracy = model.score(test[features], test[target])
        
        return model, accuracy, features
    
    except Exception as e:
        st.error(f"Error entrenando modelo: {str(e)}")
        return None, 0.0, []

# Función CORREGIDA para obtener noticias financieras
def get_financial_news(ticker):
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_items = []
        
        # Primero intentar con el nuevo formato
        articles = soup.find_all('div', class_='article__content', limit=5)
        
        # Si no se encuentran, intentar con el formato antiguo
        if not articles:
            articles = soup.find_all('li', class_='article', limit=5)
        
        for article in articles:
            try:
                # Intentar diferentes formas de obtener el encabezado
                headline_elem = article.find('a', class_='link')
                if not headline_elem:
                    headline_elem = article.find('h3', class_='article__headline')
                    if headline_elem:
                        headline_elem = headline_elem.find('a')
                
                if headline_elem:
                    headline = headline_elem.text.strip()
                    link = headline_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = 'https://www.marketwatch.com' + link
                else:
                    continue
                
                # Intentar diferentes formas de obtener la fecha
                time_elem = article.find('span', class_='article__timestamp')
                if not time_elem:
                    time_elem = article.find('span', class_='article__details')
                time_text = time_elem.text.strip() if time_elem else 'Reciente'
                
                news_items.append({
                    'headline': headline,
                    'link': link,
                    'time': time_text
                })
            except Exception as e:
                # Manejo interno de errores
                continue
        
        return news_items if news_items else []
    
    except Exception as e:
        st.warning(f"No se pudieron obtener noticias: {str(e)}")
        return []

# Función para analizar sentimiento de noticias con manejo de errores
def analyze_sentiment(news):
    if not news:
        return [], 0.0
    
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    analyzed_news = []
    
    for item in news:
        try:
            score = sia.polarity_scores(item['headline'])
            sentiment_scores.append(score['compound'])
            item['sentiment'] = score['compound']
            analyzed_news.append(item)
        except:
            continue
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    
    return analyzed_news, avg_sentiment

# Función para generar recomendación con valores predeterminados seguros
def generate_recommendation(prediction, probability, sentiment, accuracy):
    if prediction is None or probability is None:
        return "DATOS INSUFICIENTES", "⚪"
    
    try:
        if prediction == 1 and probability > 0.65 and sentiment > 0.1:
            return "FUERTE COMPRA", "🟢"
        elif prediction == 1 and probability > 0.55:
            return "COMPRA MODERADA", "🟡"
        elif prediction == 0 and probability > 0.65 and sentiment < -0.1:
            return "FUERTE VENTA", "🔴"
        elif prediction == 0 and probability > 0.55:
            return "VENTA MODERADA", "🟠"
        else:
            return "MANTENER", "⚪"
    except:
        return "ERROR EN ANÁLISIS", "⚪"

# Interfaz principal de la aplicación con mejor manejo de errores
def main():
    # Cabecera
    st.markdown('<div class="header">Stock Forecast Pro</div>', unsafe_allow_html=True)
    st.caption("Herramienta avanzada de análisis bursátil basada en IA - Datos en tiempo real")
    
    # Barra lateral para parámetros
    with st.sidebar:
        st.header("⚙️ Configuración")
        ticker = st.text_input("Símbolo bursátil (ej: AAPL, MSFT, TSLA)", "AAPL").upper()
        
        # Fechas por defecto
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)  # 5 años de datos
        
        # Selección de mercado para empresas internacionales
        market = st.selectbox("Mercado", ["NYSE", "NASDAQ", "EURONEXT", "TSE", "LSE", "FWB"])
        
        st.subheader("Parámetros del Modelo")
        st.info("El modelo utiliza Random Forest con 200 estimadores")
        st.warning("⚠️ Precisión varía según mercado y condiciones")
        
        st.subheader("Advertencia Legal")
        st.error("""
        **Esto NO es asesoramiento financiero.**  
        Las inversiones implican riesgo de pérdida.  
        Esta aplicación es para fines educativos solamente.  
        Consulte con un profesional antes de invertir.
        """)
    
    # Obtener datos y entrenar modelo
    with st.spinner('Obteniendo datos y entrenando modelo...'):
        try:
            # Usar ticker completo para mercados específicos
            full_ticker = f"{ticker}"
            if market == "EURONEXT":
                full_ticker += ".PA"
            elif market == "TSE":
                full_ticker += ".T"
            elif market == "LSE":
                full_ticker += ".L"
            elif market == "FWB":
                full_ticker += ".DE"
            
            data = get_stock_data(full_ticker, start_date, end_date)
            
            if data.empty:
                # Intentar sin extensión de mercado
                data = get_stock_data(ticker, start_date, end_date)
                if data.empty:
                    st.error(f"No se encontraron datos para {ticker} en {market}")
                    return
            
            processed_data = create_features(data)
            
            if processed_data.empty:
                st.error("Error procesando datos. Intente con otro símbolo.")
                return
            
            model, accuracy, features = train_model(processed_data)
            
            # Obtener noticias y analizar sentimiento
            news = get_financial_news(ticker)
            news, avg_sentiment = analyze_sentiment(news)
            
            # Preparar datos para predicción si el modelo está disponible
            if model and features:
                latest_data = processed_data.iloc[-1][features].values.reshape(1, -1)
                
                # Hacer predicción
                prediction = model.predict(latest_data)[0]
                probability = model.predict_proba(latest_data)[0][prediction]
            else:
                prediction = None
                probability = None
            
            # Generar recomendación
            recommendation, icon = generate_recommendation(prediction, probability, avg_sentiment, accuracy)
            
        except Exception as e:
            st.error(f"Error inesperado: {str(e)}")
            st.info("Intente recargar la página o probar con otro símbolo bursátil")
            return
    
    # Layout principal
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"📊 Análisis de {ticker}")
        
        # Precio actual con verificación
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100 if prev_close != 0 else 0
            
            st.metric("Precio Actual", f"${current_price:.2f}", 
                     f"{price_change:.2f} ({percent_change:.2f}%)", 
                     delta_color="inverse" if price_change < 0 else "normal")
        else:
            st.warning("Datos de precio no disponibles")
        
        # Rendimiento clave
        st.subheader("Indicadores Clave")
        
        col1a, col2a = st.columns(2)
        with col1a:
            if not processed_data.empty and 'RSI' in processed_data.columns:
                st.metric("RSI", f"{processed_data['RSI'].iloc[-1]:.2f}", 
                         help="RSI > 70: Sobrecompra, RSI < 30: Sobreventa")
            else:
                st.warning("RSI no disponible")
            
            if not processed_data.empty and 'Volatility' in processed_data.columns:
                st.metric("Volatilidad (30d)", f"{processed_data['Volatility'].iloc[-1]*100:.2f}%", 
                         help="Volatilidad anualizada")
            else:
                st.warning("Volatilidad no disponible")
            
        with col2a:
            if not processed_data.empty and 'MA20' in processed_data.columns and 'MA50' in processed_data.columns:
                st.metric("MA20/MA50", 
                         f"{'↑' if processed_data['MA20'].iloc[-1] > processed_data['MA50'].iloc[-1] else '↓'}",
                         help="MA20 sobre MA50: tendencia alcista")
            else:
                st.warning("Medias móviles no disponibles")
            
            if accuracy > 0:
                st.metric("Precisión Modelo", f"{accuracy*100:.2f}%", 
                         help="Precisión en los últimos 30 días")
            else:
                st.warning("Precisión no disponible")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"📉 Gráficos de Tendencias")
        
        # Crear gráfico de precios si hay datos
        if not data.empty and not processed_data.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Precio y medias móviles
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Precio'), row=1, col=1)
            
            if 'MA20' in processed_data.columns:
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA20'], 
                                       name='MA20', line=dict(color='orange')), row=1, col=1)
            
            if 'MA50' in processed_data.columns:
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA50'], 
                                       name='MA50', line=dict(color='green')), row=1, col=1)
            
            if 'MA200' in processed_data.columns:
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA200'], 
                                       name='MA200', line=dict(color='purple')), row=1, col=1)
            
            # RSI si está disponible
            if 'RSI' in processed_data.columns:
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['RSI'], 
                                      name='RSI', line=dict(color='blue')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Actualizar diseño
            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Datos insuficientes para generar gráficos")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"🧠 Recomendación de IA")
        
        # Mostrar recomendación
        st.markdown(f"### {icon} **{recommendation}**")
        
        # Detalles de la predicción
        st.subheader("Análisis Predictivo")
        
        if probability is not None:
            st.metric("Probabilidad Predicción", f"{probability*100:.2f}%", 
                     help="Confianza del modelo en la predicción")
        else:
            st.warning("Probabilidad no disponible")
        
        if prediction is not None:
            st.metric("Predicción", "ALZA" if prediction == 1 else "BAJA", 
                     delta="Positiva" if prediction == 1 else "Negativa", 
                     delta_color="normal" if prediction == 1 else "inverse")
        else:
            st.warning("Predicción no disponible")
        
        # Análisis de sentimiento
        if avg_sentiment != 0:
            sentiment_label = "Positivo" if avg_sentiment > 0.1 else "Negativo" if avg_sentiment < -0.1 else "Neutral"
            st.metric("Sentimiento Noticias", sentiment_label, 
                     f"{avg_sentiment:.2f}", 
                     delta_color="normal" if avg_sentiment > 0 else "inverse")
        else:
            st.warning("Sentimiento no disponible")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sección de noticias
    st.markdown("---")
    st.subheader(f"📰 Últimas Noticias de {ticker}")
    
    if news:
        for item in news:
            sentiment_class = "positive" if item.get('sentiment', 0) > 0.1 else "negative" if item.get('sentiment', 0) < -0.1 else ""
            
            with st.expander(f"{item.get('time', '')}: {item.get('headline', '')}"):
                if 'sentiment' in item:
                    st.markdown(f"**Sentimiento:** `{item['sentiment']:.2f}`")
                if item.get('link'):
                    st.markdown(f"[Leer más]({item['link']})")
    else:
        st.warning("No se encontraron noticias recientes para este símbolo.")
    
    # Sección de datos históricos
    st.markdown("---")
    st.subheader(f"📊 Datos Históricos de {ticker}")
    
    if not data.empty:
        # Mostrar últimos 10 días
        st.dataframe(data.tail(10).style.format({
            'Open': '{:.2f}', 'High': '{:.2f}', 
            'Low': '{:.2f}', 'Close': '{:.2f}', 'Adj Close': '{:.2f}', 'Volume': '{:,.0f}'
        }).background_gradient(cmap='Blues', subset=['Volume']))
        
        # Descarga de datos
        csv = data.to_csv().encode('utf-8')
        st.download_button(
            label="Descargar datos históricos (CSV)",
            data=csv,
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("Datos históricos no disponibles")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
