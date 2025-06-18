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

# Descargar recursos de NLP
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

# Función para calcular RSI
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función para obtener datos bursátiles
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Función para crear características técnicas
def create_features(data):
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

# Función para entrenar el modelo
def train_model(data):
    features = ['MA20', 'MA50', 'MA200', 'RSI', 'Volatility']
    target = 'Target'
    
    # Dividir datos
    train = data.iloc[:-30]
    test = data.iloc[-30:]
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=50)
    model.fit(train[features], train[target])
    
    # Evaluar modelo
    accuracy = model.score(test[features], test[target])
    
    return model, accuracy, features

# Función para obtener noticias financieras
def get_financial_news(ticker):
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker}?mod=search_symbol"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_items = []
        articles = soup.find_all('div', class_='article__content', limit=5)
        
        for article in articles:
            headline = article.find('a', class_='link').text.strip()
            link = article.find('a', class_='link')['href']
            time = article.find('span', class_='article__timestamp').text.strip()
            
            news_items.append({
                'headline': headline,
                'link': link,
                'time': time
            })
        
        return news_items
    
    except Exception as e:
        st.warning(f"No se pudieron obtener noticias: {str(e)}")
        return []

# Función para analizar sentimiento de noticias
def analyze_sentiment(news):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for item in news:
        score = sia.polarity_scores(item['headline'])
        sentiment_scores.append(score['compound'])
        item['sentiment'] = score['compound']
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    
    return news, avg_sentiment

# Función para generar recomendación
def generate_recommendation(prediction, probability, sentiment, accuracy):
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

# Interfaz principal de la aplicación
def main():
    # Cabecera
    st.markdown('<div class="header">Stock Forecast Pro</div>', unsafe_allow_html=True)
    st.caption("Herramienta avanzada de análisis bursátil basada en IA - Datos en tiempo real")
    
    # Barra lateral para parámetros
    with st.sidebar:
        st.header("⚙️ Configuración")
        ticker = st.text_input("Símbolo bursátil (ej: AAPL, MSFT)", "AAPL").upper()
        
        # Fechas por defecto
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)  # 5 años de datos
        
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
            data = get_stock_data(ticker, start_date, end_date)
            if data.empty:
                st.error(f"No se encontraron datos para {ticker}")
                return
            
            processed_data = create_features(data)
            model, accuracy, features = train_model(processed_data)
            
            # Obtener noticias y analizar sentimiento
            news = get_financial_news(ticker)
            news, avg_sentiment = analyze_sentiment(news)
            
            # Preparar datos para predicción
            latest_data = processed_data.iloc[-1][features].values.reshape(1, -1)
            
            # Hacer predicción
            prediction = model.predict(latest_data)[0]
            probability = model.predict_proba(latest_data)[0][prediction]
            
            # Generar recomendación
            recommendation, icon = generate_recommendation(prediction, probability, avg_sentiment, accuracy)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return
    
    # Layout principal
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"📊 Análisis de {ticker}")
        
        # Precio actual
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        price_change = current_price - prev_close
        percent_change = (price_change / prev_close) * 100
        
        st.metric("Precio Actual", f"${current_price:.2f}", 
                 f"{price_change:.2f} ({percent_change:.2f}%)", 
                 delta_color="inverse" if price_change < 0 else "normal")
        
        # Rendimiento clave
        st.subheader("Indicadores Clave")
        
        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("RSI", f"{processed_data['RSI'].iloc[-1]:.2f}", 
                     help="RSI > 70: Sobrecompra, RSI < 30: Sobreventa")
            
            st.metric("Volatilidad (30d)", f"{processed_data['Volatility'].iloc[-1]*100:.2f}%", 
                     help="Volatilidad anualizada")
            
        with col2a:
            st.metric("MA20/MA50", 
                     f"{'↑' if processed_data['MA20'].iloc[-1] > processed_data['MA50'].iloc[-1] else '↓'}",
                     help="MA20 sobre MA50: tendencia alcista")
            
            st.metric("Precisión Modelo", f"{accuracy*100:.2f}%", 
                     help="Precisión en los últimos 30 días")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"📉 Gráficos de Tendencias")
        
        # Crear gráfico de precios
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Precio y medias móviles
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Precio'), row=1, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA20'], 
                                name='MA20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA50'], 
                                name='MA50', line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA200'], 
                                name='MA200', line=dict(color='purple')), row=1, col=1)
        
        # RSI
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"🧠 Recomendación de IA")
        
        # Mostrar recomendación
        st.markdown(f"### {icon} **{recommendation}**")
        
        # Detalles de la predicción
        st.subheader("Análisis Predictivo")
        st.metric("Probabilidad Predicción", f"{probability*100:.2f}%", 
                 help="Confianza del modelo en la predicción")
        
        st.metric("Predicción", "ALZA" if prediction == 1 else "BAJA", 
                 delta="Positiva" if prediction == 1 else "Negativa", 
                 delta_color="normal" if prediction == 1 else "inverse")
        
        # Análisis de sentimiento
        sentiment_label = "Positivo" if avg_sentiment > 0.1 else "Negativo" if avg_sentiment < -0.1 else "Neutral"
        st.metric("Sentimiento Noticias", sentiment_label, 
                 f"{avg_sentiment:.2f}", 
                 delta_color="normal" if avg_sentiment > 0 else "inverse")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sección de noticias
    st.markdown("---")
    st.subheader(f"📰 Últimas Noticias de {ticker}")
    
    if news:
        for item in news:
            sentiment_class = "positive" if item['sentiment'] > 0.1 else "negative" if item['sentiment'] < -0.1 else ""
            
            with st.expander(f"{item['time']}: {item['headline']}"):
                st.markdown(f"**Sentimiento:** `{item['sentiment']:.2f}`")
                st.markdown(f"[Leer más](https://www.marketwatch.com{item['link']})")
    else:
        st.warning("No se encontraron noticias recientes para este símbolo.")
    
    # Sección de datos históricos
    st.markdown("---")
    st.subheader(f"📊 Datos Históricos de {ticker}")
    
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

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
