# app.py
import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEXSCRFPROTECTION"] = "false"
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 Dashboard de Portafolio de Inversión")

# Sidebar
st.sidebar.title("📌 Parámetros del Portafolio")
tickers_input = st.sidebar.text_area("Tickers (separados por coma)", "IVV,KO,LQD,NVDA,TIP,TLT,META,SMR,QQQ,BND,JPM,AVGO,IYY,AGG")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Sidebar: selección de período
st.sidebar.markdown("### ⏱️ Rango de tiempo")
period = st.sidebar.selectbox(
    "Selecciona el período",
    options=["1d", "1mo","3mo","6mo","9mo", "ytd", "1y", "2y", "3y", "5y"],
    index=5  # por defecto 5y
)

# Botón de análisis
if st.sidebar.button("🔄 Actualizar análisis"):

    with st.spinner("🔍 Descargando datos y calculando métricas..."):

        # Descargar precios ajustados según período elegido
        data = yf.download(tickers, period=period)["Close"]
        returns = data.pct_change().dropna()

        # Calcular métricas una sola vez
        annual_return = returns.mean() * 252 * 100
        annual_volatility = returns.std() * (252 ** 0.5) * 100
        
        # Asignar pesos iguales por defecto si no hay pesos definidos
        n = len(annual_return)
        weights = [1/n] * n  # o personaliza con tus propios pesos

        # Calcular retorno esperado del portafolio (ponderado)
        retorno_esperado = sum(w * r for w, r in zip(weights, annual_return))

        st.subheader("📐 Retorno Esperado del Portafolio")
        st.metric("Rentabilidad Anual Esperada", f"{retorno_esperado:.2f}%")

        # Generar resumen de cada ticker
        summary = []

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info

                name = info.get('shortName', ticker)
                quote_type = info.get('quoteType', 'Unknown')
                dividend_yield = info.get('dividendYield', 0.0)
                dividend = '✅ Sí' if dividend_yield and dividend_yield > 0 else '❌ No'

                name_lower = name.lower()
                if 'bond' in name_lower or 'treasury' in name_lower or 'corporate' in name_lower:
                    tipo = 'Renta Fija'
                elif 'gold' in name_lower or 'silver' in name_lower:
                    tipo = 'Commodity'
                elif quote_type == 'EQUITY':
                    tipo = 'Renta Variable'
                elif quote_type == 'ETF':
                    tipo = 'Renta Variable'
                else:
                    tipo = 'Desconocido'

                summary.append({
                    'Ticker': ticker,
                    'Nombre': name,
                    'Tipo Detectado': quote_type,
                    'Clase Inferida': tipo,
                    'Dividendos': dividend,
                    'Dividend Yield (%)': round(dividend_yield * 100, 2) if dividend_yield else 0.0,
                    'Rentabilidad (%)': round(annual_return.get(ticker, 0.0), 2),
                    'Volatilidad (%)': round(annual_volatility.get(ticker, 0.0), 2)
                })
            except Exception as e:
                st.warning(f"Error procesando {ticker}: {e}")

        df = pd.DataFrame(summary)
        df = df.sort_values(by=['Clase Inferida', 'Dividend Yield (%)'], ascending=[True, False])

        st.header("📋 Detalle del Portafolio")
        st.dataframe(df, use_container_width=True)

        # 🔗 Correlación promedio
        corr_matrix = returns.corr()
        lower_triangle = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
        correlacion_promedio = lower_triangle.stack().mean()

        # 📉 Volatilidad del portafolio vs. promedio individual
        cov_matrix = returns.cov() * 252
        weights_np = np.array(weights)
        portfolio_volatility = np.sqrt(weights_np @ cov_matrix.values @ weights_np.T)
        avg_individual_vol = annual_volatility.mean()

        # 📈 Sharpe Ratio
        risk_free_rate = 0.02  # 2% anual
        sharpe_ratio = (retorno_esperado / 100 - risk_free_rate) / portfolio_volatility

        st.subheader("📊 Indicadores Globales del Portafolio")
        st.caption("""
🔍 **Interpretación rápida de indicadores**:

- **Correlación Promedio**: mide cuán relacionados están los activos. Un valor bajo (cerca de 0) indica buena diversificación.
- **Volatilidad del Portafolio**: qué tan variable es el rendimiento del portafolio completo. Menor suele ser mejor si el retorno se mantiene.
- **Volatilidad Prom. de Activos**: el riesgo individual promedio. Si el portafolio tiene menos, estás diversificando bien.
- **Sharpe Ratio**: mide retorno ajustado por riesgo.  
  - 🟥 < 1: bajo (poca compensación por el riesgo)  
  - 🟨 1 – 2: bueno  
  - 🟩 > 2: excelente (alta eficiencia)

Estos indicadores te ayudan a evaluar si tu portafolio es robusto, balanceado y eficiente.
""")

        st.metric("🔗 Correlación Promedio", f"{correlacion_promedio:.2f}")
        st.metric("📉 Volatilidad del Portafolio", f"{portfolio_volatility*100:.2f}%")
        st.metric("📊 Volatilidad Prom. de Activos", f"{avg_individual_vol:.2f}%")
        st.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")

        st.subheader("📉 Correlación entre activos")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("📈 Evolución Acumulada del Precio (Total Return)")
        cumulative = (1 + returns).cumprod()
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative.plot(ax=ax, linewidth=2)
        ax.set_title("Evolución Acumulada de los Activos")
        ax.set_ylabel("Índice Total Retorno")
        ax.grid(True)
        st.pyplot(fig)

        retorno_esperado_decimal = retorno_esperado / 100
        monthly_return = (1 + retorno_esperado_decimal) ** (1/12) - 1
        months = len(cumulative)
        baseline = [1 * ((1 + monthly_return) ** i) for i in range(months)]
        baseline_series = pd.Series(baseline, index=cumulative.index)

        st.subheader("📈 Evolución Acumulada vs. Retorno Esperado del Portafolio")
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative.plot(ax=ax, linewidth=2)
        baseline_series.plot(ax=ax, color='black', linestyle='--', label=f"Retorno Esperado ({retorno_esperado:.2f}% anual)")
        ax.set_title("Evolución Acumulada de los Activos y Línea Objetivo")
        ax.set_ylabel("Índice Total Retorno")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 Rentabilidad vs. Volatilidad (Risk-Return Plot)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(annual_volatility * 100, annual_return * 100)
        for ticker in annual_return.index:
            ax.annotate(
                ticker,
                (annual_volatility[ticker] * 100, annual_return[ticker] * 100),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        ax.set_xlabel("Volatilidad Anual (%)")
        ax.set_ylabel("Rentabilidad Anual (%)")
        ax.set_title("Rentabilidad vs. Volatilidad")
        ax.grid(True)
        st.pyplot(fig)