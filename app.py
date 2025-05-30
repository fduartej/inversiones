# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📈 Dashboard de Portafolio de Inversión")

# Sidebar
st.sidebar.title("📌 Parámetros del Portafolio")
tickers_input = st.sidebar.text_area("Tickers (separados por coma)", "IVV,KO,LQD,NVDA,TIP,TLT")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Sidebar: selección de período
st.sidebar.markdown("### ⏱️ Rango de tiempo")
period = st.sidebar.selectbox(
    "Selecciona el período",
    options=["1d", "1mo", "ytd", "1y", "2y", "3y", "5y"],
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
        
        # Asignar pesos iguales por defecto si no hay pesos definidos
        n = len(annual_return)
        weights = [1/n] * n  # o personaliza con tus propios pesos

        # Calcular retorno esperado del portafolio (ponderado)
        retorno_esperado = sum(w * r for w, r in zip(weights, annual_return))

        st.subheader("📐 Retorno Esperado del Portafolio")
        st.metric("Rentabilidad Anual Esperada", f"{retorno_esperado:.2f}%")

        
        annual_volatility = returns.std() * (252 ** 0.5) * 100

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

        # Crear DataFrame y ordenarlo
        df = pd.DataFrame(summary)
        df = df.sort_values(by=['Clase Inferida', 'Dividend Yield (%)'], ascending=[True, False])

        # Mostrar tabla
        st.header("📋 Detalle del Portafolio")
        st.dataframe(df, use_container_width=True)

        # Heatmap de correlación
        st.subheader("📉 Correlación entre activos")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Gráfico de evolución acumulada
        st.subheader("📈 Evolución Acumulada del Precio (Total Return)")

        cumulative = (1 + returns).cumprod()

        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative.plot(ax=ax, linewidth=2)
        ax.set_title("Evolución Acumulada de los Activos")
        ax.set_ylabel("Índice Total Retorno")
        ax.grid(True)
        st.pyplot(fig)


        # Calcular retorno esperado del portafolio
        n = len(annual_return)
        weights = [1/n] * n  # Puedes personalizar esto más adelante
        retorno_esperado = sum(w * r for w, r in zip(weights, annual_return)) / 100  # en decimal

        # Calcular retorno mensual equivalente
        monthly_return = (1 + retorno_esperado) ** (1/12) - 1

        # Línea base de retorno compuesto
        months = len(cumulative)
        baseline = [1 * ((1 + monthly_return) ** i) for i in range(months)]
        baseline_series = pd.Series(baseline, index=cumulative.index)

        # Gráfico acumulado + rendimiento esperado
        st.subheader("📈 Evolución Acumulada vs. Retorno Esperado del Portafolio")

        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative.plot(ax=ax, linewidth=2)
        baseline_series.plot(ax=ax, color='black', linestyle='--', label=f"Retorno Esperado ({retorno_esperado*100:.2f}% anual)")
        ax.set_title("Evolución Acumulada de los Activos y Línea Objetivo")
        ax.set_ylabel("Índice Total Retorno")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)


        # Gráfico de rentabilidad vs. volatilidad
        st.subheader("📉 Rentabilidad vs. Volatilidad (Risk-Return Plot)")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Crear scatter
        ax.scatter(annual_volatility * 100, annual_return * 100)

        # Agregar etiquetas a cada punto
        for ticker in annual_return.index:
            ax.annotate(
                ticker,
                (annual_volatility[ticker] * 100, annual_return[ticker] * 100),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )

        # Configurar gráfico
        ax.set_xlabel("Volatilidad Anual (%)")
        ax.set_ylabel("Rentabilidad Anual (%)")
        ax.set_title("Rentabilidad vs. Volatilidad")
        ax.grid(True)

        st.pyplot(fig)