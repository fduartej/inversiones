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
default_weight = round(1 / len(tickers), 4)


st.sidebar.markdown("📅 Opcional: Historial de Dividendos")

show_dividends = st.sidebar.checkbox("📥 Mostrar calendario histórico de dividendos por activo", value=False)


st.sidebar.markdown("### 🧮 Asignación de pesos (%)")
weights = []
for t in tickers:
    w = st.sidebar.number_input(f"Peso {t}", min_value=0.0, max_value=1.0, value=default_weight, step=0.01)
    weights.append(w)

# Validación
if abs(sum(weights) - 1.0) > 0.01:
    st.sidebar.error("❌ Los pesos deben sumar 1. Ajusta los valores.")
    st.stop()

# Monto total invertido
total_investment = st.sidebar.number_input("💰 Monto total invertido (USD)", min_value=1000.0, value=10000.0, step=100.0)
weights_np = np.array(weights)

st.subheader("📊 Distribución del Portafolio (Pesos)")
fig, ax = plt.subplots()
ax.pie(weights, labels=tickers, autopct="%1.1f%%", startangle=90)
ax.axis("equal")
st.pyplot(fig)

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
        portfolio_returns = returns.dot(weights_np)
        
        # Benchmark: SPY
        benchmark_ticker = "SPY"
        benchmark_data = yf.download(benchmark_ticker, period=period)["Close"]
        benchmark_returns = benchmark_data.pct_change().dropna()

        # Alinear fechas
        combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        combined.columns = ['Portafolio', 'Benchmark']

        # Regresión lineal para Alpha y Beta
        from sklearn.linear_model import LinearRegression

        X = combined['Benchmark'].values.reshape(-1, 1)
        y = combined['Portafolio'].values

        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        alpha = model.intercept_ * 252  # anualizado
        r_squared = model.score(X, y)
        
        st.subheader("📌 Comparación con Benchmark (SPY)")
        st.metric("📈 Beta", f"{beta:.2f}")
        st.metric("📊 Alpha anualizado", f"{alpha:.2%}")
        st.metric("📐 R² (Explicación)", f"{r_squared:.2%}")

        st.info("""
        - **Beta** > 1: más volátil que el mercado.  
        - **Alpha** positivo: generas retorno por encima del mercado.  
        - **R²**: qué tan bien tus movimientos siguen al benchmark.
        """)
                
                
        st.subheader("📈 Evolución acumulada: Portafolio vs SPY")

        cumulative_returns = (1 + combined).cumprod()
        fig, ax = plt.subplots(figsize=(12,6))
        cumulative_returns.plot(ax=ax, linewidth=2)
        plt.title("Portafolio vs. Benchmark (SPY)")
        plt.grid(True)
        st.pyplot(fig)
        
        # Rolling Sharpe Ratio (90 días)
        window_days = 90
        rolling_return = returns.dot(weights_np).rolling(window=window_days).mean()
        rolling_volatility = returns.dot(weights_np).rolling(window=window_days).std()
        rolling_sharpe = (rolling_return / rolling_volatility) * np.sqrt(252)

        # Calcular métricas una sola vez
        annual_return = returns.mean() * 252 * 100
        annual_volatility = returns.std() * (252 ** 0.5) * 100
        
        # Asignar pesos iguales por defecto si no hay pesos definidos
        retorno_esperado = np.dot(weights_np, annual_return / 100) * 100  # expresado en porcentaje
        cov_matrix = returns.cov() * 252
        portfolio_volatility = np.sqrt(weights_np @ cov_matrix.values @ weights_np.T)

        st.subheader("📐 Retorno Esperado del Portafolio")
        st.metric("Rentabilidad Anual Esperada", f"{retorno_esperado:.2f}%")
        ganancia_esperada = total_investment * retorno_esperado / 100
        st.metric("📈 Ganancia Estimada (1 año)", f"${ganancia_esperada:,.2f}")
        
        # Parámetros (puedes ajustarlos)
        tasa_impuesto = 0.05
        tasa_comision = 0.005  # 0.5%

        # Cálculos
        impuesto = ganancia_esperada * tasa_impuesto
        comision = total_investment * tasa_comision
        ganancia_neta = ganancia_esperada - impuesto - comision

        # Mostrar en Streamlit
        st.markdown("### 💸 Ajustes por Impuestos y Comisiones (Perú)")

        st.write(f"**🧾 Impuesto a la ganancia de capital (5%)**: ${impuesto:,.2f}")
        st.write(f"**💼 Comisión estimada (0.5%)**: ${comision:,.2f}")
        st.success(f"**✅ Ganancia neta esperada**: ${ganancia_neta:,.2f}")
        
        st.subheader("📈 Sharpe Ratio Móvil (90 días)")
        fig, ax = plt.subplots(figsize=(12, 5))
        rolling_sharpe.plot(ax=ax, color='purple')
        ax.axhline(0, linestyle='--', color='gray', linewidth=1)
        ax.set_title("Sharpe Ratio Móvil del Portafolio")
        ax.set_ylabel("Sharpe")
        ax.grid(True)
        st.pyplot(fig)
        


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
        
        st.subheader("💵 Valor Acumulado del Portafolio (USD reales)")

        valor_inversion_acumulado = cumulative.dot(weights_np) * total_investment
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        valor_inversion_acumulado.plot(ax=ax2, linewidth=2, color='green', label="Valor Portafolio (USD)")
        ax2.set_title("💰 Evolución del Valor Total del Portafolio")
        ax2.set_ylabel("USD")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

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
        
        # ==============================
        # 📉 Cálculo del Drawdown
        # ==============================

        st.subheader("📉 Drawdown del Portafolio (Caída desde el Máximo)")

        # Índice de riqueza (rendimiento acumulado)
        wealth_index = (1 + returns.dot(weights_np)).cumprod()

        # Máximos anteriores
        previous_peaks = wealth_index.cummax()

        # Drawdown en %
        drawdown = (wealth_index - previous_peaks) / previous_peaks

        # Gráfico
        fig_dd, ax_dd = plt.subplots(figsize=(12, 4))
        drawdown.plot(ax=ax_dd, color='crimson')
        ax_dd.set_title("Drawdown del Portafolio")
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.grid(True)
        st.pyplot(fig_dd)

        # ==============================
        # ⚖️ Sortino Ratio
        # ==============================

        # Retornos negativos (downside)
        negative_returns = portfolio_returns[portfolio_returns < 0]

        # Desviación estándar de las pérdidas
        downside_std = negative_returns.std()

        # Rentabilidad media anualizada
        mean_return = portfolio_returns.mean() * 252

        # Sortino Ratio
        sortino_ratio = mean_return / downside_std if downside_std != 0 else np.nan

        # Mostrar en la app
        st.subheader("⚖️ Sortino Ratio – Riesgo Ajustado a Caídas")
        st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")

        # Interpretación automática
        if sortino_ratio >= 1.5:
            color = "🟢"
            interpret = "Excelente relación rentabilidad / riesgo negativo."
        elif sortino_ratio >= 1.0:
            color = "🟡"
            interpret = "Aceptable. El portafolio maneja bien el riesgo bajista."
        else:
            color = "🔴"
            interpret = "Riesgo alto en las caídas. Revisa diversificación y estabilidad."

        st.info(f"{color} {interpret}")

        st.subheader("📉 Valor en Riesgo (VaR Histórico)")

        # Monto invertido inicial simulado
        monto_invertido = st.number_input("💵 Monto invertido (USD)", min_value=1000, value=10000, step=1000)

        # VaR diario al 95%
        var_95 = np.percentile(portfolio_returns, 5)
        valor_en_riesgo = -var_95 * monto_invertido

        # Mostrar
        st.metric(label="🔻 VaR Diario al 95%", value=f"${valor_en_riesgo:,.2f}",
                delta=f"{var_95*100:.2f}%")

        # Interpretación automática
        st.info("Con 95% de confianza, **no deberías perder más de ese valor en un mal día promedio.**")


        st.subheader("🗓️ Retornos Mensuales del Portafolio (Calendar Heatmap)")

        # Retornos mensuales del portafolio
        monthly_returns = portfolio_returns.resample('M').apply(lambda r: (1 + r).prod() - 1)

        # Convertir a DataFrame Año x Mes
        monthly_returns_df = monthly_returns.to_frame(name="Retorno")
        monthly_returns_df["Año"] = monthly_returns_df.index.year
        monthly_returns_df["Mes"] = monthly_returns_df.index.strftime("%b")

        # Pivot para el heatmap
        calendar = monthly_returns_df.pivot(index="Año", columns="Mes", values="Retorno")

        # Asegurar orden de meses
        meses_ordenados = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        calendar = calendar.reindex(columns=[m for m in meses_ordenados if m in calendar.columns])


        # Gráfico
        fig_cal, ax_cal = plt.subplots(figsize=(12, 5))
        sns.heatmap(calendar * 100, cmap="RdYlGn", center=0, annot=True, fmt=".1f", linewidths=0.5, ax=ax_cal)
        ax_cal.set_title("Retornos Mensuales del Portafolio (%)")
        st.pyplot(fig_cal)
        
      
        if show_dividends:
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    dividends = t.dividends

                    if dividends.empty:
                        st.write(f"📭 {ticker}: No se encontraron dividendos históricos.")
                        continue

                    dividends.index = dividends.index.date  # fechas legibles
                    df_div = dividends.rename("Dividendo").to_frame()
                    df_div.index.name = "Fecha"

                    st.subheader(f"📌 {ticker}")
                    st.dataframe(df_div.tail(10).style.format({"Dividendo": "${:,.2f}"}))
                except Exception as e:
                    st.warning(f"⚠️ {ticker}: Error al obtener dividendos. {e}")
        else:
            st.caption("✅ Puedes activar esta opción si deseas ver el historial real de dividendos por activo.")


