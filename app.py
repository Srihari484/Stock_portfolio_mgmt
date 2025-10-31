import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests_cache

# Cache Yahoo Finance responses (prevents API rate issues)
requests_cache.install_cache('yfinance_cache', expire_after=1800)

st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")
st.title("📊 Stock Portfolio Optimizer")

# --- Inputs ---
ticker_input = st.text_input(
    "Enter Stock Symbols separated by comma (e.g. AAPL, MSFT, TSLA or RELIANCE.NS, TCS.NS):",
    "AAPL, MSFT"
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

weights_input = st.text_input(
    "Enter Weights separated by comma (e.g. 0.5, 0.3, 0.2):",
    "0.5, 0.5"
)
weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]

# --- Fetch Data Safely ---
try:
    if tickers:
        # Try downloading data
        data = yf.download(tickers, period="1y", progress=False, threads=False, timeout=30)

        # --- Validate data structure ---
        if data.empty:
            st.error("⚠️ No data downloaded. Please check ticker symbols or try again later.")
            st.stop()

        # Handle multi-index DataFrame (for multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        elif 'Close' in data.columns:
            data = data[['Close']]
        else:
            st.error("⚠️ Could not find 'Close' prices. Try valid stock tickers (e.g., AAPL, TCS.NS).")
            st.stop()

        # Drop missing columns or rows
        data = data.dropna(axis=1, how='all').dropna()
        if data.empty:
            st.error("⚠️ No valid closing price data found for the selected tickers.")
            st.stop()

        st.write("### ✅ Recent Stock Data", data.tail())
        st.line_chart(data)

        # --- Returns ---
        returns = data.pct_change().dropna()

        # Validate weights count
        if len(weights) != len(tickers):
            st.error("⚠️ Number of weights must match number of tickers.")
            st.stop()

        weights_series = pd.Series(weights, index=data.columns)
        portfolio_returns = (returns * weights_series).sum(axis=1)

        st.write("### 📉 Portfolio Daily Returns", portfolio_returns.tail())
        st.line_chart(portfolio_returns)

        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        st.write("### 📈 Portfolio Cumulative Returns", cumulative_returns.tail())
        st.line_chart(cumulative_returns)

        # --- Metrics ---
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        risk_free_rate = 0.0675
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else 0

        st.subheader("📊 Portfolio Performance Metrics")
        st.write(f"**Annualized Return:** {mean_return:.2%}")
        st.write(f"**Annualized Volatility:** {volatility:.2%}")
        st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (mean_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
        st.write(f"**Sortino Ratio:** {sortino_ratio:.2f}")

        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / rolling_max - 1
        max_drawdown = drawdown.min()
        st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")

        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        st.write(f"**Calmar Ratio:** {calmar_ratio:.2f}")

        VaR_95 = portfolio_returns.quantile(0.05)
        CVaR_95 = portfolio_returns[portfolio_returns <= VaR_95].mean()
        st.write(f"**Value at Risk (95%):** {VaR_95:.2%}")
        st.write(f"**Conditional Value at Risk (95%):** {CVaR_95:.2%}")

        # --- Correlation Heatmap ---
        st.write("### 🔥 Correlation Matrix")
        corr_matrix = returns.corr()
        st.dataframe(corr_matrix)
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

        # --- Efficient Frontier ---
        st.title("💹 Efficient Frontier Optimization")
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_portfolios = 3000
        results = np.zeros((3, num_portfolios))
        weight_array = []

        for i in range(num_portfolios):
            w = np.random.random(len(data.columns))
            w /= np.sum(w)
            weight_array.append(w)
            portfolio_return = np.dot(w, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
            results[0, i], results[1, i], results[2, i] = portfolio_return, portfolio_volatility, sharpe

        results_df = pd.DataFrame({"Return": results[0], "Volatility": results[1], "Sharpe": results[2]})
        max_sharpe_idx = results_df["Sharpe"].idxmax()
        max_sharpe_port = results_df.loc[max_sharpe_idx]
        opt_weights = weight_array[max_sharpe_idx]

        st.subheader("📌 Optimal Portfolio (Max Sharpe Ratio)")
        st.write(f"**Expected Return:** {max_sharpe_port['Return']:.2%}")
        st.write(f"**Volatility:** {max_sharpe_port['Volatility']:.2%}")
        st.write(f"**Sharpe Ratio:** {max_sharpe_port['Sharpe']:.2f}")
        st.write("**Weights Allocation:**")
        st.write(dict(zip(data.columns, [round(w, 3) for w in opt_weights])))

        plt.figure(figsize=(10, 6))
        plt.scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sharpe"], cmap="viridis", s=10)
        plt.colorbar(label="Sharpe Ratio")
        plt.scatter(max_sharpe_port["Volatility"], max_sharpe_port["Return"], c="red", s=100, marker="*", label="Max Sharpe")
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.legend()
        st.pyplot(plt)

        # --- Benchmark Comparison ---
        st.subheader("📈 Portfolio vs Benchmark (NIFTY 50)")
        benchmark = "^NSEI"
        benchmark_data = yf.download(benchmark, period="1y", progress=False, timeout=30)["Close"].dropna()

        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (benchmark_data.pct_change().fillna(0) + 1).cumprod()

        comparison_df = pd.DataFrame({"Portfolio": portfolio_cum, "Benchmark (NIFTY 50)": benchmark_cum})
        st.line_chart(comparison_df)

        benchmark_return = benchmark_data.pct_change().mean() * 252
        st.write(f"**Portfolio Annual Return:** {mean_return:.2%}")
        st.write(f"**Benchmark Annual Return (NIFTY 50):** {benchmark_return:.2%}")
        excess_return = mean_return - benchmark_return
        st.write(f"**Excess Return vs NIFTY:** {excess_return:.2%}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
