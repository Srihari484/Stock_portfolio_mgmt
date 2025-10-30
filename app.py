import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stock Portfolio Optimizer", layout="wide")
st.title("📊 Stock Portfolio Optimizer - Starter")

# --- Inputs ---
ticker_input = st.text_input("Enter Stock Symbols separated by comma (e.g. AAPL, MSFT, TSLA):", "AAPL, MSFT")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

weights_input = st.text_input("Enter Weights separated by comma (e.g. 0.5, 0.3, 0.2):", "0.5, 0.5")
weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]

# --- Fetch data safely ---
try:
    if tickers:
        data = yf.download(tickers, period="1y", progress=False, threads=False, timeout=30)["Close"]

        if data.isna().all().all():
            st.error("⚠️ Failed to download stock data. Please check your tickers or network connection.")
            st.stop()

        st.write("### Recent Data", data.tail())
        st.line_chart(data)

        # --- Returns ---
        returns = data.pct_change().dropna()
        weights_series = pd.Series(weights, index=tickers)

        if len(weights) != len(tickers):
            st.error("⚠️ Number of weights must match number of tickers.")
            st.stop()

        portfolio_returns = (returns * weights_series).sum(axis=1)
        st.write("### Portfolio Daily Returns", portfolio_returns.tail())
        st.line_chart(portfolio_returns)

        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        st.write("### Portfolio Cumulative Returns", cumulative_returns.tail())
        st.line_chart(cumulative_returns)

        # --- Metrics ---
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        risk_free_rate = 0.0675
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else 0

        st.subheader("📈 Portfolio Performance Metrics")
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
        st.write("### Correlation Matrix")
        corr_matrix = returns.corr()
        st.dataframe(corr_matrix)
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

        # --- Efficient Frontier ---
        st.title("📊 Stock Portfolio Optimizer - Efficient Frontier")
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weight_array = []

        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            weight_array.append(weights)

            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0

            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio

        results_df = pd.DataFrame({"Return": results[0], "Volatility": results[1], "Sharpe": results[2]})

        if results_df["Sharpe"].isna().all():
            st.error("⚠️ Unable to calculate Sharpe ratios — data may be missing.")
            st.stop()

        max_sharpe_idx = results_df["Sharpe"].idxmax()
        max_sharpe_port = results_df.loc[max_sharpe_idx]
        opt_weights = weight_array[max_sharpe_idx]

        st.subheader("📌 Optimal Portfolio (Max Sharpe Ratio)")
        st.write(f"**Expected Return:** {max_sharpe_port['Return']:.2%}")
        st.write(f"**Volatility:** {max_sharpe_port['Volatility']:.2%}")
        st.write(f"**Sharpe Ratio:** {max_sharpe_port['Sharpe']:.2f}")
        st.write("**Weights Allocation:**")
        st.write(dict(zip(tickers, [round(w, 3) for w in opt_weights])))

        # Plot Frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sharpe"], cmap="viridis", s=10)
        plt.colorbar(label="Sharpe Ratio")
        plt.scatter(max_sharpe_port["Volatility"], max_sharpe_port["Return"], c="red", s=100, marker="*", label="Max Sharpe")
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.legend()
        st.pyplot(plt)

        # --- Monte Carlo Simulation ---
        st.subheader("🎲 Monte Carlo Simulation of Portfolio Value")
        initial_investment = st.number_input("Enter Initial Investment (e.g. 100000):", value=100000)
        num_simulations = 10000
        num_days = 252

        mean_daily_return = portfolio_returns.mean()
        std_daily_return = portfolio_returns.std()

        simulations = np.zeros((num_days, num_simulations))
        for i in range(num_simulations):
            daily_returns = np.random.normal(mean_daily_return, std_daily_return, num_days)
            price_series = initial_investment * (1 + daily_returns).cumprod()
            simulations[:, i] = price_series

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(simulations[:, :100], alpha=0.1, color="blue")

        ending_values = simulations[-1, :]
        p5, p95 = np.percentile(ending_values, [5, 95])
        ax.axhline(p5, color="red", linestyle="--", label=f"5th Percentile: {p5:,.0f}")
        ax.axhline(p95, color="green", linestyle="--", label=f"95th Percentile: {p95:,.0f}")
        ax.set_title("Monte Carlo Simulation of Portfolio Value")
        ax.set_xlabel("Days")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        st.pyplot(fig)

        prob_loss = np.mean(ending_values < initial_investment)
        st.write(f"📉 Probability of Loss after 1 year: {prob_loss:.2%}")
        st.write(f"💡 Expected Portfolio Value Range (90% CI): {p5:,.0f} – {p95:,.0f}")

        # --- Benchmark Comparison ---
        benchmark = "^NSEI"   # NIFTY 50
        benchmark_data = yf.download(benchmark, period="1y", progress=False, timeout=30)["Close"].dropna()

        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (benchmark_data.pct_change().fillna(0) + 1).cumprod()

        st.write("### Portfolio vs Benchmark (NIFTY 50)")
        comparison_df = pd.DataFrame({"Portfolio": portfolio_cum, "Benchmark (NIFTY 50)": benchmark_cum})
        st.line_chart(comparison_df)

        benchmark_return = benchmark_data.pct_change().mean() * 252
        st.write(f"**Portfolio Annual Return:** {mean_return:.2%}")
        st.write(f"**Benchmark Annual Return (NIFTY 50):** {benchmark_return:.2%}")

        excess_return = mean_return - benchmark_return
        st.write(f"**Excess Return vs NIFTY:** {excess_return:.2%}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
