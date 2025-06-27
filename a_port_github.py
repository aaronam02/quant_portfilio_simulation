import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ---- User-defined inputs ----
tickers = ["PRU", "BRK-B", "GOOGL", "SPYG", "VOOG", "VUG", "SCHG", "VTI", "IWF"]
weights = np.array([.275, 0.0754, 0.0549, 0.1272, .089, .0836, .0691, .0589, .0404])
lookback_period = "3y" # choose your horizon (e.g., '6mo', '2y', etc.)
T = 3      # years
N = 252 * T    # trading days
dt = T / N
M = 10000  # number of simulations
n_assets = len(tickers)

# ---- Download historical data ----
data = yf.download(tickers, period=lookback_period)["Close"]

# ---- Calculate returns and parameters ----
log_returns = np.log(data / data.shift(1)).dropna()
mu = log_returns.mean().values * 252                      # annualized drift
sigma = log_returns.std().values * np.sqrt(252)          # annualized volatility
corr_matrix = log_returns.corr().values                  # correlation
cov_matrix = np.outer(sigma, sigma) * corr_matrix        # covariance
L = np.linalg.cholesky(cov_matrix)                       # Cholesky for correlated shocks

# ---- Starting prices ----
S0 = data.iloc[-1].values  # most recent closing prices

# ---- Simulate price paths ----
price_paths = np.zeros((M, N + 1, n_assets))
price_paths[:, 0, :] = S0

for m in range(M):
    Z = np.random.normal(size=(N, n_assets))
    correlated_Z = Z @ L.T
    for t in range(1, N + 1):
        price_paths[m, t, :] = price_paths[m, t - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + np.sqrt(dt) * correlated_Z[t - 1]
        )

# ---- Compute portfolio values ----
aum_amt = 40000
portfolio_values = np.sum(price_paths * weights, axis=2)
initial_portfolio_value = np.sum(S0 * weights)
scale_factor = aum_amt / initial_portfolio_value
portfolio_values *= scale_factor

# ---- Plotting ----
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Left: Simulated Paths
for i in range(10000):
    axs[0].plot(portfolio_values[i], color='blue', alpha=0.01)
axs[0].set_title("Sample Simulated Portfolio Paths")
axs[0].set_xlabel("Time Step (Day)")
axs[0].set_ylabel("Portfolio Value ($)")
axs[0].grid(True)

# Portfolio metadata textbox
weights_percent = weights * 100
textstr = '\n'.join((
    "Portfolio Weights & Starting Prices:",
    *(f"{tickers[i]}: {weights_percent[i]:.1f}% @ ${S0[i]:.2f}" for i in range(n_assets))
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

# Right: Histogram of final values
final_values = portfolio_values[:, -1]
axs[1].hist(final_values, bins=50, edgecolor='black')
axs[1].set_title("Distribution of Final Portfolio Values")
axs[1].set_xlabel("Portfolio Value at T ($)")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

# Risk metrics
VaR_95 = np.percentile(final_values, 5)
pct_loss = np.mean(final_values < aum_amt) * 100

axs[1].axvline(VaR_95, color='red', linestyle='--', label=f'VaR (5%): ${VaR_95:,.0f}')
axs[1].axvline((2*aum_amt), color='orange', linestyle='--', label='Initial Portfolio Value')
axs[1].legend()

textstr2 = f"Value at Risk (5% quantile): ${VaR_95:,.2f}\n" \
           f"Chance of Loss (below ${aum_amt}): {pct_loss:.2f}%"
axs[1].text(0.95, 0.95, textstr2, transform=axs[1].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.show()
