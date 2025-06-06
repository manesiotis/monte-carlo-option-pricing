# monte_carlo_pricing.py
# Author: Konstantinos Manesiotis

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ----- Stock and Option Parameters -----

S0 = 100        # initial stock price
K = 105         # strike price
T = 1.0         # time to maturity (in years)
r = 0.05        # risk-free interest rate
sigma = 0.2     # volatility

n_simulations = 10000  # number of simulated paths
n_steps = 252          # time steps (1 year = 252 trading days)
dt = T / n_steps       # time increment

# ----- Simulating Stock Price Paths using Geometric Brownian Motion -----

np.random.seed(42)  # for reproducibility

S = np.zeros((n_simulations, n_steps + 1))
S[:, 0] = S0  # initialize all paths with the starting stock price

for t in range(1, n_steps + 1):
    Z = np.random.standard_normal(n_simulations)  # random shocks from N(0,1)
    S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# ----- Plotting sample paths -----

plt.figure(figsize=(10, 6))
for i in range(100):  # plot only 100 paths for clarity
    plt.plot(S[i], linewidth=0.7, alpha=0.6)

plt.title("Monte Carlo Simulation of Stock Price Paths")
plt.xlabel("Time Steps (days)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/stock_paths.png")
plt.show()

# ----- Monte Carlo Estimation of Option Prices -----

S_T = S[:, -1]  # final stock price at maturity

call_payoff = np.maximum(S_T - K, 0)  # payoff for Call option
call_price = np.exp(-r * T) * np.mean(call_payoff)  # discounted average payoff

put_payoff = np.maximum(K - S_T, 0)   # payoff for Put option
put_price = np.exp(-r * T) * np.mean(put_payoff)

print(f"Monte Carlo Estimated Call Price: {call_price:.4f}")
print(f"Monte Carlo Estimated Put Price:  {put_price:.4f}")

# ----- Black-Scholes Analytical Prices -----

d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
bs_put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

print(f"Black-Scholes Call Price:         {bs_call:.4f}")
print(f"Black-Scholes Put Price:          {bs_put:.4f}")
