# monte_carlo_pricing.py
# Project 2: Option Pricing with Monte Carlo Simulation
# Author: Konstantinos Manesiotis

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----- Παράμετροι Μετοχής & Option -----

S0 = 100        # αρχική τιμή μετοχής
K = 105         # τιμή εξάσκησης (strike price)
T = 1.0         # χρόνος μέχρι τη λήξη (σε έτη)
r = 0.05        # επιτόκιο χωρίς ρίσκο
sigma = 0.2     # μεταβλητότητα (volatility)

n_simulations = 10000  # αριθμός paths
n_steps = 252          # βήματα ανά path (1 έτος = 252 trading days)

dt = T / n_steps       # μικρό χρονικό διάστημα

# ----- Monte Carlo Προσομοίωση -----

np.random.seed(42)  # για αναπαραγωγιμότητα

# αρχικοποίηση πίνακα: γραμμές = simulations, στήλες = χρονικά βήματα
S = np.zeros((n_simulations, n_steps + 1))
S[:, 0] = S0  # όλες οι διαδρομές ξεκινούν από την τιμή S0

# δημιουργία paths
for t in range(1, n_steps + 1):
    Z = np.random.standard_normal(n_simulations)  # N(0,1)
    S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

print(f"Τελική μέση τιμή μετοχής: {np.mean(S[:, -1]):.2f}")

# ----- Γράφημα paths -----

plt.figure(figsize=(10, 6))
for i in range(100):  # δείχνουμε μόνο 100 paths για καθαρότητα
    plt.plot(S[i], linewidth=0.7, alpha=0.6)

plt.title("Monte Carlo Simulation of Stock Price Paths")
plt.xlabel("Time Steps (days)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.tight_layout()

# Αποθήκευση εικόνας στο φάκελο plots/
import os
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/stock_paths.png")
plt.show()

# ----- Υπολογισμός Τιμής Option με Monte Carlo -----

S_T = S[:, -1]  # τιμές μετοχής στο τέλος κάθε path

# Call option payoff
call_payoff = np.maximum(S_T - K, 0)
call_price = np.exp(-r * T) * np.mean(call_payoff)

# Put option payoff
put_payoff = np.maximum(K - S_T, 0)
put_price = np.exp(-r * T) * np.mean(put_payoff)

print(f"Monte Carlo Estimated Call Price: {call_price:.4f}")
print(f"Monte Carlo Estimated Put Price:  {put_price:.4f}")

# ----- Υπολογισμός Black-Scholes Τιμών -----

d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
bs_put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

print(f"Black-Scholes Call Price: {bs_call:.4f}")
print(f"Black-Scholes Put Price:  {bs_put:.4f}")



