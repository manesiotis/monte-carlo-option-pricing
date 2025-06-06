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





