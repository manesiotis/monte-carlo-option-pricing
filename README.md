# Monte Carlo Option Pricing

This project simulates thousands of stock price paths using the Monte Carlo method in order to estimate the price of a European Call and Put option.

## Why Monte Carlo?

Monte Carlo methods are useful for pricing options when closed-form solutions (like Black-Scholes) are unavailable or impractical, such as in high-dimensional or path-dependent options. In this project, we use it to estimate prices and compare them to the Black-Scholes formula.

## Assumptions

- Stock prices follow a geometric Brownian motion.
- Constant volatility and interest rate.
- Log-normal distribution of prices.

## What’s included

- Simulation of thousands of paths.
- Estimation of European Call and Put option prices.
- Visualization of stock price evolution.
- Comparison with Black-Scholes analytic solution.

## How to Run

```bash
pip install -r requirements.txt
python monte_carlo_pricing.py

## Output

- Estimated Call/Put prices printed in the console.
- Plot of 100 simulated stock paths saved in `plots/stock_paths.png`.

Example:
Monte Carlo Estimated Call Price: 8.0214  
Black-Scholes Call Price:         8.0286
