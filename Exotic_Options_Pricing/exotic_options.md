---
layout: wide_default
---

# Exotic Option Pricing and Advanced Derivatives Strategies

## Project Overview

**Objective:** Price Rainbow, Chooser, and FX options using multiple sophisticated mathematical models to validate pricing accuracy and implement practical hedging strategies.

**Key Innovations:**
- Multi-model validation approach for exotic derivatives pricing
- Static portfolio replication testing (min-max parity argument)
- Cross-asset hedging strategies combining FX and Chooser options

## Technical Methodology

### Models Implemented

#### 1. Rainbow Options (Multi-Asset Options)
**Underlying Assets:** US Short Term Treasury Bond Index & US Corporate Bond Index  
**Models Used:**
- Monte Carlo Simulation with Correlated GBM using Cholesky decomposition
- Ornstein-Uhlenbeck process for mean reversion dynamics

**Mathematical Framework:**
- **Call on Max Payoff:** max(max(S₁, S₂, ..., Sₙ) - K, 0)
- **Call on Min Payoff:** max(min(S₁, S₂, ..., Sₙ) - K, 0)

#### 2. Foreign Exchange Options
**Currency Pair:** USD/JPY  
**Models Used:**
- Heston Model (Stochastic Volatility)
- Garman-Kohlhagen Model (Black-Scholes Extension)

#### 3. Chooser Options
**Underlying:** Bloomberg U.S. Universal Total Return Index  
**Models Used:**
- Cox-Ross-Rubinstein Binomial Model
- Kou Jump-Diffusion Model

### Data Sources and Risk-Free Rates
- **Market Data:** Bloomberg terminal
- **Risk-Free Rate:** US 3-Year Treasury bond rate
- **Analysis Period:** 2019-2023 (capturing tail events including COVID-19)

## Results and Analysis

### Rainbow Options Model Validation
<img src="images/rainbow_options_comparison.png?raw=true" alt="Rainbow Options Model Comparison" style="width:100%;">

**Key Findings:**
- Both Correlated GBM and Ornstein-Uhlenbeck models showed strong agreement in daily option pricing
- Min-max parity validation: Cmax(S₁, S₂, K) + Cmin(S₁, S₂, K) ≈ C(S₁,K) + C(S₂,K)
- Divergences observed during high volatility periods (2020 pandemic, 2022-2023 debt ceiling crisis)

### FX Options Model Comparison
<img src="images/fx_options_heston_vs_gk.png?raw=true" alt="FX Options: Heston vs Garman-Kohlhagen" style="width:100%;">

**Performance Analysis:**

| Model | Computational Complexity | Market Dynamics Capture | Practical Implementation |
|-------|--------------------------|-------------------------|-------------------------|
| Heston | High | Excellent (Stochastic Vol) | Research/Advanced Trading |
| Garman-Kohlhagen | Low | Good (Constant Vol) | Standard Trading Operations |

### Chooser Options Analysis
<img src="images/chooser_options_crr_kou.png?raw=true" alt="Chooser Options: CRR vs KOU Models" style="width:100%;">

**Model Comparison:**
- **CRR Model:** Binomial approach suitable for European-style exercise
- **KOU Model:** Jump-diffusion capturing extreme market movements and tail events


## Business Applications and Insights

### Practical Implementation
1. **Portfolio Diversification:** Rainbow options provide exposure to multiple assets with single instrument
2. **Currency Risk Management:** FX options essential for international portfolios
3. **Market Timing Flexibility:** Chooser options valuable in uncertain market conditions

### Risk Considerations
- **Model Risk:** Multiple model validation reduces single-model dependency
- **Liquidity Risk:** Exotic options typically have wider bid-ask spreads
- **Counterparty Risk:** OTC derivatives require careful counterparty selection

### Market Insights
- **COVID-19 Impact:** Increased volatility led to model divergences, highlighting importance of stress testing
- **Interest Rate Sensitivity:** Debt ceiling crisis affected treasury-based option pricing
- **Jump Risk:** Kou model effectively captured extreme market movements during crisis periods

## Technical Implementation

### Programming Framework
- **Languages:** Python, R
- **Libraries:** NumPy, SciPy, pandas, matplotlib
- **Mathematical Tools:** Monte Carlo simulation, numerical optimization
- **Data Sources:** Bloomberg API, Federal Reserve Economic Data (FRED)

### Model Calibration
- **Parameter Estimation:** Maximum likelihood estimation for stochastic processes
- **Volatility Modeling:** Rolling 90-day windows for dynamic volatility calculation
- **Correlation Analysis:** Cholesky decomposition for multi-asset correlation structure

## Key Mathematical Formulations

### Correlated GBM Dynamics
```
dS/S = (r - q)dt + AdW
```

**Analytical Solution:**
```
S(t+1) = S(t) × exp((r - σ²/2)Δt + σ√Δt × Z(t))
```

### Ornstein-Uhlenbeck Process
```
dX(t) = κ(θ - X(t))dt + σdW(t)
```

**Where:**
- κ: Rate of mean reversion
- θ: Long-term mean level
- σ: Volatility of the process

### Heston Model Dynamics
```
Exchange Rate: dS(t) = μS(t)dt + √v(t)S(t)dW₁(t)
Variance: dv(t) = κ(θ - v(t))dt + σ√v(t)dW₂(t)
```

**Correlation:** dW₁(t)·dW₂(t) = ρdt

## Conclusions and Future Enhancements

### Key Achievements
1. **Multi-Model Validation:** Successfully validated exotic option pricing across different mathematical frameworks
3. **Market Stress Analysis:** Captured how extreme events affect option pricing relationships

### Future Research Directions
- **Machine Learning Integration:** Incorporate ML models for parameter estimation
- **Real-Time Implementation:** Develop automated trading systems for strategy execution
- **Extended Asset Classes:** Apply methodology to cryptocurrency and commodity options

**[← Back to Portfolio](../index.md)**
