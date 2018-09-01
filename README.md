# Systemic Risk

This script calculates and analyses the following systemic risk indicators:

* `CoVaR` (Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009)
* `ΔCoVaR` (Delta Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009)
* `MES` (Marginal Expected Shortfall) proposed by Acharya et al. (2010)
* `Network Measures` proposed by Billio et al. (2011)
* `SRISK` (Conditional Capital Shortfall Index) proposed by Brownlees & Engle (2014)

## Requirements

The minimum Matlab version required is `R2014a`. In addition, the following products and toolboxes must be installed in order to properly execute the script:

* Computer Vision System Toolbox
* Curve Fitting Toolbox
* MATLAB Distributed Computing Server
* Optimization Toolbox
* Parallel Computing Toolbox
* Simulink Control Design
* Statistics and Machine Learning Toolbox
* System Identification Toolbox

## Usage

1. Create a properly structured database (see the paragraph below).
1. Edit the `run.m` script following your needs.
1. Execute the `run.m` script.

## Dataset

Every dataset must be structured like the default one included in any release of the framework (`Datasets/Example.xlsx`). The latter, based on the US financial sector, defines the following entities:

#### Benchmark: S&P 500

#### Financial Institutions (20):
* **Group 1: Insurance Companies (5)**
  * American International Group Inc. (AIG)
  * The Allstate Corp. (ALL)
  * Berkshire Hathaway Inc. (BRK)
  * MetLife Inc. (MET)
  * Prudential Financial Inc. (PRU)
* **Group 2: Investment Banks (6)**
  * Bank of America Corp. (BAC)
  * Citigroup Inc. (C)
  * The Goldman Sachs Group Inc. (GS)
  * J.P. Morgan Chase & Co. (JPM)
  * Lehman Brothers Holdings Inc. (LEH)
  * Morgan Stanley (MS) 
* **Group 3: Commercial Banks (7)**
  * American Express Co. (AXP)
  * Bank of New York Mellon Corp. (BK)
  * Capital One Financial Corp. (COF)
  * PNC Financial Services Inc. (PNC)
  * State Street Corp. (STT)
  * US Bancorp (USB)
  * Wells Fargo & Co. (WFC)
* **Group 4: Government-sponsored Enterprises (2)**
  * American International Group Inc. (AIG)
  * The Allstate Corp. (ALL)

#### State Variables (6):
* **RESI:** the DJ US Select RESI as a proxy of real estate returns.
* **VIX:** the implied volatility index.
* **TBR3M:** the 3M treasury bill rate.
* **CRESPR:** the change in the credit spread (the BAA corporate bond rate minus the 10Y treasury bond rate).
* **LIQSPR:** the change in the liquidity spread (the 3M treasury bill rate minus the federal funds rate).
* **YIESPR:** the change in the yield spread (the 10Y treasury bond rate minus the 3M treasury bond rate).

For what concerns the financial time series:
* they must contain enough observations to run consistent calculations (a minimum of 252 observations for at least 5 firms is recommended);
* they must have been previously validated and preprocessed by discarding illiquid series with too many zeroes (unless necessary), detecting and removing outliers, removing rows with NaNs or filling the gaps with interpolation approach;
* returns must expressed on a logarithmic scale, in accordance with all the systemic risk indicators;
* market capitalizations must contain a supplementar observation at the beginning because a one-day lagged version is used in order to calculate weighted averages of probabilistic measures;
* total liabilities values must be rolled forward by at least three months (keeping a daily frequency) in order to simulate the difficulty of renegotiating debt in case of financial distress following the SRISK methodology;
* state variables and groups are optional, hence their respective sheets must be removed from the dataset if the related computations aren't necessary;
* state variables, if defined, must contain a supplementar observation at the beginning because a one-day lagged version is required in order to follow the CoVaR/ΔCoVaR methodology.

## Screenshots

![Probabilistic Measures](https://i.imgur.com/1Q1SQd2.png)

![Network Measures](https://i.imgur.com/NuSHgBO.png)

![Network Graph](https://i.imgur.com/fpEVHPf.png)
