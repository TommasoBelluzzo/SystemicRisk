# Systemic Risk

This script calculates and analyses the following risk measures:

* `Cross-Sectional Measures`
  * `CoVaR & Delta CoVaR (Conditional Value-at-Risk)` by [Adrian & Brunnermeier (2008)](https://doi.org/10.2139/ssrn.1269446)
  * `MES (Marginal Expected Shortfall)` by [Acharya et al. (2010)](https://doi.org/10.2139/ssrn.1573171)
  * `SRISK (Conditional Capital Shortfall Index)` by [Brownlees & Engle (2010)](https://doi.org/10.2139/ssrn.1611229)
  * `Idiosyncratic Metrics: Beta, Value-at-Risk & Expected Shortfall`
* `Connectedness Measures` proposed in [Billio et al. (2011)](https://doi.org/10.2139/ssrn.1963216)
  * `Dynamic Causality Index`
  * `"In & Out" Connections`
  * `"In & Out - Other" Connections`
  * `Network Centralities`
* `Spillover Measures` proposed in [Diebold & Yilmaz (2014)](https://doi.org/10.1016/j.jeconom.2014.04.012)
  * `Spillover Index`
  * `Spillovers From & To`
  * `Net Spillovers`
* `Component Measures`
  * `Absorption Ratio` by [Kritzman et al. (2010)](https://doi.org/10.2139/ssrn.1633027)
  * `Correlation Surprise` by [Kinlaw & Turkington (2012)](https://doi.org/10.2139/ssrn.2133396)
  * `Turbulence Index` by [Kritzman & Li (2010)](https://doi.org/10.2469/faj.v66.n5.3)
  * `Principal Component Analysis`

Some of the aforementioned models have been adjusted and improved according to the methodologies described in the [V-Lab Documentation](https://vlab.stern.nyu.edu/docs), which represents a great source for systemic risk measurement.

_`The project has been published in "MATLAB Digest - Financial Services" of May 2019.`_

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

Datasets must be built following the structure of default ones included in every release of the framework (see `Datasets` folder). The main one (`Datasets\Example_Main.xlsx`), based on the US financial sector, defines the following entities over a period of time ranging from `2002` to `2019`:

#### Benchmark Index: S&P 500

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
  * Federal Home Loan Mortgage Corp / Freddie Mac (FMCC)
  * Federal National Mortgage Association / Fannie Mae (FNMA)

#### State Variables (8):
* **TBILL_DELTA:** the percent change in the 3M treasury bill rate.
* **CREDIT_SPREAD:** the difference between the BAA corporate bond rate and the 10Y treasury bond rate.
* **LIQUIDITY_SPREAD:** the difference between the 3M treasury bill rate and the federal funds rate.
* **TED_SPREAD:** the difference between the 3M USD LIBOR rate and the 3M treasury bill rate.
* **YIELD_SPREAD:** the difference between the 10Y treasury bond rate and the 3M treasury bond rate.
* **DJ_CA:** the excess returns between the S&P 500 and the DJ US Composite Average.
* **DJ_RESI:** the excess returns between the S&P 500 and the DJ US Select Real Estate Securities Index.
* **VIX:** the implied volatility index.

#### Notes

* The minimum allowed dataset must define financial time series of shares for a benchmark index and at least 3 firms. Observations must have a daily frequency and, in order to run consistent calculations, their minimum required amount is 253, which translates into a full business year plus an additional observation at the beginning of the period. They must have been previously validated and preprocessed by:
  * discarding illiquid series (unless necessary);
  * detecting and removing outliers;
  * removing rows with NaNs or filling the gaps through interpolation.

* It is not mandatory to include financial time series used by measures that are excluded from computations. Optional financial time series used by measures that are included in computations can be omitted, as long as their related contributions aren't necessary. Below a list of required and optional time series for every category of measures:
  * **Cross-Sectional Measures:**
    * *Required:* shares, market capitalization, assets, equity.
    * *Optional:* separate accounts, state variables.
  * **Connectedness Measures:**
    * *Required:* shares.
    * *Optional:* market capitalization, groups.
  * **Spillover Measures:**
    * *Required:* shares.
    * *Optional:* none.
  * **Component Measures:**
    * *Required:* shares.
    * *Optional:* none.
  
* In accordance with all the systemic risk indicators, returns are expressed on a logarithmic scale. Data concerning market capitalization, assets, equity and separate accounts, if present, must be expressed in the same currency and scale. Following the SRISK methodology, liabilities are rolled forward by at least 3 months in order to simulate the difficulty of renegotiating debt in case of financial distress.
* Groups are based on key-value pairs where the `Name` field represents the group names and the `Count` field represents the number of firms to include in the group. The sum of the `Count` fields must be equal to the number of firms included in the dataset. For example, the following groups definition:

  > Firms in the Shares Sheet: A, B, C, D, E, F, G, H  
  > Insurance Companies: 2  
  > Investment Banks: 2  
  > Commercial Banks: 3  
  > Government-sponsored Enterprises: 1

  produces the following outcome:

  > "Insurance Companies" contains A and B  
  > "Investment Banks" contains C and D  
  > "Commercial Banks" contains E, F and G  
  > "Government-sponsored Enterprises" contains H
  
* While stochastic measures are very fast to compute, for huge datasets like `Datasets\Example_Large.xlsx` connectedness and spillover measures may take very long time to finish. The performance of computations may vary from machine to machine, depending on the CPU processing speed and the number of cores available for parallel computing.

## Screenshots

![Cross-Sectional Measures](https://i.imgur.com/VxmTnEs.png)

![Connectedness Measures](https://i.imgur.com/yFBndPc.png)

![Network](https://i.imgur.com/rTnsYxa.png)

![Spillover Measures](https://i.imgur.com/jYCCoQr.png)

![Component Measures](https://i.imgur.com/m11XsbX.png)
