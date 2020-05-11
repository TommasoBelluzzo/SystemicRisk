# Systemic Risk

This script calculates and analyses the following risk measures:

* **COMPONENT MEASURES**
  * `Absorption Ratio` by [Kritzman et al. (2010)](https://doi.org/10.2139/ssrn.1633027)
  * `CATFIN` by [Allen et al. (2012)](https://doi.org/10.1093/rfs/hhs094)
  * `Correlation Surprise` by [Kinlaw & Turkington (2012)](https://doi.org/10.2139/ssrn.2133396)
  * `Turbulence Index` by [Kritzman & Li (2010)](https://doi.org/10.2469/faj.v66.n5.3)
  * `Principal Component Analysis`
* **CONNECTEDNESS MEASURES**
  * `DCI (Dynamic Causality Index)` by [Billio et al. (2011)](https://doi.org/10.2139/ssrn.1963216)
  * `"In & Out" Connections` by [Billio et al. (2011)](https://doi.org/10.2139/ssrn.1963216)
  * `"In & Out - Other" Connections` by [Billio et al. (2011)](https://doi.org/10.2139/ssrn.1963216)
  * `Network Centralities: Betweenness, Degree, Closeness, Clustering, Eigenvector & Katz`
* **CROSS-QUANTILOGRAM MEASURES**
  * `Full Cross-Quantilograms` by [Han et al. (2016)](https://doi.org/10.1016/j.jeconom.2016.03.001)
  * `Partial Cross-Quantilograms` by [Han et al. (2016)](https://doi.org/10.1016/j.jeconom.2016.03.001)
* **CROSS-SECTIONAL MEASURES**
  * `CoVaR & Delta CoVaR (Conditional Value-at-Risk)` by [Adrian & Brunnermeier (2008)](https://doi.org/10.2139/ssrn.1269446)
  * `MES (Marginal Expected Shortfall)` by [Acharya et al. (2010)](https://doi.org/10.2139/ssrn.1573171)
  * `SES (Systemic Expected Shortfall)` by [Acharya et al. (2010)](https://doi.org/10.2139/ssrn.1573171)
  * `SRISK (Conditional Capital Shortfall Index)` by [Brownlees & Engle (2010)](https://doi.org/10.2139/ssrn.1611229)
  * `Idiosyncratic Metrics: Beta, Value-at-Risk & Expected Shortfall`
* **DEFAULT MEASURES**
  * `D2C (Distance To Capital)` by [Chan-Lau & Sy (2007)](https://doi.org/10.1057/palgrave.jbr.2350056)
  * `D2D (Distance To Default)` by [Vassalou & Xing (2004)](https://doi.org/10.1111/j.1540-6261.2004.00650.x)
  * `DIP (Distress Insurance Premium)` by [Black et al. (2012)](https://doi.org/10.2139/ssrn.2181645)
  * `SCCA (Systemic Contingent Claims Analysis)` by [Jobst & Gray (2013)](https://doi.org/10.5089/9781475572780.001)
* **LIQUIDITY MEASURES**
  * `ILLIQ (Illiquidity Measure)` by [Amihud (2002)](https://doi.org/10.1016/S1386-4181(01)00024-6)
  * `RIS (Roll Implicit Spread)` by [Hasbrouck (2009)](https://doi.org/10.1111/j.1540-6261.2009.01469.x)
  * `Classic Indicators: Hui-Heubel Liquidity Ratio, Turnover Ratio & Variance Ratio`
* **SPILLOVER MEASURES**
  * `SI (Spillover Index)` by [Diebold & Yilmaz (2014)](https://doi.org/10.1016/j.jeconom.2014.04.012)
  * `Spillovers From & To` by [Diebold & Yilmaz (2014)](https://doi.org/10.1016/j.jeconom.2014.04.012)
  * `Net Spillovers` by [Diebold & Yilmaz (2014)](https://doi.org/10.1016/j.jeconom.2014.04.012)

Some of the aforementioned models have been improved or extended according to the methodologies described in the [V-Lab Documentation](https://vlab.stern.nyu.edu/docs), which represents a great source for systemic risk measurement.

_`The project has been published in "MATLAB Digest - Financial Services" of May 2019.`_

## Requirements

The minimum required Matlab version is `R2014b`. In addition, the following products and toolboxes must be installed in order to properly execute the script:

* Computer Vision System Toolbox
* Curve Fitting Toolbox
* Econometrics Toolbox
* Financial Toolbox
* Image Processing Toolbox
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

Datasets must be built following the structure of default ones included in every release of the framework (see `Datasets` folder). Below a list of the supported Excel sheets and their respective content:

* **Shares:** prices or returns expressed on a logarithmic scale of the benchmark index (the column can be labeled with any desired name and must be placed just after observation dates) and the firms, with daily frequency.
* **Volumes:** trading volume of the firms expressed in currency amount, with daily frequency.
* **Capitalizations:** market capitalization of the firms, with daily frequency.
* **CDS:** the risk-free rate expressed in decimals (the column must be called `RF` and must be placed just after observation dates) and the credit default swap spreads of the firms expressed in basis points, with daily frequency.
* **Balance Sheet Components:** the balance sheet components of the firms expressed in omogeneous currency and scale, with the specified observations frequency, structured as below:
  * **Assets:** the book value of assets.
  * **Equity:** the book value of equity.
  * **Separate Accounts:** the separate accounts of insurance firms.
* **State Variables:** systemic state variables, with daily frequency.
* **Groups:** group definitions, based on key-value pairs where the `Name` field represents the group names and the `Count` field represents the number of firms to include in the group. The sum of the `Count` fields must be equal to the number of firms. For example, the following groups definition:

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

The main dataset (`Datasets\Example_Large.xlsx`), based on the US financial sector, defines the following entities and data over a period of time ranging from `2002` to `2019` (both included):

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

#### Risk-Free Rate: 3M Treasury Bill Rate

#### State Variables (8):
* **FFR:** the effective federal funds rate rate.
* **TBILL_DELTA:** the percent change in the 3M treasury bill rate.
* **CREDIT_SPREAD:** the difference between the BAA corporate bond rate and the 10Y treasury bond rate.
* **LIQUIDITY_SPREAD:** the difference between the 3M GC repo rate and the 3M treasury bill rate.
* **TED_SPREAD:** the difference between the 3M USD LIBOR rate and the 3M treasury bill rate.
* **YIELD_SPREAD:** the difference between the 10Y treasury bond rate and the 3M treasury bond rate.
* **DJ_CA_EXC:** the excess returns of the DJ US Composite Average with respect to the S&P 500.
* **DJ_RESI_EXC:** the excess returns of the DJ US Select Real Estate Securities Index with respect to the S&P 500.
* **VIX:** the implied volatility index.

#### Notes

* The minimum allowed dataset must include the `Shares` sheet with a benchmark index and at least `3` firms. Observations must have a daily frequency and, in order to run consistent calculations, their minimum required amount is `253`, which translates into a full business year plus an additional observation at the beginning of the time series. They must have been previously validated and preprocessed by:
  * discarding illiquid series (unless necessary);
  * detecting and removing outliers;
  * removing rows with NaNs or filling the gaps through interpolation.

* It is not mandatory to include financial time series used by unwanted measures. Optional financial time series used by included measures can be omitted, as long as their contribution isn't necessary. Below a list of required and optional time series for each category of measures:

  * **Component Measures:**
    * *Required:* shares (any).
    * *Optional:* none.
  * **Connectedness Measures:**
    * *Required:* shares (any).
    * *Optional:* groups.
  * **Cross-Quantilogram Measures:**
    * *Required:* shares (any).
    * *Optional:* state variables.
  * **Cross-Sectional Measures:**
    * *Required:* shares (any), capitalizations, assets, equity.
    * *Optional:* separate accounts, state variables.
  * **Default Measures:**
    * *Required:* shares (any), capitalizations, cds, assets, equity.
    * *Optional:* none.
  * **Liquidity Measures:**
    * *Required:* shares (prices), volumes, capitalizations.
    * *Optional:* state variables.
  * **Spillover Measures:**
    * *Required:* shares (any).
    * *Optional:* none.
  
* Firms whose `Shares` value is constantly null in the tail of the time series, for a span that includes at least `5%` of the total observations, are considered to be `defaulted`. Firms whose `Equity` value is constantly null in the tail of the time series, for a span that includes at least `5%` of the total observations, are considered to be `insolvent`. This allows the scripts to exclude them from computations starting from a certain time point onward; defaulted firms are excluded by all the measures, while insolvent firms are excluded only by SCCA default measures.

* If the dataset parsing process is too slow, the best way to solve the issue is to provide a standard unformatted Excel spreadsheet (`.xlsx`) or a binary Excel spreadsheet (`.xlsb`). Once a dataset has been parsed, the script stores its output so that the parsing process happens only at first run.

* Some scripts may take very long time to finish in presence of huge datasets and/or extreme parametrizations. The performance of calculations may vary depending on the CPU processing speed and the number of CPU cores available for parallel computing.

## Screenshots

![Cross-Sectional Measures](https://i.imgur.com/VxmTnEs.png)

![Connectedness Measures](https://i.imgur.com/yFBndPc.png)

![Network](https://i.imgur.com/rTnsYxa.png)

![Spillover Measures](https://i.imgur.com/jYCCoQr.png)

![Component Measures](https://i.imgur.com/m11XsbX.png)

## Donation

If you find this project useful to you, please consider making a donation to support it.

[![PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=ASMXC3LYNV96J)
