# Systemic Risk

This script calculates and analyses the following systemic risk indicators:

* `CoVaR` and `ΔCoVaR` (Conditional Value-at-Risk) proposed by [Adrian & Brunnermeier (2008)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1269446)
* `MES` (Marginal Expected Shortfall) proposed by [Acharya et al. (2010)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1573171)
* `Network Measures` proposed by [Billio et al. (2011)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1963216)
* `SRISK` (Conditional Capital Shortfall Index) proposed by [Brownlees & Engle (2010)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1611229)

Some of the aforementioned models have been adjusted and improved according to the methodologies described in the [V-Lab Documentation](https://vlab.stern.nyu.edu/docs), which represents a great hub for systemic risk measurement.

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

Datasets must be structured like the default one included in every release of the framework (`Datasets/Example.xlsx`). The latter, based on the US financial sector, defines the following entities:

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
  * Federal Home Loan Mortgage Corp / Freddie Mac (FMCC)
  * Federal National Mortgage Association / Fannie Mae (FNMA)

#### State Variables (6):
* **RESI:** the DJ US Select RESI as a proxy of real estate returns.
* **VIX:** the implied volatility index.
* **TBR3M:** the 3M treasury bill rate.
* **CRESPR:** the change in the credit spread (the BAA corporate bond rate minus the 10Y treasury bond rate).
* **LIQSPR:** the change in the liquidity spread (the 3M treasury bill rate minus the federal funds rate).
* **YIESPR:** the change in the yield spread (the 10Y treasury bond rate minus the 3M treasury bond rate).

#### Notes

* Financial time series must contain the benchmark index and at least 3 firms. They must be based on a daily frequency and contain enough observations to run consistent calculations (a minimum of 253 observations, which translates into a full business year plus an additional observation at the beginning). They must have been previously validated and preprocessed by:
  * discarding illiquid series with too many zeroes (unless necessary);
  * detecting and removing outliers;
  * removing rows with NaNs or filling the gaps through interpolation.
* Returns must be expressed on a logarithmic scale, in accordance with all the systemic risk indicators.
* Market capitalizations and total liabilities must be expressed in the same currency. Following the SRISK methodology, the latter must be rolled forward by at least 3 months in order to simulate the difficulty of renegotiating debt in case of financial distress.
* Data concerning state variables and firm groups are optional, hence their respective sheets must be removed from the dataset if the related computations aren't necessary. Groups are based on key-value pairs where the Name field represents the group names and the Count field represents the number of firms to include in the group. The sum of the Count fields must be equal to the number of firms included in the dataset. For example, the following groups definition:

> Firms in the Returns Sheet: A, B, C, D, E, F, G, H  
> Insurance Companies - 2  
> Investment Banks - 2  
> Commercial Banks - 3  
> Government-sponsored Enterprises - 1

produces the following outcome:

> "Insurance Companies" contains A and B  
> "Investment Banks" contains C and D  
> "Commercial Banks" contains E, F and G  
> "Government-sponsored Enterprises" contains H

## Screenshots

![Probabilistic Measures](https://i.imgur.com/1Q1SQd2.png)

![Network Measures](https://i.imgur.com/NuSHgBO.png)

![Network Graph](https://i.imgur.com/fpEVHPf.png)
