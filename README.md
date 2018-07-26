# Systemic Risk

This script can calculate and analyse the following systemic risk indicators:
* CoVaR (Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* ΔCoVaR (Delta Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* MES (Marginal Expected Shortfall) proposed by Acharya et al. (2010);
* Network Measures proposed by Billio et al. (2011);
* SRISK (Conditional Capital Shortfall Index) proposed by Brownlees & Engle (2014).

## Requirements

The minimum Matlab version required is R2014a. In addition, the following products and toolboxes are required in order to properly execute the script:
* Computer Vision System Toolbox
* Curve Fitting Toolbox
* MATLAB Distributed Computing Server
* Optimization Toolbox
* Parallel Computing Toolbox
* Simulink Control Design
* Statistics and Machine Learning Toolbox
* System Identification Toolbox'

## Notes

The dataset file must be a valid Excel spreadsheet located in the root directory and structured as the default one (`Datasets/Example.xlsx`). For what concerns the financial time series:
* they must contain enough observations to run consistent calculations;
* they must have been previously validated and preprocessed discarding illiquid series with too many zeroes (unless necessary), detecting and removing outliers, removing rows with NaNs or filling the gaps with interpolation;
* market capitalizations must contain a supplementar observation at the beginning because a one-day lagged version is used in order to calculate weighted averages of probabilistic measures;
* total liabilities values must be rolled forward by at least three months (keeping a daily frequency) in order to simulate the difficulty of renegotiating debt in case of financial distress following the SRISK methodology;
* state variables (optional) must contain a supplementar observation at the beginning because the script must apply a one-day lag following the CoVaR/ΔCoVaR methodology.
