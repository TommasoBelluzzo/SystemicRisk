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
* System Identification Toolbox'

## Notes

The dataset file must be a valid Excel spreadsheet structured like the default one (`Datasets/Example.xlsx`). For what concerns the financial time series:
* they must contain enough observations to run consistent calculations (a minimum of 1000 observations is recommended);
* they must have been previously validated and preprocessed by discarding illiquid series with too many zeroes (unless necessary), detecting and removing outliers, removing rows with NaNs or filling the gaps with interpolation approach;
* in accordance with all the systemic risk indicators, returns must expressed on a logarithmic scale;
* market capitalizations must contain a supplementar observation at the beginning because a one-day lagged version is used in order to calculate weighted averages of probabilistic measures;
* total liabilities values must be rolled forward by at least three months (keeping a daily frequency) in order to simulate the difficulty of renegotiating debt in case of financial distress following the SRISK methodology;
* state variables and groups are optional, hence their respective sheets must be removed from the dataset if it's not necessary to take them into account;
* state variables must contain a supplementar observation at the beginning because a one-day lagged version is required in order to follow the CoVaR/ΔCoVaR methodology, and the default dataset defines the following ones:
** 

