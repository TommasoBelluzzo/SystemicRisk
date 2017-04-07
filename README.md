# Systemic Risk

I created this project in 2015 for my Master of Science thesis at Università Cattolica del Sacro Cuore (Milan, Italy).
It can calculate and analyse the following systemic risk indicators:
* CoVaR (Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* ΔCoVaR (Delta Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* MES (Marginal Expected Shortfall) proposed by Acharya et al. (2010);
* Network Measures proposed by Billio et al. (2011);
* SRISK (Conditional Capital Shortfall Index) proposed by Brownlees & Engle (2014).

## Notes

The dataset file must be a valid Excel spreadsheet located in the root directory and structured exactly like the default one (`dataset.xlsx`). For what concerns the financial time series:
* they must contain enough observations to run consistent calculations;
* they must have been previously validated and preprocessed discarding illiquid series with too many zeroes (unless necessary), detecting and removing outliers, removing rows with NaNs or filling the gaps with interpolation;
* market capitalizations must contain a supplementar observation at the beginning because a one-day lagged version is used in order to calculate average probabilistic measures;
* state variables (optional) must contain a supplementar observation at the beginning because the script must apply a one-day lag following the CoVaR/ΔCoVaR methodology;
* total liabilities values must be rolled forward by at least three months (keeping a daily frequency) in order to simulate the difficulty of renegotiating debt in case of financial distress following the SRISK methodology.

## Contributions

If you want to start a discussion about the project, just open an issue.
Contributions are more than welcome, fork and create pull requests as needed.
