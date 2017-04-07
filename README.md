# Systemic Risk

I created this project in 2015 for my Master of Science thesis at Università Cattolica del Sacro Cuore (Milan, Italy).
It can calculate and analyse the following systemic risk indicators:
* CoVaR (Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* ΔCoVaR (Delta Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* MES (Marginal Expected Shortfall) proposed by Acharya et al. (2010);
* Network Measures proposed by Billio et al. (2011);
* SRISK (Conditional Capital Shortfall Index) proposed by Brownlees & Engle (2014).

## Notes

The dataset file must be a valid Excel spreadsheet located in the root directory and structured exactly like the default one (`dataset.xlsx`). The financial time series must have been previously validated and preprocessed so that:
* there are enough observations to run consistent calculations;
* illiquid series with too many zeroes have been discarded (unless necessary);
* outliers have been detected and removed;
* rows with NaNs have been removed or filled with interpolation;
* the time series of market capitalization must contain a supplementar observation at the beginning because a one-day lag version is used in order to calculate average probabilistic measures;
* in relation to CoVaR/ΔCoVaR methodology, the time series of state variables must contain a supplementar observation at the beginning because the script must apply a one-day lag;
* in relation to SRISK methodology, keeping a daily frequency, the values of total liabilities must be rolled forward by at least three months in order to simulate the difficulty of renegotiating debt in case of financial distress;

## Contributions

If you want to start a discussion about the project, just open an issue.
Contributions are more than welcome, fork and create pull requests as needed.
