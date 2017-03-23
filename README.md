# Systemic Risk

I created this project for my Master of Science thesis at Università Cattolica del Sacro Cuore (Milan, Italy) in 2015.
It can be used to calculate the following systemic risk indicators:
* CoVaR (Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* MES (Marginal Expected Shortfall) proposed by Acharya et al. (2010);
* Network Measures proposed by Billio et al. (2011);
* SRISK (Conditional Capital Shortfall Index) proposed by Brownlees & Engle (2014).

## Documentation

IO functions (like `get_firm_returns`, `get_state_variables`, `write_result`, etc...) have not been included in the project, so you have to implement them on your own. Keep in mind that



% The financial time series in the dataset must have been previously validated:
%  - illiquid series with too many zeroes have been discarded;
%  - rows with NaNs have been removed or filled with interpolation;
%  - there are enough observations to run consistent calculations;
%  - etc...





## Contributions

If you want to start a discussion about the project, just open an issue.
Contributions are more than welcome, fork and create pull requests as needed.
