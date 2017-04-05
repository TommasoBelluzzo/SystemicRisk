# Systemic Risk

I created this project in 2015 for my Master of Science thesis at Università Cattolica del Sacro Cuore (Milan, Italy).
It can calculate and analyse the following systemic risk indicators:
* CoVaR (Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* ΔCoVaR (Delta Conditional Value-at-Risk) proposed by Adrian & Brunnermeier (2009);
* MES (Marginal Expected Shortfall) proposed by Acharya et al. (2010);
* Network Measures proposed by Billio et al. (2011);
* SRISK (Conditional Capital Shortfall Index) proposed by Brownlees & Engle (2014).

## Documentation

IO functions (like `get_firm_returns`, `get_state_variables`, `write_result`, etc...) have not been included in the project, so you have to implement them on your own if you want to use it. After all, many different approaches and data providers can be used to achieve the same result. You have to take care of dataset validation and your IO functions must preprocess the financial time series in order to provide correctly formatted data.

## Contributions

If you want to start a discussion about the project, just open an issue.
Contributions are more than welcome, fork and create pull requests as needed.
