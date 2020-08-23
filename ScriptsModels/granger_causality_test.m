function [ F, c_v,  Fprob , Fprob_Corrected,   dAIC, dBIC  , chosen_x_lag, chosen_y_lag ] ...
          =  ...
         granger_causality_test(x, y, alpha, max_x_lag, use_best_x, ...
                                      max_y_lag, use_best_y, firstYlag ,  ...
         xName, yName , data_date , HomeDir ,Fprefix, Ptitle , PlotControl )
%
% Granger Causality test
% Does Y Granger Cause X?  
%
% 1) find a regression equation  to predict X based on past values of the  X's
%     this is also called the RESTRICTED model
%
% 2) find a regression equation  to predict X based on past values of the  X's
%      and the past values of the Y's ( and optionally contemporaneous values of Y)
%     this is also called the UN-restricted model
%
%  we test to see if the second is any better than the first
%
%   NaN's  are OK but reduce power of test of course
%
%  this routine uses the matlab regress that has an intercept,
%     i have read some comments that you should not use an intercept.
%     do not understand the pros and cons of this, 
%
%
% User-Specified Inputs:
%   x           -- A column vector of data  :: a series that y may be able to predict
%   y           -- A column vector of data   :: a series that is trying to predict x
%  alpha        --  probability level ( 0.01 , ... ) of critical value of F
%   max_x_lag   -- the maximum number of lags to be computedfor the
%                       initial X model.
%   use_best_x  -- search the various x-lagged models and retain the one
%                   with the lowest BIC score
%   max_y_lag   -- the maximum number of lags to be computed for the
%                       second phase Y model.
%   use_best_y  -- search the various y-lagged models and retain the one
%                   with the lowest BIC score
%   firstYlag   -- 0  means use Y(lag=0) ( the contemporaneous value)
%                   as well as lags 1 .. max_lag 
%                   otherwise 1 for using only Y(lag=1 ,...  max_lag)
%
% the next input arguments are required, but only used if you plot the data 
%  xName         -- the name of the X variable
%  yName         -- the name of the Y variable
% data_date      -- the values of time for x and y
%                   ( what date/time is sample 1 .. N 
%                   if 0, will set to sample #
%  homeDir       --  directory to write plot png file into
% Fprefix        -- ID string at the start of the file name
%  Ptitle        --  title of the plot
% PlotControl    -- how to decide if there will be a plot made
%                   0 =  NO  ;  1 = yes ;  2 = automatically decide based
%                   on BIC
%    
%
%  Output:
%
%
%
%   F  -- The value of the F-statistic of the nested model pair
%          
%              the F is for the relative improvement going from the restricted model to 
%               the unrestricted model 
%   c_v   -- alpha level critical value
%        Note that if F > c_v we reject the null hypothesis that y does not
%           Granger Cause x for the case of fixed lags - 
%
%   Fprob   --  probability of obtaining F by random chance
%                   - this is correct if you DO NOT choose the
%                        best y_lag model option 
%
%
%  Fprob_Corrected  --  (poor) attempt to correct for searching the lags to
%                   find the best one ( the multiple comparison problem ) 
%                    ad hoc, work in progress
%
%    dBIC  : change in BIC between two models
%    dAIC  : change in the AIC between the two models
%
%     AIC = n Ln(RSS/n) + 2 p 
%     BIC = n Ln(RSS/n) +   p Ln(n)
%     
%
%
%
%  chosen_x_lag, chosen_y_lag  :  BIC-chosen best lag for models if
%               use_best_(X or Y) is 1,
%         else this will be the max_lag of the model
%
% Note: The best lag length selection is chosen using the Bayesian information
% Criterion
%
%
% show us the internal workings of the code
%
%
%   0  ==>  quiet
%   1 ==> summary of F tests
%   2 ==>  show internal structures ( mostly for development )
%
DEBUG = 0  ; 
%
%  R. Boldi  2014-2016
% Zayed University, Dubia, UAE
%
% based on granger_cause by Chandler Lutz, UCR 2009
%
% R. Boldi :
%
% --  modified to add 0 lag if desired:
% --  permit missing data
% --  plot data if requested
% --  fixed failure of granger_cause to select all possible rows of model
%       this bug may be related to the observation that granger_cause rejects
%       the null hypothesis too often - it had wrong number of degrees of freedom
%       relative to the number of observation in the model:
%
% -- if you select the BEST_Y_LAG, this involves multiple comparisons 
%      and we need to adjust the probability levels or else
%       granger_cause_1 will reject Null hypothesis too often   
%       
%  -- corerction   Family-Wise_Error_Rate
%
%          alpha_Bar  = 1 - ( 1 - alpha)^k
%         
%      we  want alpha_bar = 0.05 ( standard p value )
%
%   Prob Corrected ( p, k ) = 1 - ( 1 - p )^k
%      k = 4 lags
%    raw p = 0.0127 ==>> corrected p = 0.0498  
%   this is for independent tests, ours are dependent ,
%     i have a totally ad-hoc correction
%       if someone has a real method please let me know!!!
%
%
% Chandler Lutz, UCR 2009
% Questions/Comments: chandler.lutz@email.ucr.edu
% $Revision: 1.0.0 $  $Date: 09/30/2009 $
% $Revision: 1.0.1 $  $Date: 10/20/2009 $
% $Revision: 1.0.2 $  $Date: 03/18/2009 $
%
%
% Acknowledgements (of Chandler Lutz) :
%   I would like to thank Mads Dyrholm for his helpful comments and
%   suggestions
%
%  note by R.Boldi 2014
%
%  note that the BIC and F-test are not exactly the same , meaning that
%  at small N, and using gaussian random inputs, the F-test rejects
%   the null hypotheses LESS often than it should.  This effect is a function of
%  lag and N, and seems to go away for N > 1000 .
%
%   see reference [2] for more details on this
%
%
%
% References:
% [1] Granger, C.W.J., 1969. 'Investigating causal relations by econometric
%     models and cross-spectral methods'. Econometrica 37 (3), 424?438.
%
% [2] The relationship between the F-test and the Schwarz criterion: Implications for
%      Granger-causality tests.   Erdal Atukeren  (2010)
%      ETH Zurich - KOF Swiss Economic Institute
%       economics bulletin,  V30, issue 1
%      no.1 pp. 494-499.
%    Submitted: Jan 12 2010. Published: February 08, 2010.
%
%Make sure x & y are the same length
if (length(x) ~= length(y))
    error('x and y must be the same length');
end
%Make sure x is a column vector
[a,b] = size(x);
if (b>a)
    warning('x is a row vector -- fix this for next time');
    x = x';
end
if ( ( a  > 1 ) && ( b > 1 ))
    error ( ' x must be 1 column ');
end
%Make sure y is a column vector
[a,b] = size(y);
if (b>a)
    warning ('y is a row vector -- fix this for next time');
    y = y';
end
if ( ( a  > 1 ) && ( b > 1 ))
    error ( ' y must be 1 column ');
end
%
%   Make sure max_lag is >= 1
% and
%  be sure the sample size is large enough to support the
%   multi-value models we will build
%
if max_x_lag < 1
    error('max_x_lag must be greater than or equal to one');
end
if max_y_lag < 1
    error('max_y_lag must be greater than or equal to one');
end
%
N = length(x);
%
% as model complexity increases, the N falls as well
%
beta =   max_x_lag + max_y_lag + 1 ;
if (   beta  >= (N - beta)  )
    warning(sprintf('Granger Cause: Too little data for such a big search'));
    warning(sprintf('max_x_lag=%d ; max_y_lag=%d  N = %d results in max vars = %d with N of %d ', ...
                     max_x_lag, max_y_lag,N , beta, (N-beta) ));
    error('granger cause failed');
end
%
% note - it can still fail  based on the amount of missing data
%  or long lists of a constant value
%
tmp_N_Good_x =  length(x( ~isnan(x) ));
tmp_N_Good_y =  length(y( ~isnan(y) ));
tmp_N_Uniq_x =  length(unique(x( ~isnan(x) )));
tmp_N_Uniq_y =  length(unique(y( ~isnan(y) )));
if ( ( tmp_N_Uniq_x < beta/2) || ( tmp_N_Uniq_y < beta/2) )
    warning(sprintf('Granger : too little unique data Ngood( %d,%d ) ;; Nunique(%d,%d) \n',tmp_N_Good_x, tmp_N_Good_y, tmp_N_Uniq_x, tmp_N_Uniq_y));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%    missing data mask 
%
% try for a consistent sample size for all models
%
%  load up for the max model     max_x_lag,  0 .. max_y_lag
%  make a list of what x,y samples survive the complete-row selection
%  void out the others
%
% - this will have the effect of ensuring we compare apples to apples
%
%   to do this we make Ystar and Xstar
%
% if we  zap only x,y we will not get a constant N for all tests
% because missing data now comes from n+i term
% so we zap xstar after we fill it in ~
%
clear xstar ystar
First_Obs =max( max_x_lag, max_y_lag+1-firstYlag) + 1 ;
 Last_Obs = N ;
N_y_lags = max_y_lag + 1 - firstYlag;
%
% allocate space for the X arrays
%  directly create the Y
%
%    Y =  a X 
%   
ystar =     x(First_Obs:Last_Obs) ;
nObs = length(ystar);
xstar = [ones(nObs,1)    zeros(nObs,max_x_lag)     zeros(nObs,max_y_lag + 1 - firstYlag)    ] ;
%
% fill in the past x variables 
%
j = 1;
while j <= max_x_lag
    obs_init =  First_Obs - j ;
    obs_fini =   obs_init + nObs - 1 ;
    xstar(:,j+1) = x(obs_init:obs_fini);
    j = j+1;
end
%
% fill in the (contemporaneous) and past values of Y 
%
%  display('  fill in the (contemporaneous) and past values of Y ');
j = 1;
while j <= N_y_lags
    
    obs_init =  First_Obs -j   + 1 -firstYlag;
    obs_fini =   obs_init + nObs - 1 ;
    tmp_colN = j+1+max_y_lag   ;
    %  display(sprintf('j %3d / %3d; tmp_colN  %3d ; obs_init  %3d; obs_fini %3d ', j, N_y_lags, tmp_colN, obs_init, obs_fini));
    xstar(:,tmp_colN ) = y(obs_init:obs_fini);
    j = j+1;
end
if ( DEBUG > 1 )
YstarA_ = ystar';
XstarA_ = xstar'
end
%
% now that we have our big arrays, 
%   map out missing data
%
BadVals         =    logical  ( ones(length(x),1) )  ;
BadVals(First_Obs:N) =    logical  (sum (isnan ([ystar xstar]), 2));
    % diagnostic to check complex address calculations
    if ( DEBUG > 1 )
        warning(' phase 0 ');
        Ystar_ =  ystar'
        Xstar_ = xstar'
        badVals_ = BadVals'
        warning('- - end phase 0 - ');
    end
    
 
% display(' done with missing value mask ');
%
%   done with missing value mask
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determine the total variance - just to place in the title
%
TotalVar =   nanvar( x ) ;
%
% predict x in terms of the past values of x
%
% this the the Restricted model
%
% we will save the BIC, AIC, F stats,  and RSS
%
AIC_R       = zeros(max_x_lag,1);
BIC_R       = zeros(max_x_lag,1);
RSS_R       = zeros(max_x_lag,1);
Nvalid_R    = zeros(max_x_lag,1);
% in case we plot the models, we save tmp_Yhat_X's
 Yhat_x_arr  = NaN .* zeros(length(x),max_x_lag);
Ystar_x_arr  = NaN .* zeros(length(x),max_x_lag);
%
% loop over the lags
%
i = 1;
while i <= max_x_lag
    
    clear xstar ystar
    
    %
    %  we drop  x1 , x2, ... as the lag increases
    %
    xstar = zeros(N-i,i+1);
    ystar = zeros(N-i,1);
    
    % zap missing data    
    tmp_x = x ;
    tmp_x(BadVals) = NaN ;
    
    ystar =  tmp_x(i+1:N);             % drop early data - would need x(-1) to predict
    xstar = [ones(N-i,1) zeros(N-i,i)];    % we use intercept
    
    % Populate the xstar matrix with the corresponding vectors of lags
    j = 1;
    while j <= i
        xstar(:,j+1) = x(i+1-j:N-j);
        j = j+1;
    end
    
    
    % Apply the ordinary least squares linear regression
    %  regress function.  we want just the residuals  r
    
    %  warning(sprintf('GrangerP1_1 %d [ %d %d ]  [ %d %d ] ', i, size(xstar) , size(ystar) ));
    if ( DEBUG > 1 )
    Ystar_ = ystar'
    Xstar_ = xstar'
    
    display('A');
    end
    
    
    
    [b,bint,r ] = regress(ystar,xstar) ;  
    %
    % remove NaN's from r
    %
    r(find(isnan(r))) = []; 
       
    % how many valid samples are in ystar
    yvalid = ystar ;
    nans = (isnan(ystar) | any(isnan(xstar),2));
    if any(nans)
        yvalid(nans) = [];
    end
    Nvalid  = length(yvalid);
    
    % diagnostic to check complex address calculations
    if ( DEBUG > 1 )
        warning(' phase 1 ');
        i
        ystar'
        xstar'
        warning('- - end phase 1 - ');
    end
    
 
    
    % Find and save the bayesian information criterion
    % this will judge how good the fit is relative to other models
    % do we get any NaN's back from regress ?
    %
    sse  = r'*r ;
    AIC_R(i)      = Nvalid*log(sse/Nvalid)  +  2*(i+1)  ;
    BIC_R(i)      = Nvalid*log(sse/Nvalid)  +    (i+1)*log(Nvalid);
    RSS_R(i)      = sse ;
    Nvalid_R(i)   = Nvalid ;
    
    
    
% in case we plot the models, we save tmp_Yhat_X's    
    Ystar_x_arr(1+i:length(ystar)+i,i) =  ystar ;
     Yhat_x_arr(1+i:length(ystar)+i,i) =  xstar * b ;
    
    i = i+1;
end
%
% do we pick the best X-nodel or the specified number of x-lags
%
if (use_best_x > 0 )
    [~ ,x_lag] = min(BIC_R);
else
    x_lag = max_x_lag ; 
end
chosen_x_lag = x_lag ; 
if (  DEBUG > 0 )
    display(sprintf(' Chosen X lag = %d , BIC_R  = %8.3E', x_lag, BIC_R(x_lag)));
    if ( DEBUG > 1 )
        Ystar_x_arr'
        Yhat_x_arr'
    end
    
end
%
%
%
%
%  phase 2  ::  predict x in terms of the past values of y  ( and the best x model )
%
%
%
%
%
%
% Specify a matrix for the unrestricted RSS
%
% allocate  BIC and unrestricted SS arrays
%
AIC_U    = zeros(max_y_lag+1-firstYlag,1);
BIC_U    = zeros(max_y_lag+1-firstYlag,1);
RSS_U    = zeros(max_y_lag+1-firstYlag,1);
Nvalid_U = zeros(max_y_lag+1-firstYlag,1);
% in case we plot the models, we save tmp_Yhat_X's
 Yhat_y_arr  = NaN .* zeros(length(x),max_y_lag+1-firstYlag);
Ystar_y_arr  = NaN .* zeros(length(x),max_y_lag+1-firstYlag);
%
% we can use the contemporaneous values of y as well as its past
%   if we want,
%   firstLag [ 0 or 1 ]
i =  firstYlag ;
while i <= max_y_lag
    
    clear xstar ystar
    
    % max effective lag
    % set by max of the two lags,  x_lag and i
    % this sets how many usable entries we have given N
    % you can predict x   from x(elag+1) to x(N)
    %  this gives an N of  N - (elag+1) + 1 = N - elag
    
    eLag  = max(i,x_lag) ;
    
    nObs = N    - eLag ;
    First_Obs = eLag + 1 ;
    
    
    tmp_x = x ;
    tmp_x(BadVals) = NaN ;   %   zap missing data
    
    ystar =   tmp_x(First_Obs:N) ;
    
    %
    % allocate the model coefficients,                   number of new terms
    %         constant        previous model        this model, might have i=0
    %
    xstar = [ones(nObs,1)    zeros(nObs,x_lag)     zeros(nObs,i + 1 - firstYlag)    ] ;
    
    
    % Populate the xstar matrix with the corresponding vectors of lags of x
    % that came from the above best model of x using past values of x
    %  eLag matters here
    j = 1;
    while j <= x_lag
        obs_init =  First_Obs - j ;
        obs_fini =   obs_init + nObs - 1 ;
        %warning(sprintf('%d gets %d : %d',j+1, obs_init, obs_fini ));
        xstar(:,j+1) = x(obs_init:obs_fini);
        j = j+1;
    end
    
    
    %
    %Populate the xstar matrix with the corresponding vectors of lags of y
    %  if i is 0, we are using contemporaneous value of y as well as the lags
    j = 1;
    while j <= i+ 1 -firstYlag
        % add 1 if we are doing y(0)
        obs_init =  First_Obs -j   + 1 -firstYlag;
        obs_fini =   obs_init + nObs - 1 ;
        
        tmp_colN = j+1+x_lag   ;
        
        %warning(sprintf('%d GetY %d : %d',tmp_colN, obs_init, obs_fini ));
        
        xstar(:,tmp_colN ) = y(obs_init:obs_fini);
        j = j+1;
    end
    
    
    % Apply the ordinary least squares linear regression
    %  regress function.  we want just the residuals  r
    % 
    [b,bint,r ] = regress(ystar,xstar);
    
    %
    % remove NaN's from r
    %
    r(find(isnan(r))) = []; 
    % how many valid samples are in ystar
    yvalid = ystar ;
    nans = (isnan(ystar) | any(isnan(xstar),2));
    if any(nans)
        yvalid(nans) = [];
    end
    Nvalid  = length(yvalid);
    
    
    if ( DEBUG > 1 )
        display('Phase 2 Diags ');
        i
        Xstar_ = xstar'
        Ystar_ = ystar'
        display('-- end phase 2  --');
    end
    
    % Find and save the bayesian information criterion
    %  and the UN.restricted RSS
    %  the BIC here may be for NON-nested models
    %  we have   nested models
    %
    lag_index = i + 1 - firstYlag ;
    sse  = r'*r ;
    AIC_U(lag_index)      = Nvalid*log(sse/Nvalid) +  2*(i+1+x_lag)  ;
    BIC_U(lag_index)      = Nvalid*log(sse/Nvalid) +    (i+1+x_lag)*log(Nvalid);
    RSS_U(lag_index)      = sse ;
    Nvalid_U(lag_index)   = Nvalid ;
 
    
   % in case we plot the models, we save tmp_Yhat_X's 
    Ystar_y_arr(First_Obs:N,i+1-firstYlag) =  ystar     ;
     Yhat_y_arr(First_Obs:N,i+1-firstYlag) =  xstar * b ;
    
    i = i+1;
end
%
%  choose the reference y lag model 
%
%
% if first y_lag is 0, then y_lag is an index, the actual lag number
%   is y_lag - 1
%
if ( use_best_y > 0 )
[~ ,y_lag] = min(BIC_U);
else
    y_lag = length(BIC_U);   % this is the max_y_lag of model
end
chosen_y_lag = y_lag ; 
if ( DEBUG > 0 )
    display(sprintf(' chosen y lag = %d , BIC_U  = %8.3E', y_lag, BIC_U(y_lag)));
    if ( DEBUG > 1 )
        Ystar_y_arr'
        Yhat_y_arr'
        end
end
% The numerator of the F-statistic
%
% how many params did we add to the reduced
%
%  y_lag, some combination of y0 and y-1, ...
%
%
% these come from different sized models , due to missing data
%
% at this point,  
% y_lag  = chosen_y_lag
% x_lag  = chosen_x_lag 
%
%
% this is the one we care about
%
if (Nvalid_U(y_lag) == Nvalid_R(x_lag) )
    % no missing data
    % warning(' N constant = %d  ',  Nvalid_U(y_lag)   );
    F_num =  ( RSS_R(x_lag,:)  -   RSS_U(y_lag,:) ) /  y_lag     ;
    Tave = Nvalid_U(y_lag) ;
else
    % we have missing data
    Tave = ( Nvalid_U(y_lag) + Nvalid_R(x_lag) )  / 2.0 ;
    tmp_MSS_R =  RSS_R(x_lag) / Nvalid_R(x_lag);
    tmp_MSS_U =  RSS_U(y_lag) / Nvalid_U(y_lag);
    
    F_num  =   ( Tave * ( tmp_MSS_R -  tmp_MSS_U) ) / y_lag ;
    
    warning(' N  %d  vs  %d = %8.2e',  Nvalid_U(y_lag) , Nvalid_R(x_lag) , F_num );
    
end
%
%  coefficient counts as one
%  and do not forget contemporaneous Y
%
% The denominator    of the F-statistic
%
den_DF_lags = x_lag+y_lag+1 ;  % ylag has the 0 if needed already counted + ( 1 - firstYlag) ;
dfDen = (Nvalid_U(y_lag)-den_DF_lags) ;
F_den = RSS_U(y_lag,:)/dfDen;
 
% these can be negative somehow?
%
if (F_num < 0 )
    warning(sprintf('F num is negative %8.3e', F_num));
end
if (F_den < 0 )
    warning('F den  is negative %8.3E', F_den);
end
F_num = max( F_num , 0 );
F_den = max( F_den , 0 );
%
%   F-test
%
%  some testing with random data seems to show that
%   5% rejection level happens more or less often based
%    on the number of lags and N  - under investigation
%
%
F = 1.0;
Fprob = 0.50;
c_v = 1.0 ;
%
%  note on alpha
%
%  generally [  0.10  to 0.05   to 0.01 ]
% fcdf returns area under curve to the left of F
%   finv returns value greater then 1-alpha of the observation
if (( F_den > 0.0 ) && (dfDen > 0) )
    
    F = F_num/F_den;
    
    
    Fprob = 1 -  fcdf(F,      (y_lag), dfDen );
    
    c_v   =  finv(1-alpha ,   (y_lag) , dfDen );
    
end;
%
if ( DEBUG > 0)
 fprintf('\nfcdf( %8.4f , %3d , %3d ) = P( %8.4f ) cv =  %8.4f \n\n', ...
                      F,y_lag, (Nvalid_U(y_lag)-(x_lag+y_lag+1)) , Fprob, c_v  );
end
%
%
% also look at delta BIC
% ad hoc - approximate probability of occurence
%
%  smaller is better, the U should be smaller than R
%  so a negative number means  ' y granger  causes x '
%
% this is our chosen dAIC, dBIC
%
 dBIC =    ( BIC_U(y_lag) - BIC_R(x_lag) ) ;
 dAIC =    ( AIC_U(y_lag) - AIC_R(x_lag) ) ;
  
 
 
 
%%%%%%%%%%%%%%%%%%%%%
%
%   if we chose to use the BEST y, we
%    searched a column for the lowest probability
% ---  need to correct for this
%
%  for INDEPENDENT comparisons -
%   pretty sure ours are DEpendent
%
%   Prob Corrected ( p, k ) = 1 - ( 1 - p )^k
%      k = 4 lags
%    raw p = 0.0127 ==>> corrected p = 0.0498  
%
% seems only necessary for the Y
%
if (use_best_y > 0 )
    % this is complete AD-HOC - VERY LITTLE THEORY 
    %  behind the value of the variable comparisons
    %
    %  the comaparisons are not independent so the 
    %  ?id?k correction and the Holm?Bonferroni method 
    % are way too consertative
        comparisons = 1 + ((1-Fprob)^2)*log(max_y_lag);
        Fprob_Corrected = 1.0 - ( 1.0 - Fprob)^comparisons  ;
else
    Fprob_Corrected = Fprob ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  make a plot of x best x-only model
%  also make the x and best y
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MakeGrangerPlot = 0;
if (PlotControl == 1)
    MakeGrangerPlot = 1;
end
 
if ( PlotControl == 2 )
    if  ((dBIC < -50.0) || ( Fprob < 0.00125 ))
        MakeGrangerPlot = 1;
    else
        MakeGrangerPlot = 0;
    end
end
if (  MakeGrangerPlot == 1 )  
    %
    %   get the y hat's
    %
    Ystar_x = Ystar_x_arr(:,x_lag)   ;
    Yhat_x  =  Yhat_x_arr(:,x_lag)   ;
    
    Ystar_y = Ystar_y_arr(:,y_lag)   ;
    Yhat_y  =  Yhat_y_arr(:,y_lag)   ;
    
    
    dx = ((Yhat_x - x ));
    dy = ((Yhat_y - x ));
    
    figure ;
    if ( data_date  == 0 )
        data_date = [ 1 : length(x)]';
    end
    if ( strcmp(HomeDir, '' ) == 1 )
        HomeDir = '.';
    end
    %
    % scatter plot the two on top of each other
    % we want the same axes for both - how ?
    %
    
    subplot(4,2,7);
    plot(    x , Yhat_x   );
    xlimA =   get(gca,'xlim');
    ylimA =   get(gca,'ylim');
    
    subplot(4,2,8);
    plot(   x , Yhat_y    );
    xlimB =   get(gca,'xlim') ;
    ylimB =   get(gca,'ylim');
    
    % get the axis limits
    % obs exp square plt, x and y range the same
    xlim_ =( [ min([xlimA(1) xlimB(1) ylimA(1) ylimB(1)]) ,  ...
        max([xlimA(2) xlimB(2) ylimA(2) ylimB(2)])     ])  ;
    ylim_ =  xlim_;
    
    % throw it all away - we have the limits now
    %
    hold off;
    clf;
    
    %
    % fixed axis limits
    %
    subplot(4,2,7);
    plot( x , Yhat_x , 'b.'        ,'linewidth' , 0.5  );
    xlabel(sprintf('%s',xName));
    ylabel(sprintf('Estimate\nAR(%d)',x_lag));
    
    title(sprintf('N = %d ; dBIC = %7.2f; ', Tave , dBIC ));
    
    set (gca, 'xlim',xlim_,'ylim',ylim_);  hold all;
    set (gca, 'xminortick', 'on', 'yminortick', 'on'); grid('on');
    
    subplot(4,2,8);
    plot( x , Yhat_y , 'b.'    ,'linewidth' , 0.5  );
    ylabel(sprintf('Estimate\nAR(%d,%d)',x_lag,chosen_y_lag));
    xlabel(sprintf('%s',xName));
    
    
    if (  Fprob >= 0.0001  )
        Fit_str =  sprintf('p=%4.2f%%',  100.0* Fprob   )  ;
    else
        Fit_str =  sprintf('p < 0.0001'           )  ;
    end
    
    
    % format based on size of things
    
    
    title(sprintf('F(%d,%d) = %2.1f/%2.1f = %7.1e %s', ...
        y_lag, (Nvalid_U(y_lag)-(x_lag+y_lag+1)) , F_num, F_den, F, Fit_str));
    
    set (gca, 'xlim',xlim_,'ylim',ylim_);  hold all;
    set (gca, 'xminortick', 'on', 'yminortick', 'on'); grid('on');
    
    %
    % top images
    %
    %   4.1  input time series
    % cirlces if just a few
    %
    
    subplot(4,1,1);
    
    % size(data_date)
    % size(x)
    % size(y)
    
    Ntodo = sum( ~isnan(x) ) + sum( ~isnan(y) )  ;
    
    if ( Ntodo <= 200 )
        
        plot(   data_date ,  x  , 'g-o'  ,'linewidth' , 1 ); hold all ;
        plot(   data_date ,  y  , 'r-o'  ,'linewidth' , 0.5 );
        
    else
        plot(    data_date ,  x , 'g-' ,'linewidth' , 1 ); hold all ;
        plot(    data_date ,  y , 'r-' ,'linewidth' , 0.5 );
    end
    
    
    legend(  sprintf('Caused: %s',xName), sprintf('Causal: %s',yName) , 'location', 'northeastoutside');
    
    ylabel('x, Y (Z score)');
    %xlabel('Year');
    
    ansQ = 'No';
    
    
    %
    %  you can use your own criteria here
    %
    if ((Fprob < 0.05 )  && ( dBIC <= -5.0))
        ansQ = 'YES'
    end;
    
    if  ( dBIC <= -100.0)
        ansQ = 'YES'
    end;
    display(sprintf('F %8.3e ; dBIC %8.3e => %s', Fprob, dBIC, ansQ ));
    
    
    title(sprintf('%s\nDoes %s Granger-cause [%s] ? (%s) : dBIC = %6.3f',Ptitle,yName,xName, ansQ,dBIC ));
    %set(gca,   'XTickLabel',  '' );
    set (gca, 'xminortick', 'on', 'yminortick', 'on'); grid('on');
    
    %
    % 4.2  the models
    %
    
    Ntodo = sum( ~isnan(x) ) + sum( ~isnan(Yhat_x) ) + sum( ~isnan(Yhat_y) )  ;
    
    subplot(4,1,2);
    
    if ( Ntodo  <= 300 )
        plot(   data_date ,       x   , 'go' ,'linewidth' , 1 ); hold all;
        plot(   data_date ,  Yhat_x   , 'g-' ,'linewidth' , 0.5 );
        plot(    data_date ,  Yhat_y   , 'r-' ,'linewidth' , 0.5     );
    else
        plot(   data_date ,       x , 'g'  ,'linewidth' , 1  ); hold all ;
         plot(    data_date ,  Yhat_x , 'g-' ,'linewidth' , 0.5);
         plot(   data_date ,  Yhat_y , 'r-' ,'linewidth' , 0.5     );
    end
    
    legend(   sprintf('%s',xName), ...
        sprintf('%s = AR(%d)',xName,chosen_x_lag), ...
        sprintf('+%s AR(%d,%d)',yName,chosen_x_lag,chosen_y_lag) , 'location', 'northeastoutside');
    
    ylabel('Model Z');
    
    %title(sprintf('Does %s Granger-cause [%s] ? : dBIC = %6.3f',yName,xName,dBIC ));
    
    set (gca, 'xminortick', 'on', 'yminortick', 'on'); grid('on');
    
    %
    % 4.3 error model-obs
    %
    
    subplot(4,1,3);
    Ntodo = sum( ~isnan(dx) ) + sum( ~isnan(dy) )  ;
    if ( Ntodo  <= 200 )
        semilogy(   data_date , ( dx .^ 2 ) + 1.0E-6  , 'g-o','linewidth' , 1 ) ; hold all ;
        semilogy(   data_date , ( dy .^ 2 ) + 1.0E-6  , 'r-o' ,'linewidth' , 0.5 );
    else
        semilogy(   data_date , ( dx .^ 2 ) + 1.0E-6  , 'g-' ,'linewidth' , 1 ); hold all ; 
       semilogy(   data_date , ( dy .^ 2 ) + 1.0E-6  , 'r-' ,'linewidth' , 0.5  );
    end
    
    legend(   sprintf( '%s = AR(%d)',xName,x_lag) , sprintf( '[+%s]=AR(%d,%d)',yName,chosen_x_lag,chosen_y_lag) , 'location', 'northeastoutside');
    
    x_var =  RSS_R(x_lag)/(Nvalid_R(x_lag) - 1 ) ;
    y_var = RSS_U(y_lag)/(Nvalid_U(y_lag) - 1 ) ;
    
    text( 0.05 ,  0.1 ,  sprintf('Total Variance = %5.3f ; AR(%d)= %5.3f ; AR(%d,%d)= %5.3f', ...
        TotalVar, x_lag, x_var, x_lag, chosen_y_lag,y_var),'units', 'normalized') ;
    
    ylabel('Error');
    set(gca,   'XTickLabel',  '' );
    set (gca, 'xminortick', 'on', 'yminortick', 'on'); grid('on');
     
    drawnow() ; refresh() ;
    
    strFN = strrep ( sprintf('%s/%s_Gc_%s_for_%s.png',HomeDir, yName , xName, Fprefix) , ' ',  '_') ;
    
    print(strFN  ,'-dpng' , '-r500') ;
    
    display(sprintf('created image <<%s>>',strFN));
    close
end
end
%
% end
%