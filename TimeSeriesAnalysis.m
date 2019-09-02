%% Shifra Abittan
% Professor Fontaine
% Financial Signal Processing
% Financial Time-Series Analysis Problem Set

%% Instructions:

% Before running this code, run the following MATLAB files:
% (1) "PreprocessFF48.m" to generate the yearly files containing the daily 
% returns for 48 different portfolios.
% (2) "PreprocessSP500.m" to generate the yearly files containing the daily 
% returns for the S&P500, i.e. the market portfolio used in this
% assignment.

% In order to execute the code in the two preprocessing files, you must 
% have the following three CSV datasets in the accessible folder path: 
% "48_Industry_Portfolios_daily.CSV", "S&P 500 Historical Data 2000-2005.csv", 
% and "S&P 500 Historical Data 2006-2016.csv".

%% Year Selection
% The analysis can be run on the data from any year between 2000 and 2016
% (inclusive), as selected by the user.
Year = input('Please enter a year between 2000 and 2016.\n');

% Load in FF48 dataset corresponding to Year selected
Name_Year = num2str(Year);
FileName = ['FF48_daily' Name_Year '.CSV'];

fileID = fopen(FileName);
formatSpec = repmat('%f',1,49);
FF48_withDates = textscan(fileID,formatSpec,'delimiter',',');
fclose(fileID);
 
FF48_withDates = cell2mat(FF48_withDates);
FF48_withDates = FF48_withDates.'; % Transpose the data so that security 
% index runs over rows and time interval index runs over columns.

FF48 = FF48_withDates(2:49,:); %strip off date row

% Load in S&P500 dataset corresponding to Year selected
FileName = ['SP500_daily' Name_Year '.CSV'];

fileID = fopen(FileName);
formatSpec = ['%*f %f'];
SP500 = textscan(fileID,formatSpec,'delimiter',',');
fclose(fileID);
 
SP500 = cell2mat(SP500);
SP500 = SP500.'; % formulas assume time step index runs over columns

%% Normalize data to exactly 250 returns
% Truncate or extrapolate the datablock so that there are exactly 250
% return values per year.

%FF48 Data
[~,sizeFF48] = size(FF48);

if sizeFF48 > 250 %check if more than 250 data points
    FF48 = FF48(:,1:250); %truncate (truncate at end, beginning, does it matter?)
end

if sizeFF48 < 250 %check if less than 250 data points
    extrapolate = 250 - sizeFF48; %number of points that need to be added
    FF48 = [FF48 repmat(mean(FF48,2),1,extrapolate)];%should we add mean value to end/beginning? extrapolate
end

% S&P500 Data
[~,sizeSP500] = size(SP500);

if sizeSP500 > 250 %check if more than 250 data points
    SP500 = SP500(:,1:250); %truncate (truncate at end, beginning, does it matter?)
    extrapolate = 250 - sizeSP500;
    SP500 = [SP500 repmat(mean(SP500),1,extrapolate)];%should we add mean value to end/beginning? extrapolate
end

if sizeSP500 < 250 %check if less than 250 data points
    extrapolate = 250 - sizeSP500; %number of points that need to be added
    %SP500 = [SP500 ]%should we add mean value to end/beginning? extrapolate
end

%% 1(a) 
mean_SP500 = mean(SP500);
% Estimating the Covariance Coefficients
% To compute the covariance coefficients, use the formula cov(m) =
% E(x(n+m)*conj(x(n))) - mean*mean.' where E(x(n+m)*conj(x(n))) is the 
% correlation coefficient, m corresponds to the time lag and n corresponds to
% the time index. We are interested in lag 0 through 10. To perform this 
% computation, shift the values of x(n) appropriatly and multiply by the
% unshifted, conjugated x(n)'s. Because all SP500 and FF48 values are real,
% we can ignore the conjugation (conjugate of a real number is the real
% number). Then take the average across all time samples available.

% cov(0) = E(x(n)*x(n))
cov0 = mean(SP500.*SP500) - (mean_SP500.*mean_SP500);
% cov(1) = E(x(n+1)*x(n)) - mean(x)*mean(x)
cov1 = cov_lag(SP500,1);
cov2 = cov_lag(SP500,2);
cov3 = cov_lag(SP500,3);
cov4 = cov_lag(SP500,4);
cov5 = cov_lag(SP500,5);
cov6 = cov_lag(SP500,6);
cov7 = cov_lag(SP500,7);
cov8 = cov_lag(SP500,8);
cov9 = cov_lag(SP500,9);
cov10 = cov_lag(SP500,10);
covcoeffs = [cov0 cov1 cov2 cov3 cov4 cov5 cov6 cov7 cov8 cov9 cov10];

% Toeplitz Matrix and eigenvalues

R = toeplitz(covcoeffs); %Toeplitz matrix has the same values along every 
% possible diagonal 

e = eig(R);
EigReal = isreal(e);
EigPos = e>0;
% Upon examination, all eigenvalues are strictly positive real as expected.

%% 1(b)
% Levinson-Durbin recursion, where a are the coefficients, k are the reflection
% coefficients and p is the prediction error power for the particular order
% model.

% If the reflection coefficient for a particular order is exactly zero, then
% the data is AR(m-1), where m is the order. When the reflection coefficient
% is close to 1, it means the particular order AR model is not sufficient.

% assume chopping off a 1 at the beginning bc k=10 should give 11 values
[a1,p1,k1] = levinson(covcoeffs,1);
[a2,p2,k2] = levinson(covcoeffs,2);
[a3,p3,k3] = levinson(covcoeffs,3);
[a4,p4,k4] = levinson(covcoeffs,4);
[a5,p5,k5] = levinson(covcoeffs,5);
[a6,p6,k6] = levinson(covcoeffs,6);
[a7,p7,k7] = levinson(covcoeffs,7);
[a8,p8,k8] = levinson(covcoeffs,8);
[a9,p9,k9] = levinson(covcoeffs,9);
[a10,p10,k10] = levinson(covcoeffs);
% Notice that the first k value should be 1. The levinson function omits
% this value. The k values are all very small, (for 2000 all less than 0.124;
% for 2001 all less than 0.1005,for 2002 all less than 0.09)

%%
% Least-Squares fit

% For clarity and convenience, the LS code is wrapped in a function
% LSFIT_1b that enables the specification of the order and also returns the
% residual

% Order 1
[ARcoeff1_1b,del1_1b,v1_1b] = LSFIT_1b(SP500,1);
% Order 2
[ARcoeff2_1b,del2_1b,v2_1b] = LSFIT_1b(SP500,2);
% Order 3
[ARcoeff3_1b,del3_1b,v3_1b] = LSFIT_1b(SP500,3);
% Order 4
[ARcoeff4_1b,del4_1b,v4_1b] = LSFIT_1b(SP500,4);
% Order 5
[ARcoeff5_1b,del5_1b,v5_1b] = LSFIT_1b(SP500,5);
% Order 6
[ARcoeff6_1b,del6_1b,v6_1b] = LSFIT_1b(SP500,6);
% Order 7
[ARcoeff7_1b,del7_1b,v7_1b] = LSFIT_1b(SP500,7);
% Order 8
[ARcoeff8_1b,del8_1b,v8_1b] = LSFIT_1b(SP500,8);
% Order 9
[ARcoeff9_1b,del9_1b,v9_1b] = LSFIT_1b(SP500,9);
% Order 10
[ARcoeff10_1b,del10_1b,v10_1b] = LSFIT_1b(SP500,10);


% Convert AR coefficients into reflection coefficients
k_AR1 = poly2rc([1 ARcoeff1_1b]);
k_AR2 = poly2rc([1 ARcoeff2_1b.']);
k_AR3 = poly2rc([1 ARcoeff3_1b.']);
k_AR4 = poly2rc([1 ARcoeff4_1b.']);
k_AR5 = poly2rc([1 ARcoeff5_1b.']);
k_AR6 = poly2rc([1 ARcoeff6_1b.']);
k_AR7 = poly2rc([1 ARcoeff7_1b.']);
k_AR8 = poly2rc([1 ARcoeff8_1b.']);
k_AR9 = poly2rc([1 ARcoeff9_1b.']);
k_AR10 = poly2rc([1 ARcoeff10_1b.']);

% The k value for model of order 1 matches for Levinson Durbin and LS fit.
% However, for higher orders the values do not match at all. This makes
% sense because the AR coefficients vary wildly from order to order in
% order to match the data. 

%% 1(c) AIC Value
% AIC value is 2*k/n - 2*ln(L) where k is the order/number of estimated parameters
% and L is the maximum value of the liklyhood function. Be careful because
% k includes as a parameter the variance of the residual. Therefore,for example,
% for a LS fit with 2 weights, k = 3. Specifically for LS, the AIC value
% reduces to AIC = 2k/n + ln(RSS/n) where n = # of observations, RSS = sum((y - estimate)^2)
% = sum(v^2)

% Order 1
RSS_1 = sum(v1_1b.^2);
n_1 = 250 - 1;
k_1 = 3;
AIC_1 = 2*k_1/n_1 + log(RSS_1./n_1);

% Order 2
AIC_2 = AIC(v2_1b,2);
% Order 3
AIC_3 = AIC(v3_1b,3);
% Order 4
AIC_4 = AIC(v4_1b,4);
% Order 5
AIC_5 = AIC(v5_1b,5);
% Order 6
AIC_6 = AIC(v6_1b,6);
% Order 7
AIC_7 = AIC(v7_1b,7);
% Order 8
AIC_8 = AIC(v8_1b,8);
% Order 9
AIC_9 = AIC(v9_1b,9);
% Order 10
AIC_10 = AIC(v10_1b,10);

AIC_all = [AIC_1 AIC_2 AIC_3 AIC_4 AIC_5 AIC_6 AIC_7 AIC_8 AIC_9 AIC_10];
MinAICvalue = min(AIC_all);
Order_min = find(MinAICvalue == AIC_all)

% Best model order corresponds to the lowest overall AIC value:
%{
2000: 2
2001: 2 
2002: 1
2003: 7
2004: 2
2005: 2
2006: 2
2007: 1
2008: 4
2009: 4
2010: 1
2011: 5
2012: 1
2016: 10

%}

%% 1(d) First Order Difference
% The first order difference is x(t) - x(t-1)
FirstOrderDif = SP500(2:end) - SP500(1:end-1); 

% Now repeat steps 1a-1c on the first order difference (FOD) data:
% Covariance Coefficients up to Lag M
mean_FOD = mean(FirstOrderDif);

cov0_FOD = mean(FirstOrderDif.*FirstOrderDif) - (mean_FOD.*mean_FOD);
cov1_FOD = cov_lag(FirstOrderDif,1);
cov2_FOD = cov_lag(FirstOrderDif,2);
cov3_FOD = cov_lag(FirstOrderDif,3);
cov4_FOD = cov_lag(FirstOrderDif,4);
cov5_FOD = cov_lag(FirstOrderDif,5);
cov6_FOD = cov_lag(FirstOrderDif,6);
cov7_FOD = cov_lag(FirstOrderDif,7);
cov8_FOD = cov_lag(FirstOrderDif,8);
cov9_FOD = cov_lag(FirstOrderDif,9);
cov10_FOD = cov_lag(FirstOrderDif,10);
covcoeffs_FOD = [cov0_FOD cov1_FOD cov2_FOD cov3_FOD cov4_FOD cov5_FOD cov6_FOD cov7_FOD cov8_FOD cov9_FOD cov10_FOD];

% Toeplitz Matrix and eigenvalues
R_FOD = toeplitz(covcoeffs_FOD); 
e_FOD = eig(R_FOD);
EigReal_FOD = isreal(e_FOD);
EigPos_FOD = e>0;
% Upon examination, all eigenvalues are strictly positive real as expected.

% Levinson-Durbin recursion
[a1_FOD,p1_FOD,k1_FOD] = levinson(covcoeffs_FOD,1);
[a2_FOD,p2_FOD,k2_FOD] = levinson(covcoeffs_FOD,2);
[a3_FOD,p3_FOD,k3_FOD] = levinson(covcoeffs_FOD,3);
[a4_FOD,p4_FOD,k4_FOD] = levinson(covcoeffs_FOD,4);
[a5_FOD,p5_FOD,k5_FOD] = levinson(covcoeffs_FOD,5);
[a6_FOD,p6_FOD,k6_FOD] = levinson(covcoeffs_FOD,6);
[a7_FOD,p7_FOD,k7_FOD] = levinson(covcoeffs_FOD,7);
[a8_FOD,p8_FOD,k8_FOD] = levinson(covcoeffs_FOD,8);
[a9_FOD,p9_FOD,k9_FOD] = levinson(covcoeffs_FOD,9);
[a10_FOD,p10_FOD,k10_FOD] = levinson(covcoeffs_FOD);

% The k values for the First Order Difference are larger than the k values
% applied directly to the regular data by approximatly one order of magnitude,
% with values reaching as high as 0.51 for 2002. However, the ks generally
% decay (while the regular data's ks do not). This indicates that each successivly
% higher ordered model is better.

% Least-Squares fit
% Order 1
[ARcoeff1_1b_FOD,del1_1b_FOD,v1_1b_FOD] = LSFIT_1b(FirstOrderDif,2);
% Order 2
[ARcoeff2_1b_FOD,del2_1b_FOD,v2_1b_FOD] = LSFIT_1b(FirstOrderDif,2);
% Order 3
[ARcoeff3_1b_FOD,del3_1b_FOD,v3_1b_FOD] = LSFIT_1b(FirstOrderDif,3);
% Order 4
[ARcoeff4_1b_FOD,del4_1b_FOD,v4_1b_FOD] = LSFIT_1b(FirstOrderDif,4);
% Order 5
[ARcoeff5_1b_FOD,del5_1b_FOD,v5_1b_FOD] = LSFIT_1b(FirstOrderDif,5);
% Order 6
[ARcoeff6_1b_FOD,del6_1b_FOD,v6_1b_FOD] = LSFIT_1b(FirstOrderDif,6);
% Order 7
[ARcoeff7_1b_FOD,del7_1b_FOD,v7_1b_FOD] = LSFIT_1b(FirstOrderDif,7);
% Order 8
[ARcoeff8_1b_FOD,del8_1b_FOD,v8_1b_FOD] = LSFIT_1b(FirstOrderDif,8);
% Order 9
[ARcoeff9_1b_FOD,del9_1b_FOD,v9_1b_FOD] = LSFIT_1b(FirstOrderDif,9);
% Order 10
[ARcoeff10_1b_FOD,del10_1b_FOD,v10_1b_FOD] = LSFIT_1b(FirstOrderDif,10);

% Convert AR coefficients into reflection coefficients
k_AR1_FOD = poly2rc([1 ARcoeff1_1b_FOD.']);
k_AR2_FOD = poly2rc([1 ARcoeff2_1b_FOD.']);
k_AR3_FOD = poly2rc([1 ARcoeff3_1b_FOD.']);
k_AR4_FOD = poly2rc([1 ARcoeff4_1b_FOD.']);
k_AR5_FOD = poly2rc([1 ARcoeff5_1b_FOD.']);
k_AR6_FOD = poly2rc([1 ARcoeff6_1b_FOD.']);
k_AR7_FOD = poly2rc([1 ARcoeff7_1b_FOD.']);
k_AR8_FOD = poly2rc([1 ARcoeff8_1b_FOD.']);
k_AR9_FOD = poly2rc([1 ARcoeff9_1b_FOD.']);
k_AR10_FOD = poly2rc([1 ARcoeff10_1b_FOD.']);
% Again k values do not match between Levinson Durbin and LS.

% AIC Values
% Order 1
AIC_1_FOD = AIC(v1_1b_FOD,2);
% Order 2
AIC_2_FOD = AIC(v2_1b_FOD,2);
% Order 3
AIC_3_FOD = AIC(v3_1b_FOD,3);
% Order 4
AIC_4_FOD = AIC(v4_1b_FOD,4);
% Order 5
AIC_5_FOD = AIC(v5_1b_FOD,5);
% Order 6
AIC_6_FOD = AIC(v6_1b_FOD,6);
% Order 7
AIC_7_FOD = AIC(v7_1b_FOD,7);
% Order 8
AIC_8_FOD = AIC(v8_1b_FOD,8);
% Order 9
AIC_9_FOD = AIC(v9_1b_FOD,9);
% Order 10
AIC_10_FOD = AIC(v10_1b_FOD,10);

AIC_all_FOD = [AIC_1_FOD AIC_2_FOD AIC_3_FOD AIC_4_FOD AIC_5_FOD AIC_6_FOD AIC_7_FOD AIC_8_FOD AIC_9_FOD AIC_10_FOD];
MinAICvalue_FOD = min(AIC_all_FOD);
Order_min_FOD = find(MinAICvalue_FOD == AIC_all_FOD)

% Notice the mininum orders do not match between ordinary time series and
% first order difference time series.

%% 1(e)
% Direct Model

% Model Order Too Low: 
% Best Model Order:

% Identify best model residual
Best = num2str(Order_min);
vb_1b = eval(['v' Best '_1b']);

mean_vb = mean(vb_1b);
cov0_vb = mean(vb_1b.*vb_1b) - (mean_vb.*mean_vb);
cov1_vb = cov_lag(vb_1b,1);
cov2_vb = cov_lag(vb_1b,2);
cov3_vb = cov_lag(vb_1b,3);
cov4_vb = cov_lag(vb_1b,4);
cov5_vb = cov_lag(vb_1b,5);
cov6_vb = cov_lag(vb_1b,6);
cov7_vb = cov_lag(vb_1b,7);
cov8_vb = cov_lag(vb_1b,8);
cov9_vb = cov_lag(vb_1b,9);
cov10_vb = cov_lag(vb_1b,10);

% Reflection Coefficients from Covariance = cov(M)/cov(0) = k_M
cov_vb = [cov0_vb cov1_vb cov2_vb cov3_vb cov4_vb cov5_vb cov6_vb cov7_vb cov8_vb cov9_vb cov10_vb];
k_b = cov_vb./cov0_vb;

% Maximum Model Order:
mean_v10 = mean(v10_1b);
cov0_v10 = mean(v10_1b.*v10_1b) - (mean_v10.*mean_v10);
cov1_v10 = cov_lag(v10_1b,1);
cov2_v10 = cov_lag(v10_1b,2);
cov3_v10 = cov_lag(v10_1b,3);
cov4_v10 = cov_lag(v10_1b,4);
cov5_v10 = cov_lag(v10_1b,5);
cov6_v10 = cov_lag(v10_1b,6);
cov7_v10 = cov_lag(v10_1b,7);
cov8_v10 = cov_lag(v10_1b,8);
cov9_v10 = cov_lag(v10_1b,9);
cov10_v10 = cov_lag(v10_1b,10);

cov_v10 = [cov0_v10 cov1_v10 cov2_v10 cov3_v10 cov4_v10 cov5_v10 cov6_v10 cov7_v10 cov8_v10 cov9_v10 cov10_v10];
k_10 = cov_v10./cov0_v10;

% FOD

%Best Model
% Identify best model residual
Best_FOD = num2str(Order_min_FOD);
vb_1b_FOD = eval(['v' Best_FOD '_1b_FOD']);

mean_vb_FOD = mean(vb_1b_FOD);
cov0_vb_FOD = mean(vb_1b_FOD.*vb_1b_FOD) - (mean_vb_FOD.*mean_vb_FOD);
cov1_vb_FOD = cov_lag(vb_1b_FOD,1);
cov2_vb_FOD = cov_lag(vb_1b_FOD,2);
cov3_vb_FOD = cov_lag(vb_1b_FOD,3);
cov4_vb_FOD = cov_lag(vb_1b_FOD,4);
cov5_vb_FOD = cov_lag(vb_1b_FOD,5);
cov6_vb_FOD = cov_lag(vb_1b_FOD,6);
cov7_vb_FOD = cov_lag(vb_1b_FOD,7);
cov8_vb_FOD = cov_lag(vb_1b_FOD,8);
cov9_vb_FOD = cov_lag(vb_1b_FOD,9);
cov10_vb_FOD = cov_lag(vb_1b_FOD,10);

% Reflection Coefficients from Covariance
cov_vb_FOD = [cov0_vb_FOD cov1_vb_FOD cov2_vb_FOD cov3_vb_FOD cov4_vb_FOD cov5_vb_FOD cov6_vb_FOD cov7_vb_FOD cov8_vb_FOD cov9_vb_FOD cov10_vb_FOD];
k_b_FOD = cov_vb_FOD./cov0_vb_FOD;


% Maximum Model Order:
mean_v10_FOD = mean(v10_1b_FOD);
cov0_v10_FOD = mean(v10_1b_FOD.*v10_1b_FOD) - (mean_v10_FOD.*mean_v10_FOD);
cov1_v10_FOD = cov_lag(v10_1b_FOD,1);
cov2_v10_FOD = cov_lag(v10_1b_FOD,2);
cov3_v10_FOD = cov_lag(v10_1b_FOD,3);
cov4_v10_FOD = cov_lag(v10_1b_FOD,4);
cov5_v10_FOD = cov_lag(v10_1b_FOD,5);
cov6_v10_FOD = cov_lag(v10_1b_FOD,6);
cov7_v10_FOD = cov_lag(v10_1b_FOD,7);
cov8_v10_FOD = cov_lag(v10_1b_FOD,8);
cov9_v10_FOD = cov_lag(v10_1b_FOD,9);
cov10_v10_FOD = cov_lag(v10_1b_FOD,10);

% Reflection Coefficients from Covariance Coefficients
cov_v10_FOD = [cov0_v10_FOD cov1_v10_FOD cov2_v10_FOD cov3_v10_FOD cov4_v10_FOD cov5_v10_FOD cov6_v10_FOD cov7_v10_FOD cov8_v10_FOD cov9_v10_FOD cov10_v10_FOD];
k_10_FOD = cov_v10_FOD./cov0_v10_FOD;

% Whiteness is defined as having a cov for k>M = 0

%{
v_all = [v1_1b v2_1b v3_1b v4_1b v5_1b v6_1b v7_1b v8_1b v9_1b v10_1b];
v_all_FOD = [v1_1b_FOD v2_1b_FOD v3_1b_FOD v4_1b_FOD v5_1b_FOD v6_1b_FOD v7_1b_FOD v8_1b_FOD v9_1b_FOD v10_1b_FOD];

mean_v = mean(v_all);
cov0_v = mean(v_all.*v_all) - (mean_v.*mean_v);
cov1_v = cov_lag(v_all,1);
cov2_v = cov_lag(v_all,2);
cov3_v = cov_lag(v_all,3);
cov4_v = cov_lag(v_all,4);
cov5_v = cov_lag(v_all,5);
cov6_v = cov_lag(v_all,6);
cov7_v = cov_lag(v_all,7);
cov8_v = cov_lag(v_all,8);
cov9_v = cov_lag(v_all,9);
cov10_v = cov_lag(v_all,10);

mean_v_FOD = mean(v_all_FOD);
cov0_v_FOD = mean(v_all_FOD.*v_all_FOD) - (mean_v_FOD.*mean_v_FOD);
cov1_v_FOD = cov_lag(v_all_FOD,1);
cov2_v_FOD = cov_lag(v_all_FOD,2);
cov3_v_FOD = cov_lag(v_all_FOD,3);
cov4_v_FOD = cov_lag(v_all_FOD,4);
cov5_v_FOD = cov_lag(v_all_FOD,5);
cov6_v_FOD = cov_lag(v_all_FOD,6);
cov7_v_FOD = cov_lag(v_all_FOD,7);
cov8_v_FOD = cov_lag(v_all_FOD,8);
cov9_v_FOD = cov_lag(v_all_FOD,9);
cov10_v_FOD = cov_lag(v_all_FOD,10);

%}


%% 1(f)
% Kurtosis is defined as K(X) = E(((X-u)/sig)^4)
v_std = std(vb_1b);
v_FOD_std = std(vb_1b_FOD);

K_v = mean(((vb_1b - mean(vb_1b))./v_std).^4);
K_v_FOD = mean(((vb_1b_FOD - mean(vb_1b_FOD))./v_FOD_std).^4);

% The kurtosis for a gaussian is 3. When the kurtosis of a distribution is
% larger than 3, it is heavier tailed than gaussian (i.e. positive excess 
% kurtosis = K-3). When the kurtosis of a distribution is smaller than 3, 
% like here, it is thinner tailed than gaussian (i.e. negative excess kurtosis = K-3)

%% 2. Cointegration

% The goal of this question is to identify two data streams that are
% conintegrated. There are two characteristics that must be confirmed. The
% first is a w1 or w2 value close to 1 and the second is stationary e_t. In
% order to identify such stream combinations, we will loop over all combos
% and identify the ones with w1 or w2 close to 1 for further analysis:

% To hold w values
w1_master = zeros(49,49);
w2_master = zeros(49,49);

alldata = [FF48;SP500];

for a = 1:49
    for b = 1:49

% r1t and r2t are two data streams (particular stocks from FF48 information
% or S&P500 data). We are looking for examples of conintegration. 
r1 = alldata(a,:).';
r2 = alldata(b,:).';

%r1 = FF48(a,:).';
%r2 = FF48(b,:).';

% The AR(1) models are of the form: 
% r1t = delta_12 + w_12*r_2t + e_t (1)
% r1t = delta_1 + w_1*r_1,t-1 + v_1t (2)
% r2t = delta_2 + w_2*r_2,t-1 + v_2t (3)

% The 3 parameters in each equation can be found using LS fit. For these
% three equations, the LS fit will output the weight coefficient and
% another weight value/the bias term. The bias term is the weight that is
% multiplied by 1. 

% For (1):
out12 = r1;
in12 = [repmat(1,250,1) r2]; %Here, r_2 = r_2t.
weights12 = lscov(in12,out12); %Perform LS fit

w_12 = weights12(2);

% Model Error term
e_t = out12.' - weights12.'*in12.';

% For (2):
r1_tminus1 = r1(1:249);
r1_t = r1(2:250);

in1 = [repmat(1,249,1) r1_tminus1];
out1 = r1_t;
weights1 = lscov(in1,out1); %Perform LS fit
w_1 = weights1(2);

% Model Error term
v_1t = out1.' - weights1.'*in1.';

% For (3):
r2_tminus1 = r2(1:249);
r2_t = r2(2:250);

in2 = [repmat(1,249,1) r2_tminus1];
out2 = r2_t;
weights2 = lscov(in2,out2); %Perform LS fit
w_2 = weights2(2);

% Model Error term
v_2t = out2.' - weights2.'*in2.';

% The covariance coefficients should decay well for e_t to be stationary,
% indicating cointegration between r1 and r2.
ets = e_t.';

etcov0 = mean(ets.*ets);
etcov1 = cov_lag(ets,1);
etcov2 = cov_lag(ets,2);
etcov3 = cov_lag(ets,3);
etcov4 = cov_lag(ets,4);
etcov5 = cov_lag(ets,5);
etcov6 = cov_lag(ets,6);
etcov7 = cov_lag(ets,7);
etcov8 = cov_lag(ets,8);
etcov9 = cov_lag(ets,9);
etcov10 = cov_lag(ets,10);

% Now, use cov(m) = E(x(n+m)*conj(x(n))) where m corresponds to the time 
% lag and n corresponds to the time index to compute covariance coefficients.
et_covcoeffs = repmat(0,1,51);
%et_covcoeffs(1) = mean(ets.*ets); %dont include this so an see decay
%better
for m=1:50 %compute for up to lag 50 and place in a vector
    et_covcoeffs(m+1) = cov_lag(ets,m);
end
w1_master(a,b) = w_1;
w2_master(a,b) = w_2;
    
    
    end
end

% To visualize whether there is any decay:
%figure
%Mindex = 0:50;
%stem(Mindex,et_covcoeffs) %Doesn't include lag 0 because obviously has high correlation so hard to see if decays or oscillates
%title('Covariance Coefficients of e_t: if they decay well, e_t is stationary and r1 and r2 are cointegrated')
%xlabel('Lag M')
%ylabel('Covariance Coefficient of e_t')

% If either of these values are close to 1, indicates unit-root
% non-stationarity
maxw1 = max(max(w1_master));
maxw2 = max(max(w2_master));

% Below are the maximum w values for each year. Notice all of the values
% were quite small (below 0.34) and far from 1, indicating that there is no
% unit-root non-stationarity in any of the FF48 or S&P500 data for any
% year.
%{
2000: 0.3386
2001: 0.2968
2002: 0.1491
2003: 0.2686
2004: 0.1895
2005: 0.1550
2006: 0.2717
2007: 0.1113
2008: 0.1095
2009: 0.1095
2010: 0.1749
2011: 0.0506
2012: 0.1717
2013: 0.0606
2014: 0.1273
2015: 0.2012
2016: 0.1549
%}

% Because all possible combinations of datastreams are not cointegrated 
% (fail to have w close to 1), we must simulate cointegrated data instead
% (unit-root nonstationarity):

% Cointegrating vector = [alpha1 alpha2]
% lamda between 0 and 1, controls how strongly cointegrated the series is
% x(t+1) = x(t) - lamda*(x(t) + (alpha2/alpha1)*y(t)) + e1t
% y(t+1) = y(t) - lamda*(y(t) + (alpha2/alpha1)*x(t)) + e2t

r1_gen = zeros(1,250);
r2_gen = zeros(1,250);

lamda = 0.1;
alpha = [6 3];

e1 = randn(1,250);
e2 = randn(1,250);

for t=2:250
    r1_gen(t) = r1_gen(t-1) - lamda.*(r1_gen(t-1) + (alpha(2)./alpha(1))*r2_gen(t-1)) + e1(t-1);
    r2_gen(t) = r2_gen(t-1) - lamda.*(r2_gen(t-1) + (alpha(2)./alpha(1))*r1_gen(t-1)) + e2(t-1);
end

% Now let us see if we can detect the cointegration by generating the w1,w2
% and covariance coefficients of et.

r1_gen = r1_gen.';
r2_gen = r2_gen.';

% For (1):
out12_gen = r1_gen;
in12_gen = [repmat(1,250,1) r2_gen]; %Here, r_2 = r_2t.
weights12_gen = lscov(in12_gen,out12_gen); %Perform LS fit

w_12_gen = weights12_gen(2);

% Model Error term
e_t_gen = out12_gen.' - weights12_gen.'*in12_gen.';

% For (2):
r1_tminus1_g = r1_gen(1:end-1);
r1_t_g = r1_gen(2:end);

in1_g = [repmat(1,249,1) r1_tminus1_g];
out1_g = r1_t_g;
weights1_g = lscov(in1_g,out1_g); %Perform LS fit
w_1_g = weights1_g(2);

% Model Error term
v_1t_g = out1_g.' - weights1_g.'*in1_g.';

% For (3):
r2_tminus1_g = r2_gen(1:249);
r2_t_g = r2_gen(2:250);

in2_g = [repmat(1,249,1) r2_tminus1_g];
out2_g = r2_t_g;
weights2_g = lscov(in2_g,out2_g); %Perform LS fit
w_2_g = weights2_g(2);

% Model Error term
v_2t_g = out2_g.' - weights2_g.'*in2_g.';

% The covariance coefficients should decay well for e_t to be stationary,
% indicating cointegration between r1 and r2.
et_covcoeffs = repmat(0,1,51);
for m=1:50 %compute for up to lag 50 and place in a vector
    et_covcoeffs(m+1) = cov_lag(e_t_gen,m);
end

figure
Mindex = 0:50;
stem(Mindex,et_covcoeffs) %Doesn't include lag 0 because obviously has high correlation so hard to see if decays or oscillates
title('Covariance Coefficients of e_t: if they decay well, e_t is stationary and r1 and r2 are cointegrated')
xlabel('Lag M')
ylabel('Covariance Coefficient of e_t')

% The graph displays how the covariance coefficients of e_t decay very
% nicely --> implying cointegration.


%% 3. (a)
% For an ARMA(1,1) GARCH(1,1) model, the equations in the problem statement
% reduce to:
% r_t = b1*r(t-1) + v_t - a1*v(t-1)
% v_t = sigma_t * eps_t
% sigma_t^2 = c0 + c1*v(t-1)^2 + d1*sigma(t-1)^2

% Select coefficients where c0 > 0, c1 => 0, d1 => 0 and c1 + d1 < 1 and
% the ARMA model is stable (i.e. 1 + a1*z^(-1) has all roots inside the
% unit circle --> 1 + a1/z = (z + a1)/z so z = -a1 is a root and must be
% inside the unit circle (magnitude of a1 less than 1)

% Obeying all of these rules, we can choose the coefficients to be:
b1 = 0.5;
a1 = 0.5;
c0 = 5;
c1 = 0.6;
d1 = 0.2;

% Now create the returns r_t of a fictitious stock
% r_t = b1*r(t-1) + v_t - a1*v(t-1)
% v_t = sigma_t * eps_t
% sigma_t^2 = c0 + c1*v(t-1)^2 + d1*sigma(t-1)^2

r = zeros(1,251);
eps = randn(1,251);
v = zeros(1,251);
sigma = zeros(1,251);
for t=2:251
    sigma(t) = (c0 + c1*(v(t-1).^2) + d1*(sigma(t-1))^2)^0.5;
    v(t) = sigma(t)*eps(t);
    r(t) = b1*r(t-1) + v(t) - a1*v(t-1);
end

%% (b)
% To build the ARMA(1,1) model equation r_t = b1*r(t-1) + v_t - a1*v(t-1),
% use a two step process. First, fit r_t = b1*r(t-1) + residual_t using LS to
% find b1.

% Least-Squares fit to find b1
r_forward = r(2:end).';
r_behind = r(1:end-1).';
b1fn = lscov(r_behind,r_forward)
%b1b = ((r_behind.'*r_behind)^(-1))*r_behind.'*r_forward


% Alternativly, using the pseudoinverse where b = (x^#)*y
x = r_behind;
y = r_forward;
b1pseudo = (((x.'*x)^(-1))*x.')*y

%Sanity check: using the LS formula and pseudoinverse give back the same b
%coefficient

% Find residual using b1 coefficient
residual = r_forward - b1fn.*r_behind;

% To solve for a1:
% Use the formula
% gam_0 = var(v)*(1+a1^2)
% gam_1 = -a1*var(v)
% Therefore, ro1 = gam_1/gam_1 = -a1/(1+a1^2)
% When you expand this and use the quadratic formula: a1 = (+/-)0.5*(1-4ro^2)^0.5

% Estimate ro:
gam_0 = mean(residual.*residual) - (mean(residual).*mean(residual));
gam_1 = cov(residual,1);
ro1 = gam_1./gam_0;

% Then insert ro into the equation for a1:
a1_pos = 0.5*(1-(4*(ro1.^2)))^0.5
a1_neg = -0.5*(1-(4*(ro1.^2)))^0.5

Is_a1pos_real = isreal(a1_pos)
Is_a1neg_real = isreal(a1_neg)
% if a1 values are complex can't proceed

if (Is_a1pos_real == 1)
    a_choices = [a1_pos a1_neg];
    a_mags = abs(a_choices);
    a = min(a_mags); % This is final a1 value (selects the one with mag < 1 
    % because the two roots appear as alpha and 1/alpha and the one with mag <1 ensures invertability
    v_t = residual + a.*v(1:end-1);
end


 
%% (c)
% Because a is complex, below code doesnt work. If a was real, this is what
% would be done:
%Mdl = garch(1,1); %requires econometrics toolbox
%EstMdl = estimate(Mdl,v_t);

%% (d)
eps_stT = trnd(8,1,251);

r_stT = zeros(1,251);
v_stT = zeros(1,251);
sigma_stT = zeros(1,251);
for t=2:251
    sigma_stT(t) = (c0 + c1*(v_stT(t-1).^2) + d1*(sigma_stT(t-1))^2)^0.5;
    v_stT(t) = sigma_stT(t)*eps_stT(t);
    r_stT(t) = b1*r_stT(t-1) + v_stT(t) - a1*v_stT(t-1);
end

r_stT_forward = r(2:end).';
r_stT_behind = r(1:end-1).';
b1fn_stT = lscov(r_stT_behind,r_stT_forward);

x = r_stT_behind;
y = r_stT_forward;
b1pseudo_stT = (((x.'*x)^(-1))*x.')*y;

residual_stT = r_stT_forward - b1fn_stT.*r_stT_behind;

gam_0_stT = mean(residual_stT.*residual_stT) - (mean(residual_stT).*mean(residual_stT));
gam_1_stT = cov(residual_stT,1);
ro1_stT = gam_1_stT./gam_0_stT;

a1_pos_stT = 0.5*(1-(4*(ro1_stT.^2)))^0.5
a1_neg_stT = -0.5*(1-(4*(ro1_stT.^2)))^0.5

Is_a1pos_real_stT = isreal(a1_pos_stT)
Is_a1neg_real_stT = isreal(a1_neg_stT)

if (Is_a1pos_real_stT == 1)
    a_choices_stT = [a1_pos_stT a1_neg_stT];
    a_mags_stT = abs(a_choices_stT);
    a_stT = min(a_mags_stT); % This is final a1 value (selects the one with mag < 1 
    % because the two roots appear as alpha and 1/alpha and the one with mag <1 ensures invertability
    v_t_stT = residual_stT + a_stT.*v_stT(1:end-1);

end

% Because a_stT is complex, below code doesnt work. If a_stT was real, this is what
% would be done:
%Mdl_stT = garch(1,1); %requires econometrics toolbox
%EstMdl_stT = estimate(Mdl_stT,v_t_stT);

%% (e)
figure
subplot(3,1,1)
sgtitle('Epsilon=Gaussian')
t = 1:251;
plot(t,sigma)
title('Sigma [var(r(t) | F(t-1)] vs. Time for Generated Return Data')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,2)
t = 4:251;
plot(t,sigma(4:end))
title('Sigma [var(r(t) | F(t-1)] ignoring the first 4 values vs. Time for Generated Return Data. This zooms in on sigma for better comparision with return plot')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,3)
t = 1:251;
plot(t,r)
title('Generated Return Data vs. Time')
xlabel('Time')
ylabel('Return')

figure
subplot(3,1,1)
sgtitle('Epsilon=Student t with 8 degrees of freedom')
t = 1:251;
plot(t,sigma_stT)
title('Sigma [var(r(t) | F(t-1)] vs. Time for Generated Return Data')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,2)
t = 4:251;
plot(t,sigma_stT(4:end))
title('Sigma [var(r(t) | F(t-1)] ignoring the first 4 values vs. Time for Generated Return Data. This zooms in on sigma for better comparision with return plot')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,3)
t = 1:251;
plot(t,r_stT)
title('Generated Return Data vs. Time')
xlabel('Time')
ylabel('Return')

%% Varying parameters
%{ 
This code is wrapped up into a function to allow for neat generation of
many different time series with different parameters
    eps = randn(1,251);
    r = zeros(1,251);
    v = zeros(1,251);
    sigma = zeros(1,251);

    eps_stT = trnd(8,1,251);
    r_stT = zeros(1,251);
    v_stT = zeros(1,251);
    sigma_stT = zeros(1,251);

    for t=2:251
        sigma(t) = (c0 + c1*(v(t-1).^2) + d1*(sigma(t-1))^2)^0.5;
        v(t) = sigma(t)*eps(t);
        r(t) = b1*r(t-1) + v(t) - a1*v(t-1);

        sigma_stT(t) = (c0 + c1*(v_stT(t-1).^2) + d1*(sigma_stT(t-1))^2)^0.5;
        v_stT(t) = sigma_stT(t)*eps_stT(t);
        r_stT(t) = b1*r_stT(t-1) + v_stT(t) - a1*v_stT(t-1);
    end
%}
% Varying a1, the value of the pole
[sigma2,r2,sigma_stT2,r_stT2]= rtdata(0.5,0.9,5,0.6,0.2);
[sigma3,r3,sigma_stT3,r_stT3]= rtdata(0.5,0.4,5,0.6,0.2);
[sigma4,r4,sigma_stT4,r_stT4]= rtdata(0.5,0.2,5,0.6,0.2);

figure
subplot(3,1,1)
sgtitle('Epsilon=Gaussian')
t = 1:251;
plot(t,sigma2)
hold on
plot(t,sigma3)
hold on
plot(t,sigma4)
legend('Pole = 0.9', 'Pole = 0.4', 'Pole = 0.2')
title('Sigma vs. Time for Generated Return Data when the pole a1 is varied')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,2)
t = 4:251;
plot(t,sigma2(4:end))
hold on
plot(t,sigma3(4:end))
hold on
plot(t,sigma4(4:end))
title('Sigma ignoring the first 4 values vs. Time for Generated Return Data when the pole a1 is varied')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,3)
t = 1:251;
plot(t,r2)
hold on
plot(t,r3)
hold on
plot(t,r4)
title('Generated Return Data vs. Time when pole is varied')
xlabel('Time')
ylabel('Return')

figure
subplot(3,1,1)
sgtitle('Epsilon=Student t with 8 degrees of freedom')
t = 1:251;
plot(t,sigma_stT)
title('Sigma [var(r(t) | F(t-1)] vs. Time for Generated Return Data')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,2)
t = 4:251;
plot(t,sigma_stT(4:end))
title('Sigma [var(r(t) | F(t-1)] ignoring the first 4 values vs. Time for Generated Return Data. This zooms in on sigma for better comparision with return plot')
xlabel('Time')
ylabel('Sigma')

subplot(3,1,3)
t = 1:251;
plot(t,r_stT)
title('Generated Return Data vs. Time')
xlabel('Time')
ylabel('Return')

% A pole close to the unit circle causes sharp peaks and drops.

%% 4. (a) Compute the correlation coefficient ro for n1t and n2t
% Correlation coefficient = ro = cov(n1,n2)/sd(n1)sd(n2)
% Cov(n1,n2) = E((n1-mean(n1))*(n2-mean(n2)).') but we assume that n1 and
% n2 have zero mean so the Cov reduces to a correlation = E(n1*n2.').
% Therefore, we must multiply the two terms:
% n1 = v1 - 0.7v1 - 0.6v2
% n2 = -0.5v1 + v2 -0.7v2
% n1*n2 = (v1)(-0.5v1 + v2 -0.7v2) + -0.7v1(-0.5v1 + v2 -0.7v2)
% -0.6v2(-0.5v1 + v2 -0.7v2)
% All cross terms drop out so
% cov(n1,n2) = -0.5 + (-0.7*-0.5) -0.6 + (-0.6*-0.7)
covn1n2 = -0.5 + (-0.7*-0.5) -0.6 + (-0.6*-0.7);
% Then for the standard deviation = square root of the sum of the sq of the
% coefficients
sdn1sdn2 = 1 + (-0.7)^2 + (-0.6)^2 + (-0.5)^2 + 1 + (-0.7)^2;
ron1n2 = covn1n2./sdn1sdn2;

%% (b) 
% For one particular instance in time, vt = eye(2)*randn(2,1), i.e. zero 
% mean white noise 2x1 vector with a covariance matrix = I. More generally,
% for all 250 time steps:
v = randn(2,251); %multiplying by an identity matrix doesn't change value

% Simulate x1_t and x2_t using 250 time steps and 0 initial conditions
% where

% x1_t = x1_tb4 + n1_t
% x2_t = x2_tb4 + n2_t

% n1_t = v1_t - 0.7*v1_tb4 - 0.6*v2_tb4
% n2_t = -0.5*v1_tb4 + v2_t - 0.7*v2_tb4

% Therefore,
% x1_t = x1_tb4 + v1_t - 0.7*v1_tb4 - 0.6*v2_tb4
% x2_t = x2_tb4 -0.5*v1_tb4 + v2_t - 0.7*v2_tb4

x1 = zeros(1,251);
x2 = zeros(1,251);

for t=2:251 
    x1(t) = x1(t-1) + v(1,t) - 0.7*v(1,t-1) - 0.6*v(2,t-1);
    x2(t) = x2(t-1) -0.5*v(1,t-1) + v(2,t) - 0.7*v(2,t-1);
end

% y1_t = y1_tb4 + n1_t - 0.4*n1_b4
% y2_t = n2_t = 0.5*x1_t + x2_t 

% Therefore,
% y1_t = y1_tb4 + v1_t - 0.7*v1_tb4 - 0.6*v2_tb4 - 0.4*(v1_tb4 - 0.7*v1_twiceb4 - 0.6*v2_twiceb4)
% y2_t = 0.5*x1_t + x2_t 

y1 = zeros(1,251);
y2 = zeros(1,251);

for t=3:251 
    y1(t) = y1(t-1) + v(1,t) - 0.7*v(1,t-1) - 0.6*v(2,t-1) - 0.4*(v(1,t-1) - 0.7*v(1,t-2) - 0.6*v(2,t-2));
    y2(t) = 0.5*x1(t) + x2(t);
end

% To clean up the code, I wrapped the above in a function. This allows for
% easy generation of the other 4 repetitions.
[x1_a,x2_a,y1_a,y2_a] = fourb_solver();
[x1_b,x2_b,y1_b,y2_b] = fourb_solver();
[x1_c,x2_c,y1_c,y2_c] = fourb_solver();
[x1_d,x2_d,y1_d,y2_d] = fourb_solver();

figure
t=0:250;
subplot(5,2,1)
plot(t,x1,t,x2)
legend('x1_1','x2_1')
xlabel('time')

subplot(5,2,2)
plot(t,y1,t,y2)
legend('y1_1','y2_1')
xlabel('time')

subplot(5,2,3)
plot(t,x1_a,t,x2_a)
legend('x1_2','x2_2')
xlabel('time')

subplot(5,2,4)
plot(t,y1_a,t,y2_a)
legend('y1_2','y2_2')
xlabel('time')

subplot(5,2,5)
plot(t,x1_b,t,x2_b)
legend('x1_3','x2_3')
xlabel('time')

subplot(5,2,6)
plot(t,y1_b,t,y2_b)
legend('y1_3','y2_3')
xlabel('time')

subplot(5,2,7)
plot(t,x1_c,t,x2_c)
legend('x1_4','x2_4')
xlabel('time')

subplot(5,2,8)
plot(t,y1_c,t,y2_c)
legend('y1_4','y2_4')
xlabel('time')

subplot(5,2,9)
plot(t,x1_d,t,x2_d)
legend('x1_5','x2_5')
xlabel('time')

subplot(5,2,10)
plot(t,y1_d,t,y2_d)
legend('y1_5','y2_5')
xlabel('time')

% two zeros in beg of y bc of 2 time lags

%% (c)
% Compute sample covariance of y2_t for lags 0 to 10
% To do this, compute the covariance for each of the 5 y_2t and then take a
% statistical mean: (1/(N-1) * sum(covariance coefficients) where N=5.

% Lag 0
y2_cov0 = mean(y2.*y2) - (mean(y2).*mean(y2));
y2a_cov0 = mean(y2_a.*y2_a) - (mean(y2_a).*mean(y2_a));
y2b_cov0 = mean(y2_b.*y2_b) - (mean(y2_b).*mean(y2_b));
y2c_cov0 = mean(y2_c.*y2_c) - (mean(y2_c).*mean(y2_c));
y2d_cov0 = mean(y2_d.*y2_d) - (mean(y2_d).*mean(y2_d));

y2samp_cov0 = (1/4).*(y2_cov0 + y2a_cov0 + y2b_cov0 + y2c_cov0 + y2d_cov0);

% Lag 1
y2_cov1 = [cov_lag(y2,1) cov_lag(y2_a,1) cov_lag(y2_b,1) cov_lag(y2_c,1) cov_lag(y2_d,1)];
y2samp_cov1 = (1/4).*sum(y2_cov1);

% Lag 2
y2_cov2 = [cov_lag(y2,2) cov_lag(y2_a,2) cov_lag(y2_b,2) cov_lag(y2_c,2) cov_lag(y2_d,2)];
y2samp_cov2 = (1/4).*sum(y2_cov2);

% Lag 3
y2_cov3 = [cov_lag(y2,3) cov_lag(y2_a,3) cov_lag(y2_b,3) cov_lag(y2_c,3) cov_lag(y2_d,3)];
y2samp_cov3 = (1/4)*sum(y2_cov3);

% Lag 4
y2_cov4 = [cov_lag(y2,4) cov_lag(y2_a,4) cov_lag(y2_b,4) cov_lag(y2_c,4) cov_lag(y2_d,4)];
y2samp_cov4 = (1/4)*sum(y2_cov4);

% Lag 5
y2_cov5 = [cov_lag(y2,5) cov_lag(y2_a,5) cov_lag(y2_b,5) cov_lag(y2_c,5) cov_lag(y2_d,5)];
y2samp_cov5 = (1/4)*sum(y2_cov5);

% Lag 6
y2_cov6 = [cov_lag(y2,6) cov_lag(y2_a,6) cov_lag(y2_b,6) cov_lag(y2_c,6) cov_lag(y2_d,6)];
y2samp_cov6 = (1/4)*sum(y2_cov6);

% Lag 7
y2_cov7 = [cov_lag(y2,7) cov_lag(y2_a,7) cov_lag(y2_b,7) cov_lag(y2_c,7) cov_lag(y2_d,7)];
y2samp_cov7 = (1/4)*sum(y2_cov7);

% Lag 8
y2_cov8 = [cov_lag(y2,8) cov_lag(y2_a,8) cov_lag(y2_b,8) cov_lag(y2_c,8) cov_lag(y2_d,8)];
y2samp_cov8 = (1/4)*sum(y2_cov8);

% Lag 9
y2_cov9 = [cov_lag(y2,9) cov_lag(y2_a,9) cov_lag(y2_b,9) cov_lag(y2_c,9) cov_lag(y2_d,9)];
y2samp_cov9 = (1/4)*sum(y2_cov9);

% Lag 10
y2_cov10 = [cov_lag(y2,10) cov_lag(y2_a,10) cov_lag(y2_b,10) cov_lag(y2_c,10) cov_lag(y2_d,10)];
y2samp_cov10 = (1/4)*sum(y2_cov10);

y2samp_covcoeffs = [y2samp_cov0 y2samp_cov1 y2samp_cov2 y2samp_cov3 y2samp_cov4 y2samp_cov5 y2samp_cov6 y2samp_cov7 y2samp_cov8 y2samp_cov9 y2samp_cov10]

% When you examine the y2t sample covariance coefficients, you see that it
% decays but not steeply. This means that y2t is stationary.


% y1t Sample Covariance Coefficients

% Lag 0
y1_cov0 = mean(y1.*y1) - (mean(y1).*mean(y1));
y1a_cov0 = mean(y1_a.*y1_a) - (mean(y1_a).*mean(y1_a));
y1b_cov0 = mean(y1_b.*y1_b) - (mean(y1_b).*mean(y1_b));
y1c_cov0 = mean(y1_c.*y1_c) - (mean(y1_c).*mean(y1_c));
y1d_cov0 = mean(y1_d.*y1_d) - (mean(y1_d).*mean(y1_d));

y1samp_cov0 = (1/4).*(y1_cov0 + y1a_cov0 + y1b_cov0 + y1c_cov0 + y1d_cov0);

% Lag 1
y1_cov1 = [cov_lag(y1,1) cov_lag(y1_a,1) cov_lag(y1_b,1) cov_lag(y1_c,1) cov_lag(y1_d,1)];
y1samp_cov1 = (1/4).*sum(y1_cov1);

% Lag 2
y1_cov2 = [cov_lag(y1,2) cov_lag(y1_a,2) cov_lag(y1_b,2) cov_lag(y1_c,2) cov_lag(y1_d,2)];
y1samp_cov2 = (1/4).*sum(y1_cov2);

% Lag 3
y1_cov3 = [cov_lag(y1,3) cov_lag(y1_a,3) cov_lag(y1_b,3) cov_lag(y1_c,3) cov_lag(y1_d,3)];
y1samp_cov3 = (1/4)*sum(y1_cov3);

% Lag 4
y1_cov4 = [cov_lag(y1,4) cov_lag(y1_a,4) cov_lag(y1_b,4) cov_lag(y1_c,4) cov_lag(y1_d,4)];
y1samp_cov4 = (1/4)*sum(y1_cov4);

% Lag 5
y1_cov5 = [cov_lag(y1,5) cov_lag(y1_a,5) cov_lag(y1_b,5) cov_lag(y1_c,5) cov_lag(y1_d,5)];
y1samp_cov5 = (1/4)*sum(y1_cov5);

% Lag 6
y1_cov6 = [cov_lag(y1,6) cov_lag(y1_a,6) cov_lag(y1_b,6) cov_lag(y1_c,6) cov_lag(y1_d,6)];
y1samp_cov6 = (1/4)*sum(y1_cov6);

% Lag 7
y1_cov7 = [cov_lag(y1,7) cov_lag(y1_a,7) cov_lag(y1_b,7) cov_lag(y1_c,7) cov_lag(y1_d,7)];
y1samp_cov7 = (1/4)*sum(y1_cov7);

% Lag 8
y1_cov8 = [cov_lag(y1,8) cov_lag(y1_a,8) cov_lag(y1_b,8) cov_lag(y1_c,8) cov_lag(y1_d,8)];
y1samp_cov8 = (1/4)*sum(y1_cov8);

% Lag 9
y1_cov9 = [cov_lag(y1,9) cov_lag(y1_a,9) cov_lag(y1_b,9) cov_lag(y1_c,9) cov_lag(y1_d,9)];
y1samp_cov9 = (1/4)*sum(y1_cov9);

% Lag 10
y1_cov10 = [cov_lag(y2,10) cov_lag(y1_a,10) cov_lag(y1_b,10) cov_lag(y1_c,10) cov_lag(y1_d,10)];
y1samp_cov10 = (1/4)*sum(y1_cov10);

y1samp_covcoeffs = [y1samp_cov0 y1samp_cov1 y1samp_cov2 y1samp_cov3 y1samp_cov4 y1samp_cov5 y1samp_cov6 y1samp_cov7 y1samp_cov8 y1samp_cov9 y1samp_cov10]

% Y1 decays in a similar manner but it doesnt start with covcoeffs that are
% quite as large as Y2

%%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute correlation coefficients R
function y = corr_lag(x,M)
    % Chop off M last values of x(n) because the corresponding x(n+m) are
    % not available (they come after t = 250)
    x_n = x(1:end-M);
    % Remove first M values of x(n+m) because the corresponding x(n) values
    % are not available (they come before t = 1)
    x_nm = x(M+1:end); %because MATLAB indexes from 0, must add 1 to M
    
    y = mean(x_n.*x_nm);
end

%% Compute the covariance coefficients, as described in the comments of part 1.
function y = cov_lag(x,M)
    uu = mean(x)*mean(x);
    y = corr_lag(x,M) - uu;
end

%% 4B
function [x1,x2,y1,y2] = fourb_solver()
    v = randn(2,251);

    x1 = zeros(1,251);
    x2 = zeros(1,251);
    for t=2:251 
        x1(t) = x1(t-1) + v(1,t) - 0.7*v(1,t-1) - 0.6*v(2,t-1);
        x2(t) = x2(t-1) -0.5*v(1,t-1) + v(2,t) - 0.7*v(2,t-1);
    end

    y1 = zeros(1,251);
    y2 = zeros(1,251);
    for t=3:251 
        y1(t) = y1(t-1) + v(1,t) - 0.7*v(1,t-1) - 0.6*v(2,t-1) - 0.4*(v(1,t-1) - 0.7*v(1,t-2) - 0.6*v(2,t-2));
        y2(t) = 0.5*x1(t) + x2(t);
    end
end

%% Function for 1b to solve the LS fit for an order M
function [ARcoeff,del,residual] = LSFIT_1b(data,M)
    out = data(M+1:end).';
    [r,c] = size(data);
    len = c - M;
    in = zeros(len,M);
    
    j = 1;
    k = M;
    for idx = 1:len
        in(idx,:) = data(j:k);
        j = j+1;
        k = k+1;
    end
    
    ARcoeff = lscov(in,out);
    
    u = mean(out);
    a = -1.*ARcoeff;
    del = u.*sum(a);  
    
    residual = out.' - [del ARcoeff.']*[repmat(1,1,len); in.'];
end

%% Function to compute AIC
function y = AIC(v,M)
    RSS = sum(v.^2);
    n = 250 - M;
    k = 2+M; %The additional 2 accounts for the delta term and the variance of the residual
    y = 2*k/n + log(RSS./n); % Computes AIC
end

%% Function to generate return data r_t for Q3
function [sigma,r,sigma_stT,r_stT]= rtdata(b1,a1,c0,c1,d1)
    eps = randn(1,251);
    r = zeros(1,251);
    v = zeros(1,251);
    sigma = zeros(1,251);

    eps_stT = trnd(8,1,251);
    r_stT = zeros(1,251);
    v_stT = zeros(1,251);
    sigma_stT = zeros(1,251);

    for t=2:251
        sigma(t) = (c0 + c1*(v(t-1).^2) + d1*(sigma(t-1))^2)^0.5;
        v(t) = sigma(t)*eps(t);
        r(t) = b1*r(t-1) + v(t) - a1*v(t-1);

        sigma_stT(t) = (c0 + c1*(v_stT(t-1).^2) + d1*(sigma_stT(t-1))^2)^0.5;
        v_stT(t) = sigma_stT(t)*eps_stT(t);
        r_stT(t) = b1*r_stT(t-1) + v_stT(t) - a1*v_stT(t-1);
    end
end