%% Shifra Abittan

% Professor Fontaine
% ECE478 Financial Signal Processing
% Problem 1: Basic Markovitz Portfolio Analysis

%% Instructions:

% Before running this code, run the following MATLAB files:
% (1) "PreprocessFF48.m" to generate the yearly files containing the daily 
% returns for 48 different portfolios.
% (2) "PreprocessSP500.m" to generate the yearly files containing the daily 
% returns for the S&P500, i.e. the market portfolio used in this
% assignment.
% (3) "PreprocessLIBOR.m" to generate the yearly files containing the
% effective daily rates for the LIBOR, i.e. the risk-free rate used in this
% assignment.

% In order to execute the code in the three preprocessing files, you must 
% have the following four CSV datasets in the accessible folder path: 
% "48_Industry_Portfolios_daily.CSV", "S&P 500 Historical Data 2000-2005.csv", 
% "S&P 500 Historical Data 2006-2016.csv", and "LIBOR Historical Data.csv".

%% Year Selection
% The analysis can be run on the data from any year between 2000 and 2016
% (inclusive), as selected by the user.
Year = input('Please enter a year between 2000 and 2016.\n');

% Load in FF48 dataset corresponding to Year selected
Name_Year = num2str(Year);
FileName = ['FF48_daily' Name_Year '.CSV'];

fileID = fopen(FileName);
formatSpec = repmat('%f',1,49);
FF48_withDates_notTranspose = textscan(fileID,formatSpec,'delimiter',',');
fclose(fileID);
 
FF48_withDates_notTranspose = cell2mat(FF48_withDates_notTranspose);
FF48_withDates = FF48_withDates_notTranspose.'; % Transpose the data so that 
% data is in the proper form to match formulas. Formulas assume security index runs 
% over rows and time interval index runs over columns.

% Load in S&P500 dataset corresponding to Year selected
FileName = ['SP500_daily' Name_Year '.CSV'];

fileID = fopen(FileName);
formatSpec = ['%*f %f'];
SP500 = textscan(fileID,formatSpec,'delimiter',',');
fclose(fileID);
 
SP500 = cell2mat(SP500);
SP500 = SP500.'; % formulas assume time step index runs over columns

% Load in LIBOR dataset corresponding to Year selected
FileName = ['LIBOR_daily' Name_Year '.CSV'];

fileID = fopen(FileName);
formatSpec = ['%f'];
LIBOR = textscan(fileID,formatSpec,'delimiter',',');
fclose(fileID);
 
LIBOR = cell2mat(LIBOR);
LIBOR = LIBOR.'; % formulas assume time step index runs over columns

%% (A) Sample mean and Covariance matrix for 48 securities.

% Remove time intervals
FF48 = FF48_withDates(2:end,:);

% Determine number of time steps (N)
[securities, N] = size(FF48);

% Mean vector
MeanFF48 = mean(FF48,2);

% Covariance matrix
C = (1/(N-1))*(FF48*FF48.' - MeanFF48*MeanFF48.');

%% (B) 
% Compute (sigma=risk,µ=return) for the naive portfolio, MVP, and market portfolio 

% Naive Portfolio (sigma,µ)
% Weight Vector: In a Naive Portfolio, each w_i = 1/M, where M is the number
% of securities in the portfolio. For FF48, M=48 for all 17 years of interest
naive_W = repmat(1/48, 48, 1);
naive_Risk = (naive_W.' * C * naive_W)^(1/2)
naive_Return = MeanFF48.' * naive_W

% Minimum Variance Portfolio (MVP) (sigma,µ)
one = repmat(1,securities,1);

mvp_Risk = 1/((one.' * inv(C) * one)^0.5)
mvp_W = mvp_Risk^2 * inv(C) * one;
mvp_Return = MeanFF48.' * mvp_W

% Market Portfolio: S&P500 (sigma,µ)
market_Risk = std(SP500)
market_Return = mean(SP500,2)

% Market Portfolio: Theoretical (sigma,µ)
% A market portfolio exists and can be computed if the det(C) is not equal 
% to 0 and the risk-free return is less than the MVP return, i.e. R <
% µ_MVP. In this assignment, we ASSUME that the market portfolio is the S&P500.
% However, we can also compute the theoretical market portfolio by applying the
% formulas and models to the FF48 data. The LIBOR is used as the risk-free
% return.
R = mean(LIBOR,2); % LIBOR rate/risk free rate
m_ex = MeanFF48 - R.*one; % Vector of Excess Returns
TH_market_W = (1./(one.' * inv(C) * m_ex)) * inv(C) * m_ex;
TH_market_Return = MeanFF48.' * TH_market_W
TH_market_Risk = (TH_market_W.' * C * TH_market_W)^0.5

% Comparision between Theoretical Market Portfolio (abbreviated below as TMP) and the SP500:
%{
        TMP_Risk    TMP_Return  SP500_Risk      SP500_Return
2000    8.4497      5.1326      1.3993          -0.0337
2001    1.3613      0.7983      1.3532          -0.0477
2002    0.7384      0.3930      1.6362          -0.0868  
2003    0.6445      0.3737      1.0751          0.0988      
2004    0.9376      0.4920      0.6977          0.0356   
2005    0.9501      -0.4763     0.6449          0.0118   
2006    0.9059      -0.3339     0.6250          0.0464
2007    0.9059      -0.3339     0.6250          0.0464
2008    1.5254      -0.6207     2.5810          -0.1588
2009    6.3374      3.4311      1.7189          0.0980
2010    2.5412      1.0943      1.1370          0.0540
2011    3.2327      1.3933      1.4691          0.0108
2012    1.5859      0.7698      0.8040          0.0538 
2013    1.0223      0.4867      0.6970          0.1053
2014    1.7407      0.7955      0.7163          0.0453
2015    1.9003      0.8173      0.9761          0.0018
2016    1.1899      0.5320      0.8250          0.0396

The S&P500 return is relatively small and has a small variance. The theoretical market
portfolio projects much larger returns with a larger measure of variation.
One possible explanation as to why the SP500 varies so much from the
theoretical market portfolio, as evidenced above, is because many of the
theoretical market portfolio weights are negative. The SP500 index contains
a portion of each of the 505 stocks. A person buy them as a bundle; one can
not short sell (negative weight) certain stocks in the SP500, while owning
positive quanitites (long position) of others. All of the stocks come as a
group. There is a weight vector but it is strictly positive. Because of
this limitation, the return possible is curtailed and can not reach that of
the theoretical market portfolio.
%}

%% Randomly select a pair of securities and draw the feasible portfolio 
% curve in the (sigma,mu) plane assuming no short selling. Repeat this 5 
% times, to generate various curves. Superimpose all this in a plot.

% Randomly select 2 securities from the FF48 data
Security1 = randi(48,1);
Security2 = randi(48,1);

Security1_data = FF48(Security1, :);
Security2_data = FF48(Security2, :);
Pair = [Security1_data; Security2_data];

% Mean values of selected securities
Mean1 = MeanFF48(Security1,1);
Mean2 = MeanFF48(Security2,1);
Mean_12 = [Mean1; Mean2];

% Covariance matrix of selected securities
C_12 = (1/(N-1))*(Pair*Pair.' - Mean_12*Mean_12.');
% Alternativly, can pick covariances off of general FF48 Covariance matrix: C(Security1,Security2)

% MVP weight vector, MVP risk and MVP return for the 2 selected securities
one = [1;1];
mvp_Risk_12 = 1/((one.' * inv(C_12) * one)^0.5);
mvp_W_12 = mvp_Risk_12^2 * inv(C_12) * one;
mvp_Return_12 = Mean_12.' * mvp_W_12;

% Equation of the Feasible Portfolio Curve (traces out a hyperbola)
syms x y f(x,y); % x represents risk, y represents return; (sigma,µ)=(x,y)

% Components needed to build the equation
SD1 = C_12(1,1)^0.5;
SD2 = C_12(2,2)^0.5;
ro = C_12(1,2)/(SD1*SD2);
% A_sq = (var1 + var2 - 2 * ro * SD1 * SD2)/(M1 - M2)^2
A_sq = (C_12(1,1) + C_12(2,2) - (2 * ro * SD1 * SD2))/(Mean1-Mean2)^2;

% Equation
f(x,y) = x^2 - A_sq*(y - mvp_Return_12)^2 - (mvp_Risk_12)^2;

% To exclude short selling, impose the constraint: Mean2 <= return <= Mean1
% Find the (sigma,mu) values over the range (Mean2, Mean1)
Mean_12 = sort(Mean_12);
ReturnValues = linspace(Mean_12(1), Mean_12(2), 1000);
RiskValues = (A_sq.*(ReturnValues - mvp_Return_12).^2 + (mvp_Risk_12).^2).^(1/2);

% Plot the feasible portfolio curve with a marker indicating the
% (simga1,mu1) and (sigma2,mu2) points.  At these points, the weight of the
% respective stock is 1 and the weight of the other stock is 0.
figure
subplot(2,1,1)
plot(RiskValues,ReturnValues,'k','Marker','o','MarkerIndices',[1,1000])
title('PART B: Feasible Portfolio Curves for 5 Pairs of Randomly Selected Securities')
xlabel('Risk (\sigma)')
ylabel('Return (\mu)')

% In order to clean up the code, I have taken the above steps and wrapped
% them in a function FeasiblePC.  This allows neat generatation of four more 
% random portfolios containing two assets each.
[ReturnValues2,RiskValues2] = FeasiblePC(FF48,MeanFF48,N);
[ReturnValues3,RiskValues3] = FeasiblePC(FF48,MeanFF48,N);
[ReturnValues4,RiskValues4] = FeasiblePC(FF48,MeanFF48,N);
[ReturnValues5,RiskValues5] = FeasiblePC(FF48,MeanFF48,N);

% Graph the four feasible portfolio curves on the same graph as the first
hold on
plot(RiskValues2,ReturnValues2,'g','Marker','o','MarkerIndices',[1,1000])
hold on
plot(RiskValues3,ReturnValues3,'b','Marker','o','MarkerIndices',[1,1000])
hold on
plot(RiskValues4,ReturnValues4,'y','Marker','o','MarkerIndices',[1,1000])
hold on
plot(RiskValues5,ReturnValues5,'m','Marker','o','MarkerIndices',[1,1000])

%% (C) Compute the efficient frontier

% To solve the optimization problem, which outputs the efficient frontier 
% as its solution, it is helpful to combine the two constraints into one
% equation. To write this combined equation, we must define m_tilda, an Mx2 
% full rank matrix. 
m_tilda = [MeanFF48 repmat(1,48,1)];

% Below, we will define B and G. Both are helpful in finding the efficient
% frontier.
% Compute B
B = m_tilda.' * inv(C) * m_tilda;

% Compute G
G = inv(B) * m_tilda.' * inv(C) * m_tilda * inv(B);

% G is a 2x2 positive definite matrix. The elements in G are given labels.
a = G(1,1);
b = G(2,2);
d = G(1,2);
detG = a*b - d^2;

% Markovitz Bullet Equation
MBE_returns = linspace(-5,5,1000);
MBE_risks = (a*(MBE_returns + (d./a)).^2 + (1./a)*detG).^(0.5);
subplot(2,1,2)
plot(MBE_risks, MBE_returns)
title('PART C: Efficient Frontier')
xlabel('Risk (\sigma)')
ylabel('Return (\mu)')
xlim([0,3])

% If you examine the randomly selected portfolios, their hyperbolas always
% fall inside the efficient frontier, as expected.

%% Draw a line from (0, R) to (simga,mu) of the market portfolio.

hold on
plot([0 market_Risk], [R market_Return],'r','Marker','o')
% SP500 risk and return point is not on efficient frontier; can not
% possibly be the actual market portfolio

legend('Efficient Frontier','Line from (0,R) to SP500 Portfolio')

%{
% Check - mvp from markovitz equation
% mumvp = d/a;
% sigmamvp = (detG./a).^0.5;
% does match
% isequal(mumvp,mvp_Return);
% isequal(sigmamvp,mvp_Risk);
%}

%% (D) Pick three points on the market frontier (not market portfolio or MVP)

% 3 Random Numbers between MVP and Market Return
mus = mvp_Return + (TH_market_Return - mvp_Return).*rand(3,1);

% Sort returns from lowest to highest
mus = sort(mus);
mu_1 = mus(1);
mu_2 = mus(2);
mu_3 = mus(3);

% Compute risk for each of the returns
sigma_1 = (1./a)*((mu_1 + d./a).^2) + (1./a)*detG;
sigma_2 = (1./a)*((mu_2 + d./a).^2) + (1./a)*detG;
sigma_3 = (1./a)*((mu_3 + d./a).^2) + (1./a)*detG;

%sigma_1 = (a.*((mu_1 - (d/a)).^2) - (1./a).*detG).^0.5; wrong formula
%sigma_2 = (a.*((mu_2 - (d/a)).^2) - (1./a).*detG).^0.5; wrong formula
%sigma_3 = (a.*((mu_3 - (d/a)).^2) - (1./a).*detG).^0.5; wrong formula

% Check that relative sizes of returns matches relative sizes of risks, i.e.
% smallest return corresponds to smallest risk. Output of 1 verifies this. 
check_sigma_size = sigma_1 < sigma_2
check_sigma_size2 = sigma_2 < sigma_3

% Weights of the 3 Portfolios
% Define a vector u_tilda as transpose([return 1]) where the first element
% is the return of the particular portfolio (for which the weight vector is
% being calculated)
u_1_tilda = [mu_1 1].';
u_2_tilda = [mu_2 1].';
u_3_tilda = [mu_3 1].';

w_1 = inv(C) * m_tilda * inv(B) * u_1_tilda;
w_2 = inv(C) * m_tilda * inv(B) * u_2_tilda;
w_3 = inv(C) * m_tilda * inv(B) * u_3_tilda;

% Now, show that w_2 can be obtained as a convex combination of w_1 and
% w_3. Given two portfolios, w_1 and w_3, that lie on the MVL (as they do here), 
% then a third point, w_2, on the MVL is an affine combination of the two:
% w_2 = k.* w_1 + (1 - k).* w_3 where k is a real number. To impose the
% convex condition, k must be between 0 and 1.
% w_2 = kw_1 + w_3 - kw_3
% w_2 = k(w_1 - w_3) + w_3
% (w_2 - w_3)/(w_1 - w_3) = k

k = (w_2 - w_3)./(w_1 - w_3);
constant = k(1)
% The entire k vector has the same constant value. By finding the constant
% k, it shows that any weight vector along MVL can be obtained as a
% combination of 2 weight vectors on the MVL.


%% (E) Is R < mu_mvp?

Is_R_LessThan_MVP_Return = R < mvp_Return % 1 = yes; 0 = no
R
mvp_Return
%{ 
Results
2000: Yes
2001: Yes
2002: Yes 
2003: Yes; R = ; MVP_Return = 0.0915
2004: Yes; R = 0.0246; MVP_Return = 0.0672
2005: No; R = 0.0539; MVP_Return = 0.0015
2006: No; R = 0.0779; MVP_Return = 0.0344
2007: No; R = 0.0791; MVP_Return = -0.0324
2008: No; R = 0.0439; MVP_Return = -0.0769
2009: Yes; R = 0.0105; MVP_Return = 0.0390
2010: Yes; R = 0.0052; MVP_Return = 0.0431
2011: Yes; R = 0.0052; MVP_Return = 0.0405
2012: Yes; R = 0.0066; MVP_Return = 0.0626
2013: Yes; R = 0.0041; MVP_Return = 0.0747
2014: Yes; R = 0.0036; MVP_Return = 0.0477
2015: Yes; R = 0.0049; MVP_Return = 0.0563
2016: Yes; R = 0.0114; MVP_Return = 0.1132
%}

%% (F) Find the equation of the Capital Market Line
syms RETURN_y RISK_x
RETURN_y = R + ((market_Return - R)/market_Risk) * RISK_x

%% (G) Find the beta factor for the MVP, the three portfolios you took on the efficient frontier, and the naive portfolio
% Beta = cov(Km,Kv)/varM
% OR
% Beta = (muV - R)/(muM - R); ratio of excess returns

% Using the theoretical market portfolio:

% MVP Beta-Factor
mvp_TH_beta = (mvp_Return - R)/(TH_market_Return - R)
% Naive Portfolio Beta-Factor
naive_TH_beta = (naive_Return - R)/(TH_market_Return - R)
% Efficient Frontier Portfolio Sample1
sample1_TH_beta = (mu_1 - R)/(TH_market_Return - R)
% Efficient Frontier Portfolio Sample2
sample2_TH_beta = (mu_2 - R)/(TH_market_Return - R)
% Efficient Frontier Portfolio Sample3
sample3_TH_beta = (mu_3 - R)/(TH_market_Return - R)


% Using the SP500 as the market portfolio:

% MVP Beta-Factor 
mvp_beta = (mvp_Return - R)/(market_Return - R)
% Naive Portfolio Beta-Factor
naive_beta = (naive_Return - R)/(market_Return - R)
% Efficient Frontier Portfolio Sample1
sample1_beta = (mu_1 - R)/(market_Return - R)
% Efficient Frontier Portfolio Sample2
sample2_beta = (mu_2 - R)/(market_Return - R)
% Efficient Frontier Portfolio Sample3
sample3_beta = (mu_3 - R)/(market_Return - R)

%% Compute the covariance between the return of the particular portfolio and the 
% SP500 by multiplying beta and variance(risk^2) of the market

% MVP covariance
mvp_COV = mvp_beta.*(mvp_Risk.^2) 
% Naive Portfolio covariance
naive_COV = naive_beta.*(naive_Risk.^2) 
% Efficient Frontier Portfolio Sample1
sample1_COV = sample1_beta.*(sigma_1.^2)
% Efficient Frontier Portfolio Sample2
sample2_COV = sample2_beta.*(sigma_2.^2)
% Efficient Frontier Portfolio Sample3
sample3_COV = sample3_beta.*(sigma_3.^2)

%% Graph beta vs. mu
figure 
subplot(2,1,1)
% Beta = 0 for market portfolio
scatter(TH_market_Return,0)
hold on
scatter(mvp_Return,mvp_TH_beta)
hold on
scatter(naive_Return,naive_TH_beta)
hold on
scatter(mu_1,sample1_TH_beta)
hold on
scatter(mu_2,sample2_TH_beta)
hold on
scatter(mu_3,sample3_TH_beta)
hold on

xlabel('Return (\mu)')
ylabel('Beta (\beta)')
title('Return vs. Beta Factor using the Theoretical Market Portfolio')
legend('Market Portfolio','MVP','Naive Portfolio', 'Sample1','Sample2','Sample3')

hold on
% Graph the line
fplot(@(x) (x - R)/(TH_market_Return - R))


subplot(2,1,2)
plot(market_Return,0,mvp_Return,mvp_beta,naive_Return,naive_beta,mu_1,sample1_beta,mu_2,sample2_beta,mu_3,sample3_beta,'Marker','o')
xlabel('Return (\mu)')
ylabel('Beta (\beta)')
title('Return vs. Beta Factor using the S&P500 as the Market Portfolio')
legend('Market Portfolio','MVP','Naive Portfolio', 'Sample1','Sample2','Sample3')

hold on
% Graph the line
fplot(@(x) (x - R)/(market_Return - R))


% Points lie along the line. The slope of the line corresponds to whether
% the market is on the rise or fall.
%% (H) Actual Value of $1 Invested on January 1

time = 1:1:N+1;

% Naive Portfolio
% Convert returns from percentages to actual values by dividing by 100;
% then multiply each security by its weight and sum over all securities for
% each particular day
naive_actualdailyreturn = (FF48_withDates_notTranspose(:,[2:49])./100)*naive_W;
% Add the invested $1 to the beginning of the vector
naive_actualdailyreturn = [1; naive_actualdailyreturn];
% Cumulativly sum
naive_value = cumsum(naive_actualdailyreturn);


% MVP
mvp_actualdailyreturn = (FF48_withDates_notTranspose(:,[2:49])./100)*mvp_W;
% Add the invested $1 to the beginning of the vector
mvp_actualdailyreturn = [1; mvp_actualdailyreturn];
% Cumulativly sum
mvp_value = cumsum(mvp_actualdailyreturn);

% Market Portfolio
TH_market_actualdailyreturn = (FF48_withDates_notTranspose(:,[2:49])./100)*TH_market_W;
% Add the invested $1 to the beginning of the vector
TH_market_actualdailyreturn = [1; TH_market_actualdailyreturn];
% Cumulativly sum
TH_market_value = cumsum(TH_market_actualdailyreturn);

% Sample 1
sample1_actualdailyreturn = (FF48_withDates_notTranspose(:,[2:49])./100)*w_1;
% Add the invested $1 to the beginning of the vector
sample1_actualdailyreturn = [1; sample1_actualdailyreturn];
% Cumulativly sum
sample1_value = cumsum(sample1_actualdailyreturn);

% Sample 2
sample2_actualdailyreturn = (FF48_withDates_notTranspose(:,[2:49])./100)*w_2;
% Add the invested $1 to the beginning of the vector
sample2_actualdailyreturn = [1; sample2_actualdailyreturn];
% Cumulativly sum
sample2_value = cumsum(sample2_actualdailyreturn);

% Sample 3
sample3_actualdailyreturn = (FF48_withDates_notTranspose(:,[2:49])./100)*w_3;
% Add the invested $1 to the beginning of the vector
sample3_actualdailyreturn = [1; sample3_actualdailyreturn];
% Cumulativly sum
sample3_value = cumsum(sample3_actualdailyreturn);

figure 
plot(time,naive_value)
title('Part H: Actual Value of $1 Invested on January 1 in 6 Different Portfolios')
hold on
plot(time,mvp_value)
hold on
plot(time,TH_market_value)
hold on
plot(time,sample1_value)
hold on
plot(time,sample2_value)
hold on
plot(time,sample3_value)
legend('Naive Portfolio','MVP Portfolio','Market Portfolio (theoretical)','Sample 1 Portfolio','Sample 2 Portfolio','Sample 3 Portfolio')

% Label x axis
Dates = FF48_withDates_notTranspose(:,1);
Year_fixed = strcat(Name_Year,'0000');
Year_fixed = str2num(Year_fixed);

FebYear = Year_fixed + 200;
Feb_index = max(find(Dates<FebYear))+1;

MarYear = Year_fixed + 300;
Mar_index = max(find(Dates<MarYear))+1;

AprYear = Year_fixed + 400;
Apr_index = max(find(Dates<AprYear))+1;

MayYear = Year_fixed + 500;
May_index = max(find(Dates<MayYear))+1;

JuneYear = Year_fixed + 600;
June_index = max(find(Dates<JuneYear))+1;

JulYear = Year_fixed + 700;
Jul_index = max(find(Dates<JulYear))+1;

AugYear = Year_fixed + 800;
Aug_index = max(find(Dates<AugYear))+1;

SepYear = Year_fixed + 900;
Sep_index = max(find(Dates<SepYear))+1;

OctYear = Year_fixed + 1000;
Oct_index = max(find(Dates<OctYear))+1;

NovYear = Year_fixed + 1100;
Nov_index = max(find(Dates<NovYear))+1;

DecYear = Year_fixed + 1200;
Dec_index = max(find(Dates<DecYear))+1;

xlim([0 N+1])
xticks([0 Feb_index Mar_index Apr_index May_index June_index Jul_index Aug_index Sep_index Oct_index Nov_index Dec_index])
xticklabels({'Jan','Feb','Mar','April','May','June','July','August','Sept','Oct','Nov','Dec'})

% As expected, the MVP portfolio generally has the least overall change
% because it assumes the least amount of risk.
%% FeasiblePC function used in Part (B)
function [ReturnValues,RiskValues] = FeasiblePC(FF48,MeanFF48,N)
    % Randomly select 2 securities from the FF48 data
    Security1 = randi(48,1);
    Security2 = randi(48,1);

    Security1_data = FF48(Security1, :);
    Security2_data = FF48(Security2, :);
    Pair = [Security1_data; Security2_data];

    % Mean values of selected securities
    Mean1 = MeanFF48(Security1,1);
    Mean2 = MeanFF48(Security2,1);
    Mean_12 = [Mean1; Mean2];

    % Covariance matrix of selected securities
    C_12 = (1/(N-1))*(Pair*Pair.' - Mean_12*Mean_12.');
    % Alternativly, can pick covariances off of general FF48 Covariance matrix: C(Security1,Security2)

    % MVP weight, risk and return for selected securities
    one = [1;1];

    mvp_Risk_12 = 1/((one.' * inv(C_12) * one)^0.5);
    mvp_W_12 = mvp_Risk_12^2 * inv(C_12) * one;
    mvp_Return_12 = Mean_12.' * mvp_W_12;

    % Equation of the Feasible Portfolio Curve (traces out a hyperbola)
    syms x y f(x,y); % x represents risk, y represents return; (?,µ)=(x,y)

    % Components needed to build the equation
    SD1 = C_12(1,1)^0.5;
    SD2 = C_12(2,2)^0.5;
    ro = C_12(1,2)/(SD1*SD2);
    % A_sq = (var1 + var2 - 2 * ro * SD1 * SD2)/(M1 - M2)^2
    A_sq = (C_12(1,1) + C_12(2,2) - (2 * ro * SD1 * SD2))/(Mean1-Mean2)^2;

    % Equation
    f(x,y) = x^2 - A_sq*(y - mvp_Return_12)^2 - (mvp_Risk_12)^2;

    % To exclude short selling, impose the constraint: Mean2 <= return <= Mean1
    % Find the (sigma,mu) values over the range (Mean2, Mean1)
    Mean_12 = sort(Mean_12);
    ReturnValues = linspace(Mean_12(1), Mean_12(2), 1000);
    RiskValues = (A_sq.*(ReturnValues - mvp_Return_12).^2 + (mvp_Risk_12).^2).^(1/2);
end