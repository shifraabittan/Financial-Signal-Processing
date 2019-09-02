%% Shifra Abittan

% Professor Fontaine
% ECE478 Financial Signal Processing
% Problem 2: Sparse Portfolio Analysis

%%%%%%% The following portion of the code is taken from the Problem 1 Markovitz Portfolio
%%%%%%% Analysis solution. It enables us to regenerate the portfolios of 
%%%%%%% interest that we would like to sparsify, compute Sharpe ratios for
%%%%%%% and analyze.

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

%% Sample mean and Covariance matrix for 48 securities.

% Remove time intervals
FF48 = FF48_withDates(2:end,:);
% Determine number of time steps (N)
[securities, N] = size(FF48);

% Mean vector
MeanFF48 = mean(FF48,2);
% Covariance matrix
C = (1/(N-1))*(FF48*FF48.' - MeanFF48*MeanFF48.');

%% Compute (sigma=risk,ro=return) for naive portfolio, MVP, market portfolio

% Naive Portfolio (sigma,ro)
% Weight Vector: In a Naive Portfolio, each w_i = 1/M, where M is the number
% of securities in the portfolio. For FF48, M=48 for all 17 years of interest
naive_W = repmat(1/48, 48, 1);
naive_Risk = (naive_W.' * C * naive_W)^(1/2);
naive_Return = MeanFF48.' * naive_W;

% Minimum Variance Portfolio (MVP) (sigma,ro)
one = repmat(1,securities,1);

mvp_Risk = 1/((one.' * inv(C) * one)^0.5);
mvp_W = mvp_Risk^2 * inv(C) * one;
mvp_Return = MeanFF48.' * mvp_W;

% Market Portfolio: S&P500 (sigma,ro)
market_Risk = std(SP500);
market_Return = mean(SP500,2);

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
TH_market_Return = MeanFF48.' * TH_market_W;
TH_market_Risk = (TH_market_W.' * C * TH_market_W)^0.5;

%% Compute the efficient frontier

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

%% Pick three points on the market frontier (not market portfolio or MVP)
% and generate their Weight, Risk and Return values

randseed = rand(3,1);

% 3 Random Numbers between MVP and Market Return
mus = mvp_Return + (TH_market_Return - mvp_Return).*randseed;

% Sort returns from lowest to highest
mus = sort(mus);
mu_1 = mus(1);
mu_2 = mus(2);
mu_3 = mus(3);

% Compute risk for each of the returns
sigma_1 = (1./a)*((mu_1 + d./a).^2) + (1./a)*detG;
sigma_2 = (1./a)*((mu_2 + d./a).^2) + (1./a)*detG;
sigma_3 = (1./a)*((mu_3 + d./a).^2) + (1./a)*detG;

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


%%%%%%% Below is the solution to the Sparse Portfolio Problem Set 1.Q2

%% (A)

% Check that truncation operation function, g(w,k) works using test data
% provided in the problem statement
test_w =[-0.3,0.2,1.1].';
test_expect = [-3/8 0 11/8].';
test_result = g(test_w,2);
if (isequal(test_result,test_expect))
    disp('Truncation operation works as expected')
end


% NAIVE PORTFOLIO ANALYSIS
%S1
naive_S1 = S(naive_W,1);
naive_S1_L0 = norm0(naive_S1);

naive_S1_Risk = (naive_S1.' * C * naive_S1)^(1/2);
naive_S1_Return = MeanFF48.' * naive_S1;

%S2
naive_S2 = S(naive_W,2);
naive_S2_L0 = norm0(naive_S2);

naive_S2_Risk = (naive_S2.' * C * naive_S2)^(1/2);
naive_S2_Return = MeanFF48.' * naive_S2;

%Sharpe Ratios
naive_nonsp_Sharpe = naive_Return./naive_Risk;
naive_S1_Sharpe = naive_S1_Return./naive_S1_Risk;
naive_S2_Sharpe = naive_S2_Return./naive_S2_Risk;



% MVP ANALYSIS
%S1
mvp_S1 = S(mvp_W,1);
mvp_S1_L0 = norm0(mvp_S1);

mvp_S1_Risk = mvp_Risk; %doesnt depend on weights, only on C which is not changing
mvp_S1_Return = MeanFF48.' * mvp_S1;

%S2
mvp_S2 = S(mvp_W,2);
mvp_S2_L0 = norm0(mvp_S2)

mvp_S2_Risk = mvp_Risk; %doesnt depend on weights, only on C which is not changing
mvp_S2_Return = MeanFF48.' * mvp_S2;

%Sharpe Ratios
mvp_nonsp_Sharpe = mvp_Return./mvp_Risk;
mvp_S1_Sharpe = mvp_S1_Return./mvp_S1_Risk;
mvp_S2_Sharpe = mvp_S2_Return./mvp_S2_Risk;



% MARKET PORTFOLIO (Theoretical) ANALYSIS
%S1
market_S1 = S(TH_market_W,1);
market_S1_L0 = norm0(market_S1);

market_S1_Risk = (market_S1.' * C * market_S1)^(1/2);
market_S1_Return = MeanFF48.' * market_S1;

%S2
market_S2 = S(TH_market_W,2);
market_S2_L0 = norm0(market_S2);

market_S2_Risk = (market_S2.' * C * market_S2)^(1/2);
market_S2_Return = MeanFF48.' * market_S2;

%Sharpe Ratios
market_nonsp_Sharpe = market_Return./market_Risk;
market_S1_Sharpe = market_S1_Return./market_S1_Risk;
market_S2_Sharpe = market_S2_Return./market_S2_Risk;


% Three Points on Efficient Frontier

%S1

% Three Random Numbers between MVP and Market Return
mus_S1 = mvp_S1_Return + (market_S1_Return - mvp_S1_Return).*randseed;

mu_1_S1 = mus_S1(1);
mu_2_S1 = mus_S1(2);
mu_3_S1 = mus_S1(3);

% Compute the risks where d,a,G are not dependent on weights
sigma_1_S1 = (1./a)*((mu_1_S1 + d./a).^2) + (1./a)*detG;
sigma_2_S1 = (1./a)*((mu_2_S1 + d./a).^2) + (1./a)*detG;
sigma_3_S1 = (1./a)*((mu_3_S1 + d./a).^2) + (1./a)*detG;

% Weights of the 3 Portfolios
w_1_S1 = inv(C) * m_tilda * inv(B) * [mu_1_S1 1].';
w_2_S1 = inv(C) * m_tilda * inv(B) * [mu_2_S1 1].';
w_3_S1 = inv(C) * m_tilda * inv(B) * [mu_3_S1 1].';

% Zero-Norm of the 3 Portfolios
ef1_S1_L0 = norm0(w_1_S1);
ef2_S1_L0 = norm0(w_2_S1);
ef3_S1_L0 = norm0(w_3_S1);

%S2

% Three Random Numbers between MVP and Market Return
mus_S2 = mvp_S2_Return + (market_S2_Return - mvp_S2_Return).*randseed;

mu_1_S2 = mus_S2(1);
mu_2_S2 = mus_S2(2);
mu_3_S2 = mus_S2(3);

% Compute the risks where d,a,G are not dependent on weights
sigma_1_S2 = (1./a)*((mu_1_S2 + d./a).^2) + (1./a)*detG;
sigma_2_S2 = (1./a)*((mu_2_S2 + d./a).^2) + (1./a)*detG;
sigma_3_S2 = (1./a)*((mu_3_S2 + d./a).^2) + (1./a)*detG;

% Weights of the 3 Portfolios
w_1_S2 = inv(C) * m_tilda * inv(B) * [mu_1_S2 1].';
w_2_S2 = inv(C) * m_tilda * inv(B) * [mu_2_S2 1].';
w_3_S2 = inv(C) * m_tilda * inv(B) * [mu_3_S2 1].';

% Zero-Norm of the 3 Portfolios
ef1_S2_L0 = norm0(w_1_S2);
ef2_S2_L0 = norm0(w_2_S2);
ef3_S2_L0 = norm0(w_3_S2);

%Sharpe Ratios
ef1_nonsp_Sharpe = mu_1./sigma_1;
ef1_S1_Sharpe = mu_1_S1./sigma_1_S1;
ef1_S2_Sharpe = mu_1_S2./sigma_1_S2;

ef2_nonsp_Sharpe = mu_2./sigma_2;
ef2_S1_Sharpe = mu_2_S1./sigma_2_S1;
ef2_S2_Sharpe = mu_2_S2./sigma_2_S2;

ef3_nonsp_Sharpe = mu_3./sigma_3;
ef3_S1_Sharpe = mu_3_S1./sigma_3_S1;
ef3_S2_Sharpe = mu_3_S2./sigma_3_S2;

%% Display results of partA in a table
NAIVE = [naive_S1_L0 naive_S1_Return naive_S1_Risk naive_S2_L0 naive_S2_Return naive_S2_Risk naive_nonsp_Sharpe naive_S1_Sharpe naive_S2_Sharpe].';
MVP = [mvp_S1_L0 mvp_S1_Return mvp_S1_Risk mvp_S2_L0 mvp_S2_Return mvp_S2_Risk mvp_nonsp_Sharpe mvp_S1_Sharpe mvp_S2_Sharpe].';
MARKET = [market_S1_L0 market_S1_Return market_S1_Risk market_S2_L0 market_S2_Return market_S2_Risk market_nonsp_Sharpe market_S1_Sharpe market_S2_Sharpe].';
EFFICIENT1 = [ef1_S1_L0 mu_1_S1 sigma_1_S1 ef1_S2_L0 mu_1_S2 sigma_1_S2 ef1_nonsp_Sharpe ef1_S1_Sharpe ef1_S2_Sharpe].';
EFFICIENT2 = [ef2_S1_L0 mu_2_S1 sigma_2_S1 ef2_S2_L0 mu_2_S2 sigma_2_S2 ef2_nonsp_Sharpe ef2_S1_Sharpe ef2_S2_Sharpe].';
EFFICIENT3 = [ef3_S1_L0 mu_3_S1 sigma_3_S1 ef3_S2_L0 mu_3_S2 sigma_3_S2 ef3_nonsp_Sharpe ef3_S1_Sharpe ef3_S2_Sharpe].';

Results = table(NAIVE,MVP,MARKET,EFFICIENT1,EFFICIENT2,EFFICIENT3,'RowNames',{'S1 0-norm','S1 Return','S1 Risk','S2 0-norm','S2 Return','S2 Risk','Sharpe Ratio: old','Sharpe Ratio: S1','Sharpe Ratio: S2'})


%% (B) Non-quadratic Optimization Problem
R = FF48.';
[N,M] = size(R);
u = MeanFF48;
T = 3;
oneN = repmat(1,N,1);
oneM = repmat(1,M,1);
ro = 5;

x0 = repmat(0,48,1);
A = []; %no inequalities
b = []; %no inequalities
Aeq = [ones(1,48);u.'];
beq = [1;ro];
lb = []; %no bounds
ub = []; %no bounds
nonlcon = [];

op_w = fmincon(@(w) ((norm((ro.*oneN - R*w),2)).^2 + T.*norm(w,1)),x0,A,b,Aeq,beq,lb,ub,nonlcon);
opRisk = (op_w.' * C * op_w)^(1/2);

%% For fixed T, vary ro
T = 3;
ros = randi(1000,100,1);
figure
for i=1:100
    [weights,risk] = opt(ros(i),T,R,u,C);
    hold on
    plot(ros(i),risk,'*')
end
title('Plot of Return vs. Risk for fixed T, using an optimization solver')
xlabel('Return \ro')
ylabel('Risk sigma')
% Notice that for fixed T, there is a linear relationship between risk and
% return.

%% Varying T over the same ros
ts = randi(300,5,1);
ts_hold = zeros(5,100);
L0norm_hold = zeros(5,100);
for tval = 1:5
    T = ts(tval);
    for i=1:100
        [weights,risk] = opt(ros(i),T,R,u,C);
        ts_hold(tval,i) = risk;
        L0norm_hold(tval,i) = norm0(weights);
    end
end
%%
ros = ros.';
figure
plot(ros,ts_hold(1,:),'bo')
hold on
plot(ros,ts_hold(2,:),'yo')  
hold on
plot(ros,ts_hold(3,:),'ro')  
hold on
plot(ros,ts_hold(4,:),'go')  
hold on
plot(ros,ts_hold(5,:),'co')  
title('Sweep over T values and ro values')
xlabel('Return (ro)')
ylabel('Risk (sigma)')

% Sweeping over different T values made no impact on the result; all of the
% graphs are superimposed on top of each other for the same ro and
% different T values

%% 
figure
scatter3(ros,ts_hold(1,:),L0norm_hold(1,:))
hold on
scatter3(ros,ts_hold(2,:),L0norm_hold(2,:))
hold on
scatter3(ros,ts_hold(3,:),L0norm_hold(3,:))
hold on
scatter3(ros,ts_hold(4,:),L0norm_hold(4,:))
hold on
scatter3(ros,ts_hold(5,:),L0norm_hold(5,:))
title('L0 Norm for each portfolio defined by the return and regularization parameter')
xlabel('Return(ro)')
ylabel('Regularization Parameter (T)')




%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Function to perform the "truncation operation" used in Part A
function sparse = g(w,k)
    if (k == 0) %if k is equal to 0, return the original weight vector
        sparse = w;
    end
    
    if (k ~= 0) %check that k not equal to 0 because will not index properly
        
        % Step1: Sort the elements of w by descending value of |w_i|
        w_sorted = sort(w,'descend','ComparisonMethod','abs');

        % Step2: Keep the k largest coefficients and 0 out the rest
        w_trunc = w_sorted;
        w_trunc(k+1:end) = 0; %zero out all elements that come after the kth w_i

        % The result must be returned with each weight component, w_i, in its
        % original location. To accomplish this, we compare each element in
        % the original weight vector and zero it out if its magnitude is
        % smaller than the magnitude of w_k in w_trunc, aka the smallest w_i
        % that was preserved.
        w_k = w_trunc(k); %identify smallest magnitude to keep
        w_trunc_loc = w;
        w_trunc_loc(abs(w_trunc_loc)<abs(w_k)) = 0; %0s out all values with mag
        %smaller than threshhold/kth value in sorted

        % Upon running this function on the naive portfolio, where all of the
        % weights have the same magnitude, it became clear that there is a
        % boundary case that must be handled: if the threshhold magnitude
        % corresponds to multiple weights in the original vector, none of them
        % will be zeroed out, even though some of them may be above the k limit
        % of terms that we are allowed to keep. To overcome this issue:
        IndexofMatch = abs(w) == abs(w_k);
        NumAbsMatch = sum(IndexofMatch); %this checks how many weights have 
        %an absolute value that matches the threshhold

        if (NumAbsMatch > 1)
            % First check if any of the occurences happen after the kth index.
            % These are the only cases that we care about because they must be
            % set to 0.
            AfterK = IndexofMatch(k+1:end);
            if (sum(AfterK) > 0)
                index = find(AfterK>0);
                index = index+k; %indexs that have to be zeroed out
                [r,c] = size(index);
                % Loop through problematic index and correct
                for j=1:r
                    w_trunc_loc(index(j)) = 0; %zeros out w_i with mag matching w_k
                end
            end
        end

        % Step3: Check if the sum of w_trunc_loc is less than or equal to zero. If 
        % it is the result is unresonable and k is rejected. The truncated 
        % weight vector is defined as infinity.
        if (sum(w_trunc_loc) <= 0)
            sparse = inf;
        end

        % Step4: If the sum of w_trunc is strictly positive, we normalize by a
        % positive constant so that w_trunc sums to 1.
        if (sum(w_trunc_loc) > 0)
            scale = 1./sum(w_trunc_loc); %compute the positive scaling factor
            sparse = w_trunc_loc .* scale; %scale each w_i value
        end
        
    end
end


%% Function to perform the "sparsification operation" used in PartA
function y = S(w,p)
    % The sparsification operation, S_p(w) is defined as g(w,k) such that
    % k = arg min || w - g(w,k) || <= threshhold * ||w|| where ||~|| is the
    % p-norm. For the purpose of this question, we will restrict ourselves
    % to the L1 and L2 norms, p = 1,2.
    
    % || w - g(w) || is the amount of error introduced by zeroing out terms. In
    % order to achieve sparcity, we want to keep as few terms as possible, i.e.
    % minimize k. The error produced by choosing a k is not monotonic. This
    % means the error does not necessarily get larger by decreasing k and
    % removing more terms! Therefore, when computing the minimum k to minimize
    % the error, all values of k must be checked and then the smallest k that
    % meets a certain threshhold can be selected.
    
    % First, determine how many elements are in w. This is the upper bound
    % on k and reveals how many possible k values we must check for.
    [maxk,~] = size(w);
    kthreshhold = zeros(1,maxk); %This vector will hold the k value if the
    % k value meets the threshold spec and a 0 if the k value produces an
    % error that is greater than threshhold.
    
    threshhold = 1;
    % This threshhold value, through trial and error, was tuned to achieve
    % interesting results. The MVP and Market zero-norms fall off very
    % quickly, even with a slight rise in threshhold, the naive falls off 
    % moderatly quickly, while the efficient norms require a very high threshhold
    % to begin to decrease. This is because the magnitude of the weights
    % are all very small.
    
    for k=1:maxk %all the possible k values
        if (norm(w-g(w,k),p) <= threshhold .* norm(w,p)) %checks if condition is met
            kthreshhold(k) = k; %otherwise value stays at 0 to indicate failure
        end
    end
         
    % Now determine the first time a k value produced a small enough error.
    % Use this k to form S_p(w)
    k_min = min(kthreshhold(kthreshhold>0));
    y = g(w,k_min);
end

%% Function to compute the 0-norm/how many non-zero coefficients
function y = norm0(w)
    y = nnz(w); %This function counts the number of non-zero elements
end

%% Function that optimizes w and finds risk given return and regularization parameter
function [weights,risk] = opt(ro,T,R,u,C)
    [N,M] = size(R);
    oneN = repmat(1,N,1);
    oneM = repmat(1,M,1);

    x0 = repmat(0,48,1);
    A = []; %no inequalities
    b = []; %no inequalities
    Aeq = [ones(1,48);u.'];
    beq = [1;ro];
    lb = []; %no bounds
    ub = []; %no bounds
    nonlcon = [];

    weights = fmincon(@(w) ((norm((ro.*oneN - R*w),2)).^2 + T.*norm(w,1)),x0,A,b,Aeq,beq,lb,ub,nonlcon);
    risk = (weights.' * C * weights)^(1/2);
end

