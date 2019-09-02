%% Shifra Abittan
% Professor Fontaine
% Financial Signal Processing
% Stochastic Calculus Problem Set

%% 1.
% Parameters
delta = 0.01; %controls variance of dW increments
N = 250; %number of intervals

n = 0:1:N;
t = n*delta;

S_0 = 1; %initial price of the stock at time 0
alpha = 0.1/N; %return
sigma = alpha; %risk/volatility
r = alpha/3; %intrest rate

%% (a) Estimate S[N/2] directly by averaging over 1000 paths (Monte Carlo)

% According to notation in pset, S[N/2]=S(T/2)=S(N*delta/2)
t_Ndiv2 = N*delta/2; %corresponding time index

% dW = Gaussian random variables with mean 0, variance delta; N(0,delta)
dW = (delta)^0.5 * randn(1000,N);

% Prepare matrix to hold prices. Index runs from 1 to N + 1 because the 
% first price is S_0 and MATLAB indexing begins at 1.
S = zeros(1000,N+1);
S(:,1) = S_0; %insert given value for S_0 for all paths

% Compute price at each time step: S_n+1 = S_n + dS and 
% dS = alpha*S*dt + sigma*S*dW --> 
% S_n+1 = S_n + alpha*S_n*(t_n+1 - t_n)+ sigma*S_n*(W_n+1 - W_n)
% where t_n+1 - t_n = delta and (W_n+1 - W_n) is N(0,delta)
for i=2:N+1 
    S(:,i) = S(:,i-1) + alpha.*S(:,i-1).*delta + sigma.*S(:,i-1).*dW(:,i-1);
end

S_Ndiv2 = S(:,N/2 + 1);

AvgS_Ndiv2 = mean(S_Ndiv2) %(1/1000)*sum(S_Ndiv2)


%% The approach below utilized discrete random walks. Instead, the Gaussian
% property of dW was harnessed above to solve part 1A.
%{ 
%% Function to Generate L Brownian Motions
function BM = BrownianMotion(L,N)
    % L = number of symmetric random walks to generate
    % N = number of steps in each random walk
    % In the return matrix, BM, each individual walk is represented in its own
    % row.
    
    % Generates an L x N-1 matrix of randomly generated 0 and 1,
    % probability = 0.5
    iid = round(rand(L,N-1));
    
    % Replaces the 0s with -1s
    iid_neg = zeros(L,N-1);
    iid_neg(iid==0)=(-1);
    iid_neg(iid==1)=(1);
    
    % Adjusts +1/-1 values for the increment size (NOT SURE IF RIGHT)
    iid_neg = iid_neg*(1/(N^(0.5)));
    
    % Computes the value of the walk at each time step
    BM = cumsum(iid_neg,2);
    
    % Start the motion at zero
    start = zeros(L,1);
    BM = [start BM];
end


W = BrownianMotion(1000,250);%1000 BMs to be used for Monte Carlo

% S[N/2]=S(T/2)=S(N*delta/2) according to notation outlined in problem
% statement
t_halfN = N*delta/2; %corresponding time index
W_halfN = W(:,N/2); %corresponding BM values

S_halfN = S_0*exp((alpha - (sigma.^2)/2)*t_halfN + sigma*W_halfN);
AvgS_halfN = mean(S_halfN);
%}


%% (b) Select and plot 10 paths S(t) and graph them
S_10 = S(1:10,:); %select 10 paths

figure
plot(t,S_10(1,:))
hold on
plot(t,S_10(2,:))
hold on
plot(t,S_10(3,:))
hold on
plot(t,S_10(4,:))
hold on
plot(t,S_10(5,:))
hold on
plot(t,S_10(6,:))
hold on
plot(t,S_10(7,:))
hold on
plot(t,S_10(8,:))
hold on
plot(t,S_10(9,:))
hold on
plot(t,S_10(10,:))
xlabel('time')
ylabel('stock price')
title('Prices Modeled by Geometric Brownian Motion: dS = \alphaSdt + \sigmaSdW')

%% (c) Discounted Stock Price
% Discounted Stock Price = X(t) = S(t) * D(t)

% In general, D(t) = exp(-integral from 0 to t of R(s). In our problem,
% R(s) is a constant, r. Therefore, D(t) = exp(-r*t) --> X(t) = S(t) * exp(-r*t)
% and S(t) = X(t) * exp(r*t).

% SDE for X(t) by Girsanov's Theorem: dX = sigma*X*dW' where dW' is a
% standard weiner process under the risk neutral measure (dW_tilda)

% To simulate the SDE, X[n+1] = X[n] + sigma*X[n]*(W'[n+1] - W'[n])
% Because W' is a standard weiner process under the risk neutral measure,
% W'[n+1] - W'[n] = N(0,delta). 

% The first step is to convert S_Ndiv2 for each of the 10 paths in part 1B
% to X[N/2] using X(t) = S(t) * exp(-r*t)
S_Ndiv2_exact = S(1:10,N/2 + 1); %pulls out the 10 values of S[N/2]
X_Ndiv2_exact = S_Ndiv2_exact * exp(-r*(N/2*delta)); %converts these values to X[N/2];T/2 = N/2*delta

% Now, generate 1000 extension paths for each of the 10 X[N/2] values using
% X[n+1] = X[n] + sigma*X[n]*N(0,delta)
X1 = discounted(X_Ndiv2_exact(1));
X2 = discounted(X_Ndiv2_exact(2));
X3 = discounted(X_Ndiv2_exact(3));
X4 = discounted(X_Ndiv2_exact(4));
X5 = discounted(X_Ndiv2_exact(5));
X6 = discounted(X_Ndiv2_exact(6));
X7 = discounted(X_Ndiv2_exact(7));
X8 = discounted(X_Ndiv2_exact(8));
X9 = discounted(X_Ndiv2_exact(9));
X10 = discounted(X_Ndiv2_exact(10));

% Next, estimate X[N/2] using the martingale formula --> expectation of
% X[N] under risk neutral measure
X1_Ndiv2_mg = mean(X1(:,125));
X2_Ndiv2_mg = mean(X2(:,125));
X3_Ndiv2_mg = mean(X3(:,125));
X4_Ndiv2_mg = mean(X4(:,125));
X5_Ndiv2_mg = mean(X5(:,125));
X6_Ndiv2_mg = mean(X6(:,125));
X7_Ndiv2_mg = mean(X7(:,125));
X8_Ndiv2_mg = mean(X8(:,125));
X9_Ndiv2_mg = mean(X9(:,125));
X10_Ndiv2_mg = mean(X10(:,125));

% Now, convert this value into estimate of S[N/2]. Compare the two values
% and calculate the error.

X_Ndiv2_mg = [X1_Ndiv2_mg; X2_Ndiv2_mg; X3_Ndiv2_mg; X4_Ndiv2_mg; X5_Ndiv2_mg; X6_Ndiv2_mg; X7_Ndiv2_mg; X8_Ndiv2_mg; X9_Ndiv2_mg; X10_Ndiv2_mg];
% Exact and martingale values are identical with 0 error? Am i doing
% something wrong?? (error in very far away decimal place)
S_Ndiv2_est = X_Ndiv2_mg * exp(r*(N/2*delta)); %convert back to S from X
MSE_S = immse(S_Ndiv2_exact,S_Ndiv2_est);

%% Complete the sentence in 1(c):
%{ 

Actually this becomes a fairly trivial issue, because dx = (0)dt + (sigma*X(t))dW'
and it is clearly obvious, therefore, that X[N] = X[N/2] + Y where Y is a
random variable that is independent of F(N/2) and under the risk-neutral
measure has mean value equal to 0, so obviously
E'[X[N]|Fn] = X[n].

%}


%% (d) Compute the variance
% Below is an estimate of the variance and NOT the conditional variance
% because the conditional variance is the sum of the mean of the variance
% and the variance of the mean. This computation is the variance of the
% mean and the mean of the variance term is missing. (?)
variance = (1/10)*sum((S_Ndiv2_est - S_Ndiv2_exact).^2)

%% 2.
% The SDE dR_HW = (a_t - b_t*R)*dt + sigma*dW' is the Hull-White interest rate
% model. The interest rate modeled can not be a martingale because there is
% a dt term. If there was only the dW' term, R would be a martingale.

%% (a) Write the Feynman-Kac PDE associated with each interest rate model
%{
The Feynman-Kac theorem enables the transformation from an SDE to a PDE.
If an SDE is of the form: dX(u) = beta(u,X(u))du + gamma(u,X(u))dW(u) then
the PDE becomes g_u(u,X(u)) + beta(u,X(u))g_X(u,X(u)) +
0.5*gamma^2(u,X(u))g_XX(u,X(u)) = 0 where g(T,X) = h(x) (initial condition)

In our case, X is R and u is t.

Hull White Model:
SDE: dR_HW = (a_t - b_t*R)*dt + sigma*dW'
            beta = a_t - b_t*R
            gamma = sigma
PDE: g_t(t,R) + (a_t - b_t*R)g_R(t,R) + 0.5*sigma^2*g_RR(t,R) = 0

Cox-Ingersoll-Ross Model:
SDE: dR_CIR = (a_t - b_t*R)*dt + sigma*R^(0.5)*dW'
            beta = a_t - b_t*R
            gamma = sigma*R^(0.5)
PDE: g_t(t,R) + (a_t - b_t*R)g_R(t,R) + 0.5*sigma^2*R*g_RR(t,R) = 0
%}

%% (b)
%fixed constants
Ka = 0.5;
Kb = 0.5;
Kc = 0.5;

T = delta*N;

b_t = Kb.*(1.1+sin(pi.*t./T));
sigma_t = Kc*(1.1+cos(4*pi.*t./T));
a_t = 0.5.*((sigma_t).^2) + Ka.*(1.1+cos(pi.*t./T));

figure 
subplot (1,3,1)
plot(t,b_t)
title('Graph of b_t')
xlabel('time')

subplot(1,3,2)
plot(t,sigma_t)
title('Graph of \sigma_t')
xlabel('time')

subplot (1,3,3)
plot(t,a_t)
title('Graph of a_t')
xlabel('time')

% Notice that the three functions remain strictly positive, above zero over
% the time interval.

epsilon = 0.0001;
R_0 = 1;

%% (b).1 Hull White Simulation
% To simulate the Hull-White SDE, use dR_HW = (a_t - b_t*R)*dt + sigma*dW'
% and R[n+1] = R[n] + dR_HW = R[n] + (a_t - b_t*R[n])*delta + sigma*N(0,delta) 

% dW' = Gaussian random variables with mean 0, variance delta; N(0,delta)
dW_tilda = (delta)^0.5 * randn(10,N+1);
    
% Matrix to hold values
R_HW = zeros(10,N+1);
    
% Insert given value R(0)
R_HW(:,1) = R_0; 

NumEpsilon_HW = zeros(10,N+1);

% Compute paths
for i=2:N+1 
    t = (i-1)*0.01; %current time
    R_HW(:,i) = R_HW(:,i-1) + (a_T(t,Ka,Kc,T) - (b_T(t,Kb,T)*R_HW(:,i-1)))*delta + sigma_T(t,Kc,T)*dW_tilda(:,i-1);
    
    %Check if R is below epsilon; if so, replace value with epsilon and
    %mark location of occurence in NumEpsilon
    for j=1:10
        if R_HW(j,i)<epsilon
            NumEpsilon_HW(j,i) = NumEpsilon_HW(j,i)+1;
            R_HW(j,i) = epsilon;
        end
    end
end

EpsilonEachPathHW = sum(NumEpsilon_HW,2);

n = 0:1:N;
t = n*delta;

figure
plot(t,R_HW(1,:))
hold on
plot(t,R_HW(2,:))
hold on
plot(t,R_HW(3,:))
hold on
plot(t,R_HW(4,:))
hold on
plot(t,R_HW(5,:))
hold on
plot(t,R_HW(6,:))
hold on
plot(t,R_HW(7,:))
hold on
plot(t,R_HW(8,:))
hold on
plot(t,R_HW(9,:))
hold on
plot(t,R_HW(10,:))
title('Hull White Interest Model 10 Simulations')
xlabel('time')
ylabel('interest rate')

%% (b).1 Cox-Ingersoll-Ross Simulation
% To simulate the Cox-Ingersoll-Ross SDE, use dR_CIR = (a_t - b_t*R)*dt + sigma*R^(0.5)*dW'
% and R[n+1] = R[n] + dR_CIR = R[n] + (a_t - b_t*R[n])*delta + sigma*R[n]^(0.5)*N(0,delta) 

% dW' = Gaussian random variables with mean 0, variance delta; N(0,delta)
%dW_tilda = (delta)^0.5 * randn(10,N+1);
    
% Matrix to hold values
R_CIR = zeros(10,N+1);
    
% Insert given value R(0)
R_CIR(:,1) = R_0; 

NumEpsilon_CIR = zeros(10,N+1);

% Compute paths
for i=2:N+1 
    t = (i-1)*0.01; %current time
    R_CIR(:,i) = R_CIR(:,i-1) + (a_T(t,Ka,Kc,T) - (b_T(t,Kb,T).*R_CIR(:,i-1))).*delta + sigma_T(t,Kc,T).*(R_CIR(:,i-1).^(0.5)).*dW_tilda(:,i-1);
    
    %Check if R is below epsilon; if so, replace value with epsilon and
    %mark location of occurence in NumEpsilon
    for j=1:10
        if R_CIR(j,i)<epsilon
            NumEpsilon_CIR(j,i) = NumEpsilon_CIR(j,i)+1;
            R_CIR(j,i) = epsilon;
        end
    end
end

EpsilonEachPathCIR = sum(NumEpsilon_CIR,2);

n = 0:1:N;
t = n*delta;

figure
plot(t,R_CIR(1,:))
hold on
plot(t,R_CIR(2,:))
hold on
plot(t,R_CIR(3,:))
hold on
plot(t,R_CIR(4,:))
hold on
plot(t,R_CIR(5,:))
hold on
plot(t,R_CIR(6,:))
hold on
plot(t,R_CIR(7,:))
hold on
plot(t,R_CIR(8,:))
hold on
plot(t,R_CIR(9,:))
hold on
plot(t,R_CIR(10,:))
title('Cox-Ingersoll-Ross Interest Model 10 Simulations')
xlabel('time')
ylabel('interest rate')

%% (b).2
% R(0) = r is the initial condition. Thus, t = 0 and c(0) = 0 because the
% integral from 0 to 0 of any integrand is 0.

% Therefore, R(T) simplifies to R(T) = r*exp(-c(T)) + integral from 0 to T
% of exp(-c(T))a(u)du + integral from 0 to T of
% exp(-2(c(T)-c(u))sigma(u)dW(u). 

syms Kb_s Kc_s Ka_s u T_s r
% Symbolically define given values
b_s = Kb_s*(1.1 + sin(pi*u/T_s));
sig_s = Kc_s*(1.1 + cos(4*pi*u/T_s));
a_s = 0.5*(sig_s)^2 + Ka_s*(1.1 + cos(u*pi/T));

% Find c(T) = integral from 0 to T of b(u)du
C_T = int(b_s,u,[0,T_s])


R0_T = r*exp(-C_T) + int(exp(-C_T)*a_s,u,0,T)
I_u = exp(-2*C_T)*sig_s %i kept the dummy variable u = t
% R_T = R0_T + integral from 0 to T of I_u dW'(u)

%% (c) Expectation and Variance %%

n = 0:1:N;
t = n*delta;
T = 250;

% dW' = Gaussian random variables with mean 0, variance delta; N(0,delta)
dW_tilda = (delta)^0.5 * randn(1000,N+1);
% Matrix to hold values
ER_HW = zeros(1000,N+1); 
% Insert given value R(0)
ER_HW(:,1) = R_0; 
ENumEpsilon_HW = zeros(1000,N+1);
% Compute paths
for i=2:N+1 
    t = (i-1)*0.01; %current time
    ER_HW(:,i) = ER_HW(:,i-1) + (a_T(t,Ka,Kc,T) - (b_T(t,Kb,T).*ER_HW(:,i-1))).*delta + sigma_T(t,Kc,T).*dW_tilda(:,i-1);
    %Check if R is below epsilon; if so, replace value with epsilon and
    %mark location of occurence in NumEpsilon
    for j=1:1000
        if ER_HW(j,i)<epsilon
            ENumEpsilon_HW(j,i) = ENumEpsilon_HW(j,i)+1;
            ER_HW(j,i) = epsilon;
        end
    end
end
EEpsilonEachPathHW = sum(NumEpsilon_HW,2);

n = 0:1:N;
t = n*delta;
T = 250;

figure
plot(t,ER_HW)
title('Hull White Interest Model 1000 Simulations')

%%
%Take the expectation by averaging over each time step
ExpR_HW = mean(ER_HW);
%Find the sample variance = sum((sample - mean)^2)/999
%VarR_HW = (1/999).*(sum((ER_HW - ExpR_HW).^2));
VarR_HW = var(ER_HW);
figure
n = 0:1:N;
t = n*delta;
plot(t, ExpR_HW)
hold on
plot(t,VarR_HW)
title('Hull-White Interest Model Expectation and Variance over Time using a Marte Carlo Approach')
legend('Expectation','Variance')

% dW' = Gaussian random variables with mean 0, variance delta; N(0,delta)
dW_tilda = (delta)^0.5 * randn(1000,N+1);
% Matrix to hold values
ER_CIR = zeros(1000,N+1); 
% Insert given value R(0)
ER_CIR(:,1) = R_0; 
ENumEpsilon_CIR = zeros(1000,N+1);
% Compute paths
for i=2:N+1 
    t = (i-1)*0.01; %current time
    ER_CIR(:,i) = ER_CIR(:,i-1) + (a_T(t,Ka,Kc,T) - (b_T(t,Kb,T).*ER_CIR(:,i-1))).*delta + sigma_T(t,Kc,T).*dW_tilda(:,i-1);
    %Check if R is below epsilon; if so, replace value with epsilon and
    %mark location of occurence in NumEpsilon
    for j=1:1000
        if ER_CIR(j,i)<epsilon
            ENumEpsilon_CIR(j,i) = ENumEpsilon_CIR(j,i)+1;
            ER_CIR(j,i) = epsilon;
        end
    end
end
EEpsilonEachPathCIR = sum(NumEpsilon_CIR,2);

%Take the expectation by averaging over each time step
ExpR_CIR = mean(ER_CIR);
%Find the sample variance = sum((sample - mean)^2)/999
VarR_CIR = (1/999).*(sum((ER_CIR - ExpR_CIR).^2));

figure
n = 0:1:N;
t = n*delta;
plot(t, ExpR_CIR)
hold on
plot(t,VarR_CIR)
title('Cox-Ingersoll-Ross Interest Model Expectation and Variance over Time using a Marte Carlo Approach')
legend('Expectation','Variance')

%% (d)
%{
NumEpsilon_HW 
EpsilonEachPathHW

NumEpsilon_CIR 
EpsilonEachPathCIR

ENumEpsilon_HW 
EEpsilonEachPathHW

ENumEpsilon_CIR 
EEpsilonEachPathCIR
%}

%% 3.(b)
C = delta.*[1 0.75;0.75 .9]; % Covariance Matrix
M = [0;0]; % Mean Vector

n = 0:1:N;
t = n*delta;

W1 = GaussianCorrelated(2,M,C);
W1 = W1.';

figure 
subplot(5,2,1)
sgtitle('10 Examples of a Pair of Correlated Brownian Motion Processes')
plot(t,W1(1,:),t,W1(2,:))
legend('W(1)','W(2)')
xlabel('time')

W2 = GaussianCorrelated(2,M,C);
W2 = W2.';

subplot(5,2,2)
plot(t,W2(1,:),t,W2(2,:))
legend('W(1)','W(2)')
xlabel('time')

W3 = GaussianCorrelated(2,M,C);
W3 = W3.';

subplot(5,2,3)
plot(t,W3(1,:),t,W3(2,:))
legend('W(1)','W(2)')
xlabel('time')

W4 = GaussianCorrelated(2,M,C);
W4 = W4.';

subplot(5,2,4)
plot(t,W4(1,:),t,W4(2,:))
legend('W(1)','W(2)')
xlabel('time')

W5 = GaussianCorrelated(2,M,C);
W5 = W5.';

subplot(5,2,5)
plot(t,W5(1,:),t,W5(2,:))
legend('W(1)','W(2)')
xlabel('time')

W6 = GaussianCorrelated(2,M,C);
W6 = W6.';

subplot(5,2,6)
plot(t,W6(1,:),t,W6(2,:))
legend('W(1)','W(2)')
xlabel('time')

W7 = GaussianCorrelated(2,M,C);
W7 = W7.';

subplot(5,2,7)
plot(t,W7(1,:),t,W7(2,:))
legend('W(1)','W(2)')
xlabel('time')

W8 = GaussianCorrelated(2,M,C);
W8 = W8.';

subplot(5,2,8)
plot(t,W8(1,:),t,W8(2,:))
legend('W(1)','W(2)')
xlabel('time')

W9 = GaussianCorrelated(2,M,C);
W9 = W9.';

subplot(5,2,9)
plot(t,W9(1,:),t,W9(2,:))
legend('W(1)','W(2)')
xlabel('time')

W10 = GaussianCorrelated(2,M,C);
W10 = W10.';

subplot(5,2,10)
plot(t,W10(1,:),t,W10(2,:))
legend('W(1)','W(2)')
xlabel('time')

%% (c)
alpha = [1;1]; %0.1/N
sigma3 = [.1 .2; .3 .4];

S1 = BMdrivenCorrBM(alpha,sigma3,W1);

figure 
subplot(5,2,1)
sgtitle('10 Examples of Geometric Brownian Processes (dS = \alphaSdt + \sigma1SdW1 + \sigma2SdW2) driven by a Pair of Correlated Brownian Motion Processes')
plot(t,S1(1,:),t,S1(2,:))
legend('S(1)','S(2)')
xlabel('time')

S2 = BMdrivenCorrBM(alpha,sigma3,W2);

subplot(5,2,2)
plot(t,S2(1,:),t,S2(2,:))
legend('S(1)','S(2)')
xlabel('time')

S3 = BMdrivenCorrBM(alpha,sigma3,W3);

subplot(5,2,3)
plot(t,S3(1,:),t,S3(2,:))
legend('S(1)','S(2)')
xlabel('time')

S4 = BMdrivenCorrBM(alpha,sigma3,W4);

subplot(5,2,4)
plot(t,S4(1,:),t,S4(2,:))
legend('S(1)','S(2)')
xlabel('time')

S5 = BMdrivenCorrBM(alpha,sigma3,W5);

subplot(5,2,5)
plot(t,S5(1,:),t,S5(2,:))
legend('S(1)','S(2)')
xlabel('time')

S6 = BMdrivenCorrBM(alpha,sigma3,W6);

subplot(5,2,6)
plot(t,S6(1,:),t,S6(2,:))
legend('S(1)','S(2)')
xlabel('time')

S7 = BMdrivenCorrBM(alpha,sigma3,W7);

subplot(5,2,7)
plot(t,S7(1,:),t,S7(2,:))
legend('S(1)','S(2)')
xlabel('time')

S8 = BMdrivenCorrBM(alpha,sigma3,W8);

subplot(5,2,8)
plot(t,S8(1,:),t,S8(2,:))
legend('S(1)','S(2)')
xlabel('time')

S9 = BMdrivenCorrBM(alpha,sigma3,W9);

subplot(5,2,9)
plot(t,S9(1,:),t,S9(2,:))
legend('S(1)','S(2)')
xlabel('time')

S10 = BMdrivenCorrBM(alpha,sigma3,W10);

subplot(5,2,10)
plot(t,S10(1,:),t,S10(2,:))
legend('S(1)','S(2)')
xlabel('time')

%% 4.
%{
The constrained optimization problem to minimize and maximize volatility
is: minimize risk^2 = w.'*C'*w subject to 1.'*w = 1 (condition that sum of the
weights is zero). We are ignoring the short selling restriction that all
weights be greater than or equal to zero.

Next, use a Lagrange multiplier: J(w,lamda) = w.'*C'*w + lamda(1 - 1.'w).
Set each partial derivative/the gradient to zero. This gives: 2C'w - 1*lamda
= 0 --> w = (lamda/2)*C^(-1)*1. Plug this back into constraint to get the
equations: w = risk^2*C'^(-1)*1 where risk = 1/(1.'*C'^(-1)*1)^0.5. C' is
defined in the problem as sigma*C*sigma.'

Putting this all together:
%}

alpha = [.1/250;.1/250]; 
sigma = [4 7; 2 6];
C = [1 0.75;0.75 .9];
C1 = sigma*C*sigma.';
risk_gradient0 = 1/([1 1]*(C1^(-1))*[1;1])^0.5;
w_gradient0 = (risk_gradient0.^2)*(C1^(-1))*[1;1]

% Because the weight vector where the gradient is 0 has a negative value,
% we reject it because this involves short selling.

% Next check boundary points to find the minimum and maximum portfolio.
w1 = [0;1];
w2 = [1;0];

sigmaprime1 = (w1.'*C1*w1)^0.5
sigmaprime2 = (w2.'*C1*w2)^0.5

% Because sigmaprime1 is the smaller of the two, w2 corresponds to the
% minimum risk portfolio and w1 corresponds to the maximum risk portfolio.

% S1 has more variance so it corresponds to the portfolio with a maximum
% variance/w1 is maximum weights. S2 has a smaller variance/fluctuating 
% behavior so it corresponds to w2 the minimum portfolio.

w_max = w1;
w_min = w2;

% V is defined as w.'S and can be used to simulate 10 pairs of Vmin,Vmax
[V1max,V1min] = Portfolio(w_max,w_min,S1);
[V2max,V2min] = Portfolio(w_max,w_min,S2);
[V3max,V3min] = Portfolio(w_max,w_min,S3);
[V4max,V4min] = Portfolio(w_max,w_min,S4);
[V5max,V5min] = Portfolio(w_max,w_min,S5);
[V6max,V6min] = Portfolio(w_max,w_min,S6);
[V7max,V7min] = Portfolio(w_max,w_min,S7);
[V8max,V8min] = Portfolio(w_max,w_min,S8);
[V9max,V9min] = Portfolio(w_max,w_min,S9);
[V10max,V10min] = Portfolio(w_max,w_min,S10);

figure 
subplot(5,2,1)
sgtitle('10 Examples of Two Portfolios with Minimum and Maximum Variances/Risks')
plot(t,V1max,t,V1min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,2)
plot(t,V2max,t,V2min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,3)
plot(t,V3max,t,V3min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,4)
plot(t,V4max,t,V4min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,5)
plot(t,V5max,t,V5min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,6)
plot(t,V6max,t,V6min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,7)
plot(t,V7max,t,V7min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,8)
plot(t,V8max,t,V8min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,9)
plot(t,V9max,t,V9min)
legend('Vmax','Vmin')
xlabel('time')

subplot(5,2,10)
plot(t,V10max,t,V10min)
legend('Vmax','Vmin')
xlabel('time')

% Notice that V_min is more or less the same for all 10 trials, steadily
% increasing. The V_max fluctuates around the V_min value and is different
% at each iteration. This shows the greater volatility of V_max relative 
% to V_min.

%% 1c Function to compute X[n] for N/2 < n < N using a Marte Carlo approach given X[N/2]
function X = discounted(X_Ndiv2)
    N = 250;
    delta = 0.01;
    sigma = 0.1/N;
    
    % dW' = Gaussian random variables with mean 0, variance delta; N(0,delta)
    dW_tilda = (delta)^0.5 * randn(1000,N/2);
    
    % Matrix to hold values
    X = zeros(1000,N/2);
    
    % Insert given value for X[N/2] as computed in X_Ndiv2_exact for all paths
    X(:,1) = X_Ndiv2; 
    
    % Compute paths
    for i=2:N/2 
        X(:,i) = X(:,i-1) + sigma.*X(:,i-1).*dW_tilda(:,i-1);
    end
end

%% 2b.1 Function to compute b_t at a specific time
function y = b_T(t,Kb,T)
    y = Kb.*(1.1+sin(pi.*t./T));
end

%% 2b.1 Function to compute sigma_t at a specific time
function y = sigma_T(t,Kc,T)
    y = Kc*(1.1+cos(4*pi.*t./T));
end

%% 2b.1 Function to compute a_t at a specific time
function y = a_T(t,Ka,Kc,T)
    y = 0.5.*((sigma_T(t,Kc,T)).^2) + Ka.*(1.1+cos(pi.*t./T));
end

%% 3.a Function to generate a set of m Gaussian random vectors
function Y = GaussianCorrelated(m,mean,covar)
    %Decompose the covariance matrix into its factorization, i.e. find C
    %such that covar = C*C.'
    C = cholcov(covar);
    Y = mean + C*randn(m,251); %mean in the form of a column vector
    Y = Y.'; %transpose so that each of the m vectors are column
end

%% 3.c Function to compute the geometric Brownian motion processes driven 
% by correlated Browninan motion processes
function S = BMdrivenCorrBM(alpha,sigma3,dW)
    % S[n+1] = S[n] + dS where dS1[n+1] = alpha1*S1[n]*dt +
    % sigma11*S1[n]*dW1 + sigma12*S1[n]*dW2 and dS2[n+1] = alpha2*S2[n]*dt +
    % sigma21*S2[n]*dW1 + sigma22*S2[n]*dW2 
    
    S = zeros(2,251); %initialize matrix to hold results
    S(:,1) = 1; %set S(0)=1 for both components
    
    for i=2:251 
        S(:,i) = S(:,i-1) + alpha.*S(:,i-1).*0.01 + sigma3(:,1).*S(:,i-1).*dW(1,i-1)  + sigma3(:,2).*S(:,i-1).*dW(2,i-1);
    end
end    

%% 4 Function to compute V, portfolio
function [Vmax,Vmin] = Portfolio(w_max,w_min,S)
    Vmax = zeros(1,251);
    Vmin = zeros(1,251);
    for i=1:251
        Vmax(i) = w_max.'*S(:,i);
        Vmin(i) = w_min.'*S(:,i);
    end
end