%% Shifra Abittan
% Professor Fontaine

% As per the problem statement, the price of the security at time zero is:
S_0 = 1;

%% (A) 
%{ 
As per the problem statement,let:
                        R_n = log10(S_n/S0)
where S0 is the price of the stock at time zero and S_n is the price of the
stock at time n, for n => 1. Express R_n in the form:
                        R_n = a*Y_n + b
where Y_n is a binomial (n,p) random variable and a and b are constants.

R_n = log10(S_n./S0)
R_n = log10(S_n) - log10(S0)         

In the multiperiod binomial model, 
                        S_n = u^(H) * d^(n-H) * S0, 
where u, d, S0, N are deterministic and known and H is the number of heads 
obtained after tossing a coin n times. Substitute S_n into the R_n formula:
R_n = log10(u^(H) * d^(n-H) * S0) - log10(S0)
R_n = log10(u^H) + log10(d^(n-H)) + log10(S0) - log10(S0)
R_n = log10(u^H) + log10(d^(n-H))
R_n = H*log10(u) + (n-H)*log10(d)
R_n = H*log10(u) + n*log10(d) - H*log10(d)
R_n = H*log10(u/d) + n*log10(d)

Because u, d and n are all deterministic, log10(u/d) is a constant and
n*log10(d) is also a constant. H is a random variable of the binomial type,
where n is the total number of tosses that were heads and p is the
probability of rolling a head.

Therefore,
                         R_n = a*Y_n + b
                         R_n = log10(u/d)*H + n*log10(d)

Y_n = H
a = log10(u/d)
b = n*log10(d)



Expectation
E(R_n) = E(log10(u/d)*H + n*log10(d))
Expectation is a linear operation; constants can be pulled out of the
expectation and expectation distributes over additive terms:
                        For Z = a*Y + b; Y=random variable
                        E(Z) = a*E(Y) + b

E(R_n) = E(log10(u/d)*H + n*log10(d))
E(R_n) = log10(u/d)*E(H) + n*log10(d)
The expectation of a binomial random variable is n*p

E(R_n) = log10(u/d) * n * p + n*log10(d)
E(R_n) = n*(log10(u/d) * p + log10(d))
 

Variance
When taking the variance of a scaled, shifted random variable, the variance
is the square of the scaling factor multiplied by the variance of the
underlying random variable:
                        For Z = a*Y + b; Y=random variable
                        Var(Z) = (a^2)*Var(Y)

Var(R_n) = Var(log10(u/d)*H + n*log10(d))
Var(R_n) = (log10(u/d))^2 * Var(H)
The variance of a binomial random variable is n*p*(1-p)
                        Var(R_n) = (log10(u/d))^2 * n * p * (1-p) 

%}

%% (B) Compute risk-neutral probability, p_tilda
% The function (B.1) is listed at the end of the script. For clarity and
% continuity, the function is duplicated here:
%{
function p_tilda = RiskNeutral_Prob(r,u,d)
    p_tilda = ((1+r)-d)./(u-d);
end
%}

%% (C) Compute the replicating portfolio, X_0, and the portfolio strategy, 
% delta_n, for a given derivative V_N and path w

% PRICE
% First, compute the price of the security for all n <= N. The computation
% is done with the function (C.1) "price", found at the end of the script. 

% The function "price" requires as inputs: u, d, w, S_0.
% The path, w, is a vector where the index corresponds to n and is
% constructed out of ones and zeros:
%       1 = tossing a head = corresponds to u
%       0 = tossing a tail = corresponds to d
% The function outputs a price vector where the index corresponds to n. The
% price at time zero, S_0, is not included in the S_n price vector.

% For clarity and continuity, the "price" function is duplicated here:
%{
function S_n = price(u,d,w,S_0)
    [r,path_length] = size(w);
    w_ud = w;
    
    % Replace each 1 with value of u and 0 with the value of d 
    w_ud(w_ud==1)=(u);
    w_ud(w_ud==0)=(d);
    
    % Multiply security price at time zero, S_0, with first value in w_ud
    % vector (technically do not need to do this because S_0 was defined to
    % be one; however, I included this line of code to preserve generality
    % for all values of S_0)
    w_ud(1) = S_0 * w_ud(1);

    % Prices at each timestep
    S_n = cumprod(w_ud); 
end
%}

% REPLICATING PORTFOLIO
% Next, compute the replicating portfolio value, X_0. The computation
% is done with the function (C.2) "rep_port", found at the end of the script. 

% The function "rep_port" requires as inputs: p_tilda, r, K, Type. p_tilda 
% is the risk-neutral probability. For definitions of K and Type see part 
% (D)

% For clarity and continuity, the "rep_port" function is duplicated here:
%{
function X_0 = rep_port(p_tilda,r,K,Type,u,d,S_0)
    q_tilda = 1 - p_tilda;

    % As stated in the problem statement, assume there is a function that 
    % computes V for a given path. The function call follows the form
    % described in part (D).
    V1(H) = payout(1,K,Type,u,d,S_0); 
    V1(T) = payout(0,K,Type,u,d,S_0); 
    
    % X_0 = V_0 by the definition of a replicating portfolio. Notationally,
    % Vn(H) means the value of the derivative at timestep = n given the
    % first n-1 tosses and landing a head on the nth toss. Vn(T) is the value
    % of the derivative at timestep = n given the same first n-1 tosses, but
    % landing a tail on the nth toss. 
    X_0 = (1./(1+r)) * (p_tilda * V1(H) + q_tilda * V1(T)
end
%}

% PORTFOLIO STRATEGY
% Finally, compute the portfolio strategy, delta_n, for all n <= N-1 for a
% given path. The computation is done with the function (C.3) "deltas", 
% found at the end of the script. 

% The function "deltas" requires as inputs: u, d, w, K, Type, S_0. w is a 
% vector describing the path.

% The function "deltas" outputs a vector describing the portfolio strategy,
% where the first value is delta0 (the indexes are offset by 1).

% For clarity and continuity, the "deltas" function is duplicated here:
%{
%% (C.3) Function to calculate the portfolio strategy
function del = deltas(u,d,w,K,Type,S_0)
    S_n = price(u,d,w,S_0);

    [r,path_length] = size(S_n);
    del = repmat(0,1,path_length+1);

    for i=1:path_length+1
        V_H = payout([w(1:i) 1],K,Type,u,d,S_0); 
        V_T = payout([w(1:i) 0],K,Type,u,d,S_0);

        S_H = S_n(i)*u;
        S_T = S_n(i)*d;
        
        del(i) = (V_H - V_T)./(S_H - S_T);
    end
end
%}


%% (D) Compute V_N for three different derivatives

% The computation is done with the function (D.1) "payout", found at the 
% end of the script. 

% The function "payout" requires as inputs: w, K, Type, u, d, S_0. w is a
% vector describing the path leading up to V_n. K is the strike
% price. Type is either 1, 2, or 3 specifying the type of derivative:
%           1 = max S_n (for 0 <= n <= N)
%           2 = European call option
%           3 = European put option

% For clarity and continuity, the "payout" function is duplicated here:
%{
function V_n = payout(w,K,Type,u,d,S_0)
    [r,path_length] = size(w);
    % Compute vector of prices for the given path w
    S_n = price(u,d,w,S_0);
    
    % max S_n
    if Type = 1
        V_n = max(S_n);
    end
    
    % European Call Option
    if Type = 2
        V_n = (S_n(path_length) - K);
        if V_n < 0
            V_n = 0;
        end
    end
    
    % European Put Option
    if Type = 3
        V_n = (K - S_n(path_length));
        if V_n < 0
            V_n = 0;
        end
    end
end
%}


%% (E) Concrete Cases using Marte Carlo approach
u = 1.005;
r = 0.003;
d = 1.002;
N = 100; %path length
p1 = 0.4;
p2 = 0.6;
p_tilda = RiskNeutral_Prob(r,u,d);

% Strike Prices
K1 = S_0*exp(N*(log10(u/d) * p1 + log10(d)));
K2 = S_0*exp(N*(log10(u/d) * p2 + log10(d)));
Ktilda = S_0*exp(N*(log10(u/d) * p_tilda + log10(d)));
%K_half = S_0*exp(N*(log10(u/d) * 0.5 + log10(d)));
K = [K1 K2 Ktilda];

%% Using p1: compute V_N for the max(S_n) derivative, denoted 'vmax'

VMax_P1 = repmat(0,1,1000); %initialize vector to hold the Vn from each of the 1000 simulations

% The K value is not used in the computation for vmax but a K must be
% passed into the randpath function. K1 was used in the function call but is 
% not used in any of the computation.

% Using K1
for i=1:1000
    w = randpath(p1); %Generates random path of length 100 using the input 
    %as the probability for u. Function 'randpath' is defined at bottom (E.1)
    
    VMax_P1(i) = payout(w,K1,1,u,d,S_0); %Type = 1
end    

% Compute E1(Vmax)
E1_VMax = mean(VMax_P1)

%% Using p1: compute V_N for the European Call Option

VECall_P1 = repmat(0,1,1000); %initialize vector to hold the Vn from each of the 1000 simulations
E1_VECall = repmat(0,1,3); %initialize vector to hold the value for each of the 3 Ks

for k=1:3
    
    for i=1:1000
        w = randpath(p1); %Generates random path of length 100 using the input 
        %as the probability for u. Function 'randpath' is defined at bottom (E.1)

        VECall_P1(i) = payout(w,K(k),2,u,d,S_0); %Type = 2
    end

% Compute E1(VECall)
E1_VECall(k) = mean(VECall_P1)
end
%% Using p1: compute V_N for the European Put Option

VEPut_P1 = repmat(0,1,1000); %initialize vector to hold the Vn from each of the 1000 simulations
E1_VEPut = repmat(0,1,3); %initialize vector to hold the value for each of the 3 Ks

for k=1:3

    for i=1:1000
        w = randpath(p1); %Generates random path of length 100 using the input 
        %as the probability for u. Function 'randpath' is defined at bottom (E.1)

        VEPut_P1(i) = payout(w,K(k),3,u,d,S_0); %Type = 3
    end    
    
% Compute E1(VEPut)
E1_VEPut(k) = mean(VEPut_P1)
end

%% Using p2: compute V_N for the max(S_n) derivative

VMax_P2 = repmat(0,1,1000); %initialize vector to hold the Vn from each of the 1000 simulations

% The K value is not used in the computation for vmax but a K must be
% passed into the randpath function. K1 was used in the function call but is 
% not used in any of the computation.

% Using K1
for i=1:1000
    w = randpath(p2); %Generates random path of length 100 using the input 
    %as the probability for u. Function 'randpath' is defined at bottom (E.1)
    
    VMax_P2(i) = payout(w,K1,1,u,d,S_0); %Type = 1
end    

% Compute E2(Vmax)
E2_VMax = mean(VMax_P2)

%% Using p2: compute V_N for the European Call Option

VECall_P2 = repmat(0,1,1000); %initialize vector to hold the Vn from each of the 1000 simulations
E2_VECall = repmat(0,1,3); %initialize vector to hold the value for each of the 3 Ks

for k=1:3

    for i=1:1000
        w = randpath(p2); %Generates random path of length 100 using the input 
        %as the probability for u. Function 'randpath' is defined at bottom (E.1)

        VECall_P2(i) = payout(w,K(k),2,u,d,S_0); %Type = 2
    end    


% Compute E2(VECall)
E2_VECall(k) = mean(VECall_P2)
end

%% Using p2: compute V_N for the European Put Option

VEPut_P2 = repmat(0,1,1000); %initialize vector to hold the Vn from each of the 1000 simulations
E2_VEPut = repmat(0,1,3); %initialize vector to hold the value for each of the 3 Ks

for k=1:3

    for i=1:1000
        w = randpath(p2); %Generates random path of length 100 using the input 
        %as the probability for u. Function 'randpath' is defined at bottom (E.1)

        VEPut_P2(i) = payout(w,K(k),3,u,d,S_0); %Type = 3
    end    

% Compute E2(VEPut)
E2_VEPut(k) = mean(VEPut_P2)
end


%% (F) Calculates V_0 using martingale method for 3 different derivatives
% Using K1:
V_0_K1 = [0 0 0]; %Vector to hold the V_0 values for Vmax, EuroCall, EuroPut

for Type = 1:3
    V_N = repmat(0,1,1000);
    for i=1:1000
        % Generate random path of N=100 time steps using function E.1 defined
        % at bottom of script
        w = randpath(p_tilda); 
        V_N(i) = payout(w,K1,Type,u,d,S_0);
    end    
V_0_K1(Type) = (1./(1+r))^N*mean(V_N)
end

% Using K2:
V_0_K1 = [0 0 0]; %Vector to hold the V_0 values for Vmax, EuroCall, EuroPut

for Type = 1:3
    V_N = repmat(0,1,1000);
    for i=1:1000
        % Generate random path of N=100 time steps using function E.1 defined
        % at bottom of script
        w = randpath(p_tilda); 
        V_N(i) = payout(w,K2,Type,u,d,S_0);
    end    
V_0_K2(Type) = (1./(1+r))^N*mean(V_N)
end

% Using Ktilda:
V_0_Ktilda = [0 0 0]; %Vector to hold the V_0 values for Vmax, EuroCall, EuroPut

for Type = 1:3
    V_N = repmat(0,1,1000);
    for i=1:1000
        % Generate random path of N=100 time steps using function E.1 defined
        % at bottom of script
        w = randpath(p_tilda); 
        V_N(i) = payout(w,Ktilda,Type,u,d,S_0);
    end    
V_0_Ktilda(Type) = (1./(1+r))^N*mean(V_N)
end

%% (G) Portfolio Strategy Approach
%{
w1 = randpath(0.5);
w2 = randpath(1/2);
w3 = randpath(1/2);

%CHANGE K GETS SAME VALUE!!!
%vmax
X1_0_vmax = rep_port(p_tilda,r,K2,1,u,d,S_0)
X2_0_vmax = rep_port(p_tilda,r,K1,1,u,d,S_0)
X3_0_vmax = rep_port(p_tilda,r,K1,1,u,d,S_0)
% no matter which k used, aleays outputs 1? does this make sense? i am not
% feeding path in so must be a prob with the way i set up repport

%European Call Option
X1_0_EC = rep_port(p_tilda,r,Ktilda,2,u,d,S_0)
X2_0_EC = rep_port(p_tilda,r,K1,2,u,d,S_0)
X3_0_EC = rep_port(p_tilda,r,K1,2,u,d,S_0)

%European Put Option
X1_0_EP_K1 = rep_port(p_tilda,r,K1,3,u,d,S_0)
X2_0_EP_K1 = rep_port(p_tilda,r,K1,3,u,d,S_0)
X3_0_EP_K1 = rep_port(p_tilda,r,K1,3,u,d,S_0)

X1_0_EP = rep_port(p_tilda,r,K2,3,u,d,S_0)
X2_0_EP = rep_port(p_tilda,r,K2,3,u,d,S_0)
X3_0_EP = rep_port(p_tilda,r,K2,3,u,d,S_0)

X1_0 = rep_port(p_tilda,r,Ktilda,3,u,d,S_0)
X2_0 = rep_port(p_tilda,r,Ktilda,3,u,d,S_0)
X3_0 = rep_port(p_tilda,r,Ktilda,3,u,d,S_0)

t = 0:1:99;
%% Vmax
% Delta values for each path
vm_del1 = deltas(u,d,w1,K,1,S_0);
vm_del2 = deltas(u,d,w1,K,2,S_0);
vm_del3 = deltas(u,d,w1,K,3,S_0);

figure
subplot(3,1,1)
plot(t,vm_del1)

subplot(3,1,2)
plot(t,vm_del2)

subplot(3,1,3)
plot(t,vm_del3)
%Issue is that price/s==v, coded something wrong

figure 
plot(t,vm_del1,t,vm_del2,t,vm_del3)
price1 = price(u,d,w1,S_0);
price2 = price(u,d,w2,S_0);
price3 = price(u,d,w3,S_0)

figure
subplot(3,1,1)
plot(t,price1,'o')

subplot(3,1,2)
plot(t,price2,'o')

subplot(3,1,3)
plot(t,price3,'o')

figure 
plot(t,price1,'bo',t,price2,'ro',t,price3,'go')

del = deltas(u,d,[1 0 1 1 0 1 1],S_0*exp(7*(log10(u/d) * p1 + log10(d))),3,S_0)
X_0 = rep_port(p_tilda,r,S_0*exp(7*(log10(u/d) * p1 + log10(d))),3,u,d,S_0)

%% comment on always 1 value bc price increase always
%}

%% PART G FROM EMAIL
% Generate 3 paths with p=1/2
w1 = randpath(0.5);
w2 = randpath(1/2);
w3 = randpath(1/2);

[del_n1,X_n1] = RepPortNew(w1,u,d,S_0,K,Type,r);
[del_n2,X_n2] = RepPortNew(w2,u,d,S_0,K,Type,r);
[del_n3,X_n3] = RepPortNew(w3,u,d,S_0,K,Type,r);

%% (H) 

V_max = [E1_VMax E2_VMax V].';
Chebychev1_Order = [N_Cheb1_Anlg*2 N_Cheb1_Dig*2 N_Cheb1_Imp].';
Chebychev2_Order = [N_Cheb2_Anlg*2 N_Cheb2_Dig*2 N_Cheb2_Imp].';
Elliptic_Order = [N_Ellip_Anlg*2 N_Ellip_Dig*2 N_Ellip_Imp].';

table(Butterworth_Order,Chebychev1_Order,Chebychev2_Order,Elliptic_Order,'RowNames',{'E1(Vn)/(1+r)^N' 'E2(Vn)/(1+r)^N' 'Risk Neutral Price Vo'})


%% PART G X_n One Step Recursion
function [del_n,X_n] = RepPortNew(w,u,d,S_0,K,Type,r)
    %calculate price for all particular path w1...wn
    S_n = price(u,d,w,S_0);
    [r,path_length] = size(S_n);
    
    %calculate Vs
    V_H = payout([w 1],K,Type,u,d,S_0); 
    V_T = payout([w 0],K,Type,u,d,S_0);
    
    %calculate prices
    S_H = S_n(i)*u;
    S_T = S_n(i)*d;
    
    %put above together to solve for del_n
    del_n = (V_H - V_T)./(S_H - S_T);
    
    %By replicating portfolio, X_n = V_n a.s.
    X_n = payout([w],K,Type,u,d,S_0);
    
end

%% (B.1) Function to calculate risk-neutral probability
function p_tilda = RiskNeutral_Prob(r,u,d)
    p_tilda = ((1+r)-d)./(u-d);
end

%% (C.1) Function to calculate price of security
function S_n = price(u,d,w,S_0)
    [r,path_length] = size(w);
    w_ud = w;
    
    % Replace each 1 with value of u and 0 with the value of d 
    w_ud(w_ud==1)=(u);
    w_ud(w_ud==0)=(d);
    
    % Multiply security price at time zero, S_0, with first value in w_ud
    % vector (technically do not need to do this because S_0 was defined to
    % be one; however, I included this line of code to preserve generality
    % for all values of S_0)
    w_ud(1) = S_0 * w_ud(1);

    % Prices at each timestep
    S_n = cumprod(w_ud); 
end

%% (C.2) Function to calculate the replicating portfolio
function X_0 = rep_port(p_tilda,r,K,Type,u,d,S_0)
    q_tilda = 1 - p_tilda;

    % As stated in the problem statement, assume there is a function that 
    % computes V for a given path. The function call follows the form
    % described in part (D).
    V1H = payout(1,K,Type,u,d,S_0); 
    V1T = payout(0,K,Type,u,d,S_0); 
    
    % X_0 = V_0 by the definition of a replicating portfolio. Notationally,
    % Vn(H) means the value of the derivative at timestep = n given the
    % first n-1 tosses and landing a head on the nth toss. Vn(T) is the value
    % of the derivative at timestep = n given the same first n-1 tosses, but
    % landing a tail on the nth toss. 
    X_0 = (1./(1+r)) * (p_tilda * V1H + q_tilda * V1T);
end

%% (C.3) Function to calculate the portfolio strategy
function del = deltas(u,d,w,K,Type,S_0)
    S_n = price(u,d,w,S_0);

    [r,path_length] = size(S_n);
    del = repmat(0,1,path_length);

    for i=1:path_length
        V_H = payout([w(1:i-1) 1],K,Type,u,d,S_0); 
        V_T = payout([w(1:i-1) 0],K,Type,u,d,S_0);

        %shift index over
        S_H = S_n(i)*u;
        S_T = S_n(i)*d;
        
        del(i) = (V_H - V_T)./(S_H - S_T);
    end
end
%% (D.1) Function to calculate the payout
function V_n = payout(w,K,Type,u,d,S_0)
    [r,path_length] = size(w);
    % Compute vector of prices for the given path w
    S_n = price(u,d,w,S_0);
    
    % max S_n
    if Type == 1
        V_n = max(S_n);
    end
    
    % European Call Option
    if Type == 2
        V_n = (S_n(path_length) - K);
        if V_n < 0
            V_n = 0;
        end
    end
    
    % European Put Option
    if Type == 3
        V_n = (K - S_n(path_length));
        if V_n < 0
            V_n = 0;
        end
    end
end

%% (E.1) Function to generate random path
function w = randpath(p) 
    w = zeros(1,100);
    for i = 1:100
        temp = rand();  
        
        if temp < p
        w(i) = 1;
        end
        
        if temp >= p
        w(i) = 0;
        end
    end
end
