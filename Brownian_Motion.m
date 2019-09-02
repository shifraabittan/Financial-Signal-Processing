% Shifra Abittan
% Q2) Brownian Motion

% See bottom of script for solution to Part (A)
% See bottom of script for solution to Part (B)

%% (C)

N = 10^5;
L = 1000;
BM = BrownianMotion(L,N);

%% For m = 0.5
% Probability(T_m > 1)
T_m = zeros(1,1000);
for i = 1:1000 %calculate T_m for each of the 1000 paths
    [T_m(1,i), ~] = Reflect(BM(i,:),0.5,N);
end
[col,row] = size(find(T_m ~= inf)); 
Prob05 = ((1000-row)./L);
% Expectation
E05 = sum(T_m(find(T_m ~= inf)))./row;

%% For m = 1
% Probability(T_m > 1)
T_m = zeros(1,1000);
for i = 1:1000 %calculate T_m for each of the 1000 paths
    [T_m(1,i), ~] = Reflect(BM(i,:),1,N);
end
[col,row] = size(find(T_m ~= inf));

Prob1 = ((1000-row)./L);
% Expectation
E1 = sum(T_m(find(T_m ~= inf)))./row;

%% For m = 2
% Probability(T_m > 1)
T_m = zeros(1,1000);
for i = 1:1000 %calculate T_m for each of the 1000 paths
    [T_m(1,i), ~] = Reflect(BM(i,:),2,N);
end
[col,row] = size(find(T_m ~= inf)); 
Prob2 = ((1000-row)./L);
% Expectation
E2 = sum(T_m(find(T_m ~= inf)))./row;

%% (D) Graph

% For m = 0.5
[~, reflected1] = Reflect(BM(1,:),0.5,N);
[~, reflected2] = Reflect(BM(2,:),0.5,N);
[~, reflected3] = Reflect(BM(3,:),0.5,N);

figure
time = linspace(0,1,N); % x axis

subplot(1,3,1)
sgtitle('Brownian Motion and the Reflected Path when m = 0.5')
plot(time,BM(1,:),time,reflected1,time,repmat(0.5,1,N))
legend('Brownian Motion','Reflected Motion','m = 0.5')
xlabel('Time')

subplot(1,3,2)
plot(time,BM(2,:),time,reflected2,time,repmat(0.5,1,N))
xlabel('Time')

subplot(1,3,3)
plot(time,BM(3,:),time,reflected3,time,repmat(0.5,1,N))
xlabel('Time')

% For m = 1
[~, reflected4] = Reflect(BM(4,:),1,N);
[~, reflected5] = Reflect(BM(5,:),1,N);
[~, reflected6] = Reflect(BM(6,:),1,N);

figure
subplot(1,3,1)
sgtitle('Brownian Motion and the Reflected Path when m = 1')
plot(time,BM(4,:),time,reflected4,time,repmat(1,1,N))
legend('Brownian Motion','Reflected Motion','m = 1')
xlabel('Time')

subplot(1,3,2)
plot(time,BM(5,:),time,reflected5,time,repmat(1,1,N))
xlabel('Time')

subplot(1,3,3)
plot(time,BM(6,:),time,reflected6,time,repmat(1,1,N))
xlabel('Time')

% For m = 2
[~, reflected7] = Reflect(BM(7,:),2,N);
[~, reflected8] = Reflect(BM(8,:),2,N);
[~, reflected9] = Reflect(BM(9,:),2,N);

figure
subplot(1,3,1)
sgtitle('Brownian Motion and the Reflected Path when m = 2')
plot(time,BM(7,:),time,reflected7,time,repmat(2,1,N))
legend('Brownian Motion','Reflected Motion','m = 2')
xlabel('Time')

subplot(1,3,2)
plot(time,BM(8,:),time,reflected8,time,repmat(2,1,N))
xlabel('Time')

subplot(1,3,3)
plot(time,BM(9,:),time,reflected9,time,repmat(2,1,N))
xlabel('Time')

%% (E-1)

Wall1 = Wall(BM(1,:),N);
Wall2 = Wall(BM(2,:),N);
Wall3 = Wall(BM(3,:),N);
Wall4 = Wall(BM(4,:),N);
Wall5 = Wall(BM(5,:),N);

figure
plot(time,Wall1,time,Wall2,time,Wall3,time,Wall4,time,Wall5)
title('Sausage Problem: Five Brownian Motions that Reflect Upon Reaching a = -1 and b = +1')
xlabel('Time')

%% (E-2)
time2firstref = zeros(1,1000);
countref = zeros(1,1000);

for x = 1:1000
    [~,refhappen] = Wall(BM(x,:),N);
    if isempty(find(refhappen,1))
        time2firstref(x) = 0;
    end
    if ~isempty(find(refhappen,1))
        time2firstref(x) = find(refhappen,1);
    end
    countrefs(x) = sum(refhappen);
end

avgtime2first = mean(time2firstref)
avgnumref = mean(countref)

%% (E-3)
BM_E3 = BrownianMotion(10^5,10000);
Inside = ((BM_E3>=-1) & (BM_E3<=1));
CollapsePath = sum(Inside,2);
RemoveZero = (CollapsePath ~= 0);
StaysIn = sum(RemoveZero);

ProbStaysIn = StaysIn./(10^5)

%% (A.1) Function to Generate L Brownian Motions
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
    
    iid_neg = iid_neg*(1/(N^(0.5)));
    % Computes the value of the walk at each time step
    BM = cumsum(iid_neg,2);
    
    % Start the motion at zero
    start = zeros(L,1);
    BM = [start BM];
end
%}

%% (B.1) Function to generate 1st Passage Time and Reflected Path
function [T_m, reflected] = Reflect(BM,m,N) %BM is one path
    % Calculate the First Passage Time
    T = 1; %time interval is from 0 to 1
    k = min(find(BM>=m)); %returns the index of the first passage time
    T_m = k./N; %converts index to proper time scale for [0,1]
    
    if isempty(T_m) %if the brownian motion never reaches m, T_m = +infinity
        T_m = inf;
    end
    
    % Calculates the Reflected Path
    if T_m == inf
        reflected = BM;
    else
        epsilon = BM - m;
        reflect = m - epsilon;
        reflected = [BM(1:k) reflect(k+1:end)];
    end
end
    
%% (E.1)
function [wallpath,idx_ref] = Wall(BM,N) %BM is one path
    wallpath = BM;
    idx_ref = zeros(1,N);
    for i=1:N
        if wallpath(i)>1
            epsilon = wallpath - 1;
            reflect = 1 - epsilon;
            wallpath = [wallpath(1:i-1) reflect(i:end)];
            
            idx_ref(i) = 1;
        end
        if wallpath(i)<(-1)
            epsilon = (-1) - wallpath;
            reflect = (-1) + epsilon;
            wallpath = [wallpath(1:i-1) reflect(i:end)];
            
            idx_ref(i) = 1;
        end
    end
end
    
