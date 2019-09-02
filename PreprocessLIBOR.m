% This file preprocesses the USD 3-month LIBOR.

% The 3-month LIBOR rate was chosen because according to
% "Investopedia.com,": "There are a total of 35 different LIBOR rates each 
% business day. The most commonly quoted rate is the three-month U.S. dollar 
% rate (usually referred to as the ?current LIBOR rate?)."

% The dataset was downloaded from "Fred.stlouisfed.org," and a link to the
% source is https://fred.stlouisfed.org/series/USD3MTD156N.

% The dataset contains the date and the reported 3-month LIBOR rate.
% In order to utilize the rates in the analysis, the rates must be
% converted to effective daily rates.

% The code below fills in missing data points by averaging the LIBOR rates
% from the immediately preceding and immediate following values. Then all
% values are converted to an effective daily rate. Then the data is divided
% by year and a seperate file is created to store each year's respective
% data.


% Load dates and rates into a cell array
fileID = fopen('LIBOR Historical Data.CSV');
formatSpec = ['%s %s'];
LIBOR_3month = textscan(fileID,formatSpec,'headerlines',1,'Delimiter',',');
fclose(fileID);


%% Split date into its components
formatIn = 'yyyy-mm-dd';
LIBOR_3month{1,1} = datevec(LIBOR_3month{1,1},formatIn);

% Discard month,day values and only keep year. The year is the only portion
% necessary in order to be able to create a seperate files for the data 
% affiliated with each year. Individual day, month information has no relevance to the
% assignment.
LIBOR_3month{1,1} = LIBOR_3month{1,1}(:,1);


%% Convert entire cell array into a matrix
% Date Column:
Data_dates = LIBOR_3month{1,1};

% LIBOR Rate Column:
Data_string = LIBOR_3month{1,2};
% Data is still stored as character arrays so convert into doubles to allow for numerical manipulation
Data_num = repmat(0,4468,1);
for x=1:4468
    if isequal(cell2mat(Data_string(x,1)),'.') == 0
        Data_num(x,1) = str2num(cell2mat(Data_string(x,1)));
    end
end


%% Fill in missing rates

Missing_index = find(Data_num == 0); % list of indexes with missing value
[r,c] = size(Missing_index);
Missing = [Missing_index repmat(0,r,1)]; % insert column for approx value

for x=1:r
    index = Missing(x,1);
    index_b4 = index - 1;
    index_aftr = index + 1;
    
    % Ensure the index does not point to a missing value
    while Data_num(index_b4) == 0
        index_b4 = index_b4 - 1;
    end
    while Data_num(index_aftr) == 0
        index_aftr = index_aftr + 1;
    end
    
    % Calculate average
    approx = (Data_num(index_b4) + Data_num(index_aftr)).*0.5;
    
    % Replace value
    Missing(x,2) = approx;
    Data_num(index,1) = approx;
end


%% Creates seperate files with the daily return data for each year from 
% 2000 - 2005
GenericName = 'LIBOR_daily';
Extension = '.CSV';

Years = 2000:1:2017;

for x = 1:17
    year = Years(x); 
    
    % Generate matrix with particular year's data 
    start_index = max(find(Data_dates<year))+1;
    end_index_matrix = find(Data_dates>year);
    end_index = end_index_matrix(1,1)-1;
    YearMatrix = Data_num([start_index:end_index]); 
    
    % Determine number of trading days in quarter for the particular year.
    % 365/4 is not used because a return only occurs on trading days
    [TotalDayCount,c] = size(YearMatrix);
    QuarterDayCount = TotalDayCount./4;
    
    % Convert 3-month maturity LIBOR rates into effective daily returns
    % Formula: DailyReturn = 100*[(1 + QuarterlyRate/100)^(1/TradingDays_in_Quarter) - 1]
    YearMatrix_dailyrate = 100.*[((1 + YearMatrix./100).^(1./QuarterDayCount)) - 1];
    % GUY: YearMatrix_dailyrate = QuarterDayCount.*[((1 + YearMatrix).^(1./QuarterDayCount)) - 1];
    
    % Generate file name
    Name_Year = num2str(year);
    FileName = [GenericName Name_Year Extension]; 
    
    %Write to file
    dlmwrite(FileName, YearMatrix_dailyrate, 'delimiter', ',');
end
%}