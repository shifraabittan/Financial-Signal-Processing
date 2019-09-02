% This file preprocesses the S&P500 daily return historical data. 

% The dataset for 2006-2016 was downloaded from "Investing.com," and a link 
% to the source is https://www.investing.com/indices/us-spx-500-historical-data. 
% The dataset contains the opening, closing, high and low prices as well as
% the percent change in price. The code below extracts the daily returns (percentages) and creates 
% a file to store each year's respective data. All data clean up is to ensure
% the format and ordering matches the FF48 data. 

% Because Investing.com only has data since 2016, the dataset for 2000-2005
% was downloaded from "Quotes.WSJ.com," and a link to the source is
% https://quotes.wsj.com/index/SPX/historical-prices. The dataset contains
% the opening, closing, high and low prices. The code belows computes the
% percent change in price/daily return value and stores each year's in a
% different file.


%% 2006 - 2016 (Investing.com)

% Load dates and daily returns into a cell array
fileID = fopen('S&P 500 Historical Data 2006-2016.CSV');
formatSpec = ['%q' repmat('%*q',1,5) '%*1s %f %*2s'];
Data = textscan(fileID,formatSpec,'headerlines',1,'Delimiter',',');
fclose(fileID);

% Reverse the order of the dataset so that the oldest date is listed first
% and matches the chronology scheme of the FF48 preprocessing
Data{1,1} = flip(Data{1,1});
Data{1,2} = flip(Data{1,2});

% Convert month names into integer equivalences
Data{1,1} = replace(Data{1,1},'Jan','01');
Data{1,1} = replace(Data{1,1},'Feb','02');
Data{1,1} = replace(Data{1,1},'Mar','03');
Data{1,1} = replace(Data{1,1},'Apr','04');
Data{1,1} = replace(Data{1,1},'May','05');
Data{1,1} = replace(Data{1,1},'Jun','06');
Data{1,1} = replace(Data{1,1},'Jul','07');
Data{1,1} = replace(Data{1,1},'Aug','08');
Data{1,1} = replace(Data{1,1},'Sep','09');
Data{1,1} = replace(Data{1,1},'Oct','10');
Data{1,1} = replace(Data{1,1},'Nov','11');
Data{1,1} = replace(Data{1,1},'Dec','12');

% Split date into its components
formatIn = 'mm dd, yyyy';
Data{1,1} = datevec(Data{1,1},formatIn);

% Discard entire date and only keep year. The year is the only portion
% necessary in order to be able to create a seperate file for the data 
% affiliated with each year. Individual day, month information has no relevance to the
% assignment.
Data{1,1} = Data{1,1}(:,1);

% Convert cell array to matrix
Data = cell2mat(Data);
Data_returns = Data(:,2);

% Creates seperate files with the daily return data for each year from 
% 2006 - 2016
GenericName = 'SP500_daily';
Extension = '.CSV';

Data = [2005 0; Data; 2017 0];%To smoothly process edge cases of year=2006,2016
Data_dates = Data(:,1);
Years = 2006:1:2017;

for x = 1:11
    year = Years(x) 
    
    %Generate matrix with particular year's data 
    start_index = max(find(Data_dates<year))+1;
    end_index_matrix = find(Data_dates>year);
    end_index = end_index_matrix(1,1)-1;
    YearMatrix = Data([start_index:end_index],:); 
    
    %Generate file name
    Name_Year = num2str(year);
    FileName = [GenericName Name_Year Extension]; 
    
    %Write to file
    dlmwrite(FileName, YearMatrix, 'delimiter', ',', 'precision', 4);
end


%% 2000 - 2015 (Quotes.WSJ.com)

% Load dates and prices into a cell array
fileID = fopen('S&P 500 Historical Data 2000-2005.CSV');
formatSpec = ['%s' repmat('%f',1,4)];
Data2 = textscan(fileID,formatSpec,'headerlines',1,'Delimiter',',');
fclose(fileID);

% Reverse the order of the dataset so that the oldest date is listed first
% and matches the chronology scheme of the FF48 preprocessing
Data2{1,1} = flip(Data2{1,1});
Data2{1,2} = flip(Data2{1,2});
Data2{1,3} = flip(Data2{1,3});
Data2{1,4} = flip(Data2{1,4});
Data2{1,5} = flip(Data2{1,5});

% Split date into its components
formatIn = 'mm/dd/yy';
Data2{1,1} = datevec(Data2{1,1},formatIn);

% Discard entire date and only keep year. The year is the only portion
% necessary in order to be able to create a seperate file for the data 
% affiliated with each year. Individual day, month information has no relevance to the
% assignment.
Data2{1,1} = Data2{1,1}(:,1);

% Convert cell array to matrix
Data2 = cell2mat(Data2);
Data2_returns = Data2(:,2:5);

% Find daily return as a percentage. Return is calculated as 
% (ClosingPrice_Today - ClosingPrice_Yesterday)/ClosingPrice_Yesterday.
% Multiply by 100 to convert to a percentage. In the WSJ dataset we are
% dealing with, today's opening price = yesterday's closing price. This
% simplifies the return equation to 
% (ClosingPrice - OpeningPrice)/OpeningPrice. (In Investors.com dataset,
% the opening price listed was not equal to closing price of day before and
% must be careful of this when calculating daily returns.
ClosingPrice = Data2_returns(:,4);
OpeningPrice = Data2_returns(:,1);
DailyReturn = (ClosingPrice - OpeningPrice)./OpeningPrice
DailyReturn = DailyReturn.*100

% Creates seperate files with the daily return data for each year from 
% 2000 - 2005
GenericName = 'SP500_daily';
Extension = '.CSV';

ReturnMatrix = [Data2(:,1) DailyReturn];

ReturnMatrix = [1999 0; ReturnMatrix; 2006 0];%To smoothly process edge cases of year=2006,2016
Data2_dates = ReturnMatrix(:,1);
Years = 2000:1:2006;

for x = 1:6
    year = Years(x) 
    
    %Generate matrix with particular year's data 
    start_index = max(find(Data2_dates<year))+1;
    end_index_matrix = find(Data2_dates>year);
    end_index = end_index_matrix(1,1)-1;
    YearMatrix = ReturnMatrix([start_index:end_index],:); 
    
    %Generate file name
    Name_Year = num2str(year);
    FileName = [GenericName Name_Year Extension]; 
    
    %Write to file
    dlmwrite(FileName, YearMatrix, 'delimiter', ',', 'precision', 4);
end
