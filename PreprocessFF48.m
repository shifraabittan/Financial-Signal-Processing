% This file preprocesses the data from Farma and French benchmark dataset.
% The dataset contains the daily, equally weighted returns for 48 different
% portfolios since 1926. This file/function pulls out the data for one
% particular year, eliminates missing values by averaging the previous and
% next day values and eliminates portfolios with sparse/very little data.

% Load dates and daily returns for all years into a matrix
fileID = fopen('48_Industry_Portfolios_daily.CSV');
formatSpec = repmat('%f',1,49);
Data_Allyears = textscan(fileID,formatSpec,'headerlines',24280,'delimiter',','); %loads into cell array
fclose(fileID);

Data_Allyears = cell2mat(Data_Allyears);

% Eliminate data from 1926 - 1999 and 2017 - 2018
Dates = Data_Allyears(:,1);
start_index = max(find(Dates<20000000))+1;
end_index = min(find(Dates>20170000))-1;
Data_17years = Data_Allyears([start_index:end_index],:);

% Fill in missing values, deliniated in the CSV file as -99.99 or -999, by
% taking average value of the day before and after.

% Check whether any securities should be eliminated because too many
% missing data points
[rows, columns] = size(Data_17years);
Missing1 = repmat(-99.99, rows, columns);
Missing2 = repmat(-999, rows, columns);
check1 = sum(sum(Data_17years == Missing1).') %outputs number of missing data points of type -99.99
check2 = sum(sum(Data_17years == Missing2).') %outputs number of missing data points of type -999
% All data points are present in the subset of data we will work with; no need for correction.

%Number of missing data points in the entire FF48 data set.
%{
[rows_all, columns_all] = size(Data_Allyears);
Missing1_all = repmat(-99.99, rows_all, columns_all);
Missing2_all = repmat(-999, rows_all, columns_all);
check1_all = sum(sum(Data_Allyears == Missing1_all).') % 56465 missing data points
checkB_all = sum(sum(Data_Allyears == Missing2_all).') % 0 missing data points
%}

%Creates seperate files with the daily return data for each year from 2000 - 2017
GenericName = 'FF48_daily';
Extension = '.CSV';

Years = 2000:1:2017;
Years_ymd = 20000000:10000:20170000;

for x = 1:17
    %Variables used in the loop: name of year in various formats
    year = Years(x);
    year_ymd = Years_ymd(x);
    nextyear_ymd = Years_ymd(x+1); 
    
    %Generate matrix with particular year's data 
    start_index = max(find(Dates<year_ymd))+1;
    end_index = min(find(Dates>nextyear_ymd))-1;
    YearMatrix = Data_Allyears([start_index:end_index],:);
    
    %Check to make sure not losing precision, this outputs a 1 for all years
    %YearMatrixTT = YearMatrixT.';
    %isequal(YearMatrix,YearMatrixTT); 
    
    %Generate file name
    Name_Year = num2str(year);
    FileName = [GenericName Name_Year Extension]; 
    
    %Write to file
    dlmwrite(FileName, YearMatrix, 'delimiter', ',', 'precision', 10);
    %using csvwrite truncated date values
end




