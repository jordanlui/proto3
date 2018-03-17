function data = featureExtraction(raw)
% Extracts features from a column of data and builds a struct with contents
% of the analysis.
% Many of the feature extractions were suggested by Picard 2001

% Run parameters
hanningWindow = 20; % Set Hanning window to smooth noise through measurements


data.raw = raw;
data.mean = mean(raw);
data.std = std(raw);
data.norm = (data.raw - data.mean)/data.std;
% First differences
% data.diff1 = data.raw(2:end) - data.raw(1:end-1);
data.diff1 = diff(data.raw,1);
% Mean absolute of 1st diff
data.MAD1 = mean(abs(data.diff1));
% normalized 1st diff
% data.diff1norm = data.raw(2:end) - data.raw(1:end-1);
data.diff1norm = diff(data.norm,1);
% Mean absolute of normed 1st diff
data.MAD1norm = mean(abs(data.diff1norm));
% Second difference
data.diff2 = diff(data.raw,2);
data.MAD2 = mean(abs(data.diff2));
data.diff2norm = diff(data.norm,2);
data.MAD2norm = mean(abs(data.diff2norm));

% Physiology dependent features
data.hannFilt = conv(data.raw,hann(hanningWindow)); 
data.f1 = mean(data.hannFilt);
data.f2 = mean(diff(data.hannFilt,2));
data.f3 = data.hannFilt .* min(data.hannFilt)/(max(data.hannFilt) - min(data.hannFilt));
data.f4 = mean(diff(data.f3,1));
end