%
% Copyright (c) 2013, Research Infinite Solutions llp (http://www.researchinfinitesolutions.com/)
% 
%
% Project Code: YPFZ101
% Project Title: Stock Market prediction using ANFIS
% Publisher:  (http://www.researchinfinitesolutions.com/)
% 
% Developer: Ruchi Mehra (Member of Research Infinite Solutions llp)
% 
% Contact Info: info@researchinfinitesolutions.com, ruchi@webtunix.com

% For training in Stock market and Data Science contact at  ::::
% ruchi@webtunix.com
%% 

close all
clear
clc
%% Import the data
[~, ~, Finadata] = xlsread('australia.csv');
Finadata(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),Finadata)) = {''};


count=1;

for i= 10:2000
    
Dates{i,1}=Finadata{i,1};
PRICE_OPEN(i,1)=Finadata{i,2};
PRICE_HIGH(i,1)=Finadata{i,3};

PRICE_LOW(i,1)=Finadata{i,4};

PRICE_CLOSE(i,1)=Finadata{i+1,5};

end

data=double([PRICE_OPEN PRICE_HIGH PRICE_LOW])/100000;



Inputs = data(30:end,:);
Targets =(double(PRICE_CLOSE(30:end)))/100000;

nData = size(Inputs,1);


%% Shuffling Data

PERM = randperm(nData); % Permutation to Shuffle Data

pTrain=0.85;
nTrainData=round(pTrain*nData);
TrainInd=PERM(1:nTrainData);
TrainInputs=Inputs(TrainInd,:);
TrainTargets=Targets(TrainInd,:);

pTest=1-pTrain;
nTestData=nData-nTrainData;
TestInd=PERM(nTrainData+1:end);
TestInputs=Inputs(TestInd,:);
TestTargets=Targets(TestInd,:);




 nCluster=10;       
Exponent=2;        
MaxIt=200;
MinImprovment=1e-7;
DisplayInfo=1;

FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];

fis=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);


MaxEpoch=200;               
ErrorGoal=0;            
InitialStepSize=0.01;       
StepSizeDecreaseRate=0.9;   
StepSizeIncreaseRate=1.1;    
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);
% pause

%% Apply ANFIS to Data

Outputs=evalfis(Inputs,fis);
TrainOutputs=Outputs(TrainInd,:);
TestOutputs=Outputs(TestInd,:);

%% Error Calculation

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors.^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors.^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

%% Plot Results

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');

figure;
PlotResults(Targets,Outputs,'All Data');

if ~isempty(which('plotregression'))
    figure;
    plotregression(TrainTargets, TrainOutputs, 'Train Data', ...
                   TestTargets, TestOutputs, 'Test Data', ...
                   Targets, Outputs, 'All Data');
    set(gcf,'Toolbar','figure');
end

figure;
gensurf(fis, [1 2], 1, [30 30]);
xlim([min(Inputs(:,1)) max(Inputs(:,1))]);
ylim([min(Inputs(:,2)) max(Inputs(:,2))]);
