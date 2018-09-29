clear all;
% This code depends on the NetLab toolbox. It is free available from:
% http://www.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/downloads/
%--------------------------------------------------------------------------
%---------------------- Example one ---------------------------------------
x1 = linspace(-10,10,200);
y = sin(2*pi*x1).*exp(-0.5*x1) + 2*randn(1,200);
sMixR = GMR(x1',y',3,'full');
x2 = linspace(-10,10,1000);
[mExpectations,~] = GMRTest(sMixR,x2',1);
hold on
plot(x1,y,'xr');
title('Gaussian Mixture Regressor - Synthetic data','FontSize',12);
xlabel('x','FontSize',12);
ylabel('y','FontSize',12);
%---------------------- Load Data -----------------------------------------
% Energy efficiency Data Set 
% from http://archive.ics.uci.edu/ml/datasets/Energy+efficiency
load('DataEnergyConsumption.mat');
%--------------------------------------------------------------------------
iNm = size(mData,1);
%--------------------------------------------------------------------------
Error = zeros(10,10);
R2 = zeros(10,10);
%--------------------------------------------------------------------------
for Boostrap = 1:10
    %----------------------------------------------------------------------
    % Split into training and testing subsets
    vIndx = randperm(iNm);
    iNmTrain = ceil(iNm*0.7);
    mDataTrain = mData(vIndx(1:iNmTrain),:);
    mTargetTrain = mTarget(vIndx(1:iNmTrain),:);
    mDataTest = mData(vIndx(iNmTrain+1:end),:);
    mTargetTest = mTarget(vIndx(iNmTrain+1:end),:);
    %----------------------------------------------------------------------
    for Mixtures = 1:10 %Evaluation of different Gaussian in Mixture
        %------------------ Create and Adjust the GMR ---------------------
        sMixR = GMR(mDataTrain,mTargetTrain,Mixtures,'full');
        [mExpectations,~] = GMRTest(sMixR,mDataTest);
        mTem = (mExpectations - mTargetTest).^2;
        Error(Boostrap,Mixtures) = sum(mTem(:))/(iNm - iNmTrain);
        for i = 1:2
            %---- R squared estimation thanks to Jered R Wells ------------
            % http://www.mathworks.com/matlabcentral/fileexchange/34492-r-square--the-coefficient-of-determination
            [r2,~] = rsquare(mExpectations(:,i),mTargetTest(:,i));
            R2(Boostrap,Mixtures) = R2(Boostrap,Mixtures)+r2;
        end
        R2(Boostrap,Mixtures) = R2(Boostrap,Mixtures)/2;
    end
end
Mixtures = 1:10;
ErrorMean = mean(Error);
ErrorDesvia = std(Error);
%--------------------------------------------------------------------------
R2Mean = mean(R2);
R2Desvia = std(R2);
%--------------------------------------------------------------------------
%----------------------------- Plot ---------------------------------------
figure
hAx = plotyy(Mixtures,ErrorMean,Mixtures,R2Mean);
hold(hAx(1), 'on');
errorbar(hAx(1),Mixtures,ErrorMean,ErrorDesvia,'LineWidth',2);
hold(hAx(2), 'on');
errorbar(hAx(2),Mixtures,R2Mean,R2Desvia,'LineWidth',2);
grid on;
title('Results using Gaussian Mixture Regressor - Energy Consumption Dataset','FontSize',12);
xlabel('Number of Mixtures','FontSize',12)
ylabel(hAx(1),'Mean Squared Error','FontSize',12);
ylabel(hAx(2),'Determination Coefficient','FontSize',12);
%--------------------------------------------------------------------------
