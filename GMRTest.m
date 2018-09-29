% function [mExpectations,cVariances] = GMRTest(sMixR,mData)
%
% Estimates predictions for data in mData using the Gaussian Mixture 
% Regressor in sMixR. It is based on the NetLab toolbox.
%
% Inputs:
%
%        sMixR: GMR structure (see: GMR.m)
%
%        mData: represents the data whose expectation is maximized, with
%               each row corresponding to a sample.
%
% See: GMR
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Copyright (c) 2014, Julián David Arias Londoño All rights reserved. %%%
%%%%%%%%%%%%%%%%%%% Department of Systems Engineering %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Universidad de Antioquia, Colombia %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  2014  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [Expectations,Variances] = GMRTest(sMixR,mData,bFlag)

if nargin < 3, bFlag = 0; end

iNx = sMixR.iNx;
iNy = sMixR.iNy;
iM = length(sMixR.mCovars);
%--------------------------- Verification ---------------------------------
if size(mData,2) ~= iNx
    error('Incorrect number of variables');
end
%--------------------------------------------------------------------------
%---------------------- Inicialization ------------------------------------
iNdata = size(mData,1);
Variances = cell(iNdata,1);
%---------------------- Normalization -------------------------------------
mData2 = mData;
if sMixR.bFlagNorm
    mData = (mData - repmat(sMixR.vMeanX,iNdata,1))./repmat(sMixR.vSigmaX,iNdata,1);
end
%--------------------------------------------------------------------------
%--------- See: Gaussian Mixture Regression and Classification ------------
%               by Hsi Guang Sung, PhD Thesis, Rice University, Texas,2004-
%--------------------------------------------------------------------------
mW = gmmpost(sMixR.sMixX,mData);
mMeans = zeros(iNdata,iNy,iM);
cVariance = cell(iNdata,1);
for j = 1:iM
    mMeans(:,:,j) = repmat(sMixR.mCentresY(j,:),iNdata,1) + ...
        (sMixR.mCovars(j).mCovarsYX*(sMixR.mCovars(j).mCovarsXX\...
        ((mData-repmat(sMixR.mCentresX(j,:),iNdata,1))')))';
    cVariance{j} = sMixR.mCovars(j).mCovarsYY - sMixR.mCovars(j).mCovarsYX*...
        (sMixR.mCovars(j).mCovarsXX\sMixR.mCovars(j).mCovarsXY);
end
%--------------------------------------------------------------------------
Expectations = zeros(iNdata,iNy);
for i = 1:iNdata
    vTem = zeros(1,iNy);
    mTem = zeros(iNy,iNy);
    for j = 1:iM
        vTem = vTem + mW(i,j)*mMeans(i,:,j);
        mTem = mTem + mW(i,j)*(mMeans(i,:,j)'*mMeans(i,:,j) + cVariance{j});
    end
    if sMixR.bFlagNorm
        Expectations(i,:) = (vTem.*sMixR.vSigmaY) + sMixR.vMeanY;
        Variances{i} = (mTem - vTem'*vTem).*(sMixR.vSigmaY'*sMixR.vSigmaY);
    else
        Expectations(i,:) = vTem;
        Variances{i} = mTem - vTem'*vTem;
    end
end
if iNy == 1
    Variances = cell2mat(Variances);
end
%--------------------------------------------------------------------------
%--------------------- Picture Special Case iNx = 1, iNy = 1 --------------
if bFlag
   if iNx == 1 || iNy == 1
       figure
       s2 = Variances;
       m = Expectations;
       f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
       fill([mData2; flipdim(mData2,1)], f, [7 7 7]/8)
       hold on; plot(mData2, m);
   end
end




