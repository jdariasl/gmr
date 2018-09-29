% function sMixR = GMR(mData,mTarget,iM)
%
% Creates a Gaussian Mixture Regressor (GMR) for single or multiple outputs, using
% Maximum likelihood as adjusting criterion. It is based on the NetLab toolbox.
%
% Inputs:
%
%        mData: represents the data whose expectation is maximized, with
%               each row corresponding to a sample.
%
%        mTarget: represents the target variables, with each row corresponding 
%               to a sample.
%
%        iM: Number of Components in Mixture
%        
%        sCovarType: The mixture model type defines the covariance structure 
%                   of each component  Gaussian:
%                   'spherical' = single variance parameter for each component: 
%                                 stored as a vector
%               	'diag' = diagonal matrix for each component: stored as 
%                            rows of a matrix. Default option.
%                	'full' = full matrix for each component: stored as 3d array
%
% Ouputs:
%
%       sMixR is  a MatLab structure containing the GMR model.
%
% See: GMRtest
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Copyright (c) 2014, Julián David Arias Londoño All rights reserved. %%%
%%%%%%%%%%%%%%%%%%% Department of Systems Engineering %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Universidad de Antioquia, Colombia %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  2014  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function sMixR = GMR(mData,mTarget,iM,sCovarType)

if nargin < 4, sCovarType = 'diag'; end
bFlagNorm = 1;
%--------------------------------------------------------------------------
iNx = size(mData,2);
iNy = size(mTarget,2);
%--------------------------------------------------------------------------
sMixR.iNx = iNx;
sMixR.iNy = iNy;
%--------------------------------------------------------------------------
if bFlagNorm
    [mDataT,vMean,vSigma] = zscore([mData,mTarget]);
else
    mDataT = [mData,mTarget];
    vMean = zeros(1,iNx+iNy);
    vSigma = ones(1,iNx+iNy);
end
sMixR.bFlagNorm = bFlagNorm;
sMixR.vMeanX = vMean(1:iNx);
sMixR.vSigmaX = vSigma(1:iNx);
sMixR.vMeanY = vMean(iNx+1:end);
sMixR.vSigmaY = vSigma(iNx+1:end);
%--------------------- Initialization ------------------------------------- 
sMix = gmm(iNx + iNy, iM, sCovarType);
vOptions = foptions;
vOptions(14) = 20; % maximum number of iterations
vOptions(1) = -1; % Switch off all messages, including warning
sMix = gmminit(sMix, mDataT, vOptions);
%---------------------- Adjusting -----------------------------------------
vOptions(14) = 100; % maximum number of iterations
vOptions(5) = 1; % Reset covariance matrices when singular
%vOptions(1) = 1; % Switch messages on; 
try
    sMix = gmmem(sMix, mDataT, vOptions);
catch
    disp('k-means instead of EM');
end
%--------------------------------------------------------------------------
sMixR.sMixOriginal = sMix;
sMixR.mCentresX = sMix.centres(:,1:iNx);
sMixR.mCentresY = sMix.centres(:,iNx+1:end);
%--------------------------------------------------------------------------
%---------------------- Mixture of mData features -------------------------
sMixX = gmm(iNx, iM, sCovarType);
sMixX.centres = sMixR.mCentresX;
sMixX.priors = sMix.priors;
%--------------------------------------------------------------------------
%--------------------- Conditional Covarinces -----------------------------
switch sCovarType
    case 'diag'
        sMixX.covars =  sMix.covars(:,1:iNx);
        for j = 1:iM
            sMixR.mCovars(j).mCovarsXX = diag(sMix.covars(j,1:iNx));  
            sMixR.mCovars(j).mCovarsYY = diag(sMix.covars(j,iNx+1:end));
            sMixR.mCovars(j).mCovarsXY = zeros(iNx,iNy);
            sMixR.mCovars(j).mCovarsYX = zeros(iNy,iNx);
        end
        
    case 'full'
        sMixX.covars =  sMix.covars(1:iNx,1:iNx,:);
        for j = 1:iM
            sMixR.mCovars(j).mCovarsXX = sMix.covars(1:iNx,1:iNx,j);
            sMixR.mCovars(j).mCovarsYY = sMix.covars(iNx+1:end,iNx+1:end,j);
            sMixR.mCovars(j).mCovarsXY = sMix.covars(1:iNx,iNx+1:end,j);
            sMixR.mCovars(j).mCovarsYX = sMix.covars(iNx+1:end,1:iNx,j);
        end
        
    case 'spherical'
        sMixX.covars = sMix.covars; 
        for j = 1:iM
            sMixR.mCovars(j).mCovarsXX = diag(ones(1,iNx)*sMix.covars(j));
            sMixR.mCovars(j).mCovarsYY = diag(ones(1,iNy)*sMix.covars(j));
            sMixR.mCovars(j).mCovarsXY = zeros(iNx,iNy);
            sMixR.mCovars(j).mCovarsYX = zeros(iNy,iNx);
        end       
end
%--------------------------------------------------------------------------
sMixR.sMixX = sMixX;       