%==========================================================================
% Script to evaluate position estimation performance for multiple models
% Usage:
%   RMSE = testFunction_for_students_MTb(teamName)
% 
% To test different decoding algorithms, call the function from the 
% Command Window by replacing teamName with the name of the folder that 
% contains the corresponding algorithm implementation:
%
%   RMSE = testFunction_for_students_MTb('Autoencoder');
%   RMSE = testFunction_for_students_MTb('Polynomial');
%   RMSE = testFunction_for_students_MTb('KNN');
%   RMSE = testFunction_for_students_MTb('LR');
%   RMSE = testFunction_for_students_MTb('LRA');

function RMSE = testFunction_for_students_MTb(teamName)

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid
ylim([-150 150]);
xlim([-150,150]);

% Train Model
tic;
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'Color', '#01847F'); % Predicted Position (Green)
        plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'Color', '#F9D2E4'); % Actual Position (Pink)
    end
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions); 
elapsedTime = toc;
rmpath(genpath(teamName))
disp(['Elapsed time: ' num2str(elapsedTime) ' seconds']);

end
