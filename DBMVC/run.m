close all; clear all; clc
warning off;
addpath(genpath('ClusteringMeasure'));

ResSavePath = 'Res/';
MaxResSavePath = 'maxRes/';

if(~exist(ResSavePath,'file'))
    mkdir(ResSavePath);
    addpath(genpath(ResSavePath));
end

if(~exist(MaxResSavePath,'file'))
    mkdir(MaxResSavePath);
    addpath(genpath(MaxResSavePath));
end

SetParas;

for dataIndex = [7]
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    load(dataName);
    numClust = length(unique(gt));
    
    if dataIndex ~= 7
        [fea] = NormalizeData(fea);
    end
    
    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);
    
%selct hyperparameters
%r1 :\beta r2: \gammma r3:\lammda
    r1=-6:1:1;
     r2=-6:1:1;
       r3=-4;

    acc = zeros(length(r1), length(r2));
    nmi = zeros(length(r1), length(r2));
    purity = zeros(length(r1), length(r2));

    idx = 1;
     fprintf('Please wait a few minutes\n');
    for r1Index = 1 : length(r1)
        r1Temp = r1(r1Index);
        for r2Index = 1 : length(r2)
            r2Temp = r2(r2Index);
            for r3Index = 1 : length(r3)
                r3Temp = r3(r3Index);
                for r4Index = 1 : length(anc_idx)
                    if size(anc_idx{r4Index},2) < 128
                        continue;
                    end
                    r4Temp = r4Index;
             
                       
                    % Main algorithm
                    fprintf('Please wait a few minutes\n');
                   disp(['Dataset: ', datasetName{dataIndex}, ...
                       ', --r1--: ', num2str(r1Temp), ', --r2--: ', num2str(r2Temp)]);
                    tic;
                    [res] = DBMVC_main(fea, numClust, gt, 10^r1Temp, 10^r2Temp, 10^r3Temp, r4Index,anc_idx);
                    Runtime(idx) = toc;
                   disp(['runtime: ', num2str(Runtime(idx))]);
                    idx = idx + 1;
                    tempResBest(1, : ) = res(1, : );
                    tempResStd(1, : ) = res(2, : );
                      acc(r1Index,r2Index)=res(1,7);
                      nmi(r1Index,r2Index)=res(1,4);
                      purity(r1Index,r2Index)=res(1,8);
                    resFile = [ResSavePath datasetName{dataIndex}, '-ACC=', num2str(tempResBest(1, 7)), ...
                        '-r1=', num2str(r1Temp), '-r2=', num2str(r2Temp), '.mat'];
                    save(resFile, 'tempResBest', 'tempResStd');
                    for tempIndex = 1 : 8
                        if tempResBest(1, tempIndex) > ResBest(1, tempIndex)
                            ResBest(1, tempIndex) = tempResBest(1, tempIndex);
                            ResStd(1, tempIndex) = tempResStd(1, tempIndex);
                        end
                    end
                  end
                end
            end
        end

    aRuntime = mean(Runtime);
    resFile2 = [MaxResSavePath datasetName{dataIndex}, '-ACC=', num2str(ResBest(1, 7)), '.mat'];
    save(resFile2, 'ResBest', 'ResStd', 'aRuntime','gt');
    resFile3 = [MaxResSavePath datasetName{dataIndex}, '.mat'];
    save(resFile3, 'ResBest', 'ResStd', 'aRuntime','gt');
    end