function [AI] = LDPC_para_ini(BG, Z, codeRate, numFrame)

numIter = 10;
alfa = 0.75; %% NMS
% alfa = 1;    %% MinSum
betaMin = 31;
switch BG
    case 1 
        H_K = 22;
        H_matrix_BG = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 1);
        switch codeRate
            case 1/3
                N = 68;
                K = 22;
                M = 46;
            case 1/2
                N = 46;
                K = 22;
                M = 24;
            case 2/3
                N = 35;
                K = 22;
                M = 13;
            case 3/4
                N = 30;
                K = 21;
                M = 9;
            case 5/6
                N = 26;
                K = 20;
                M = 6;
            case 8/9
                N = 20;
                K = 16;
                M = 4;
            case 11/21
                N = 42;
                K = 22;
                M = 20;
        end
    case 2
        H_matrix_BG = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 1);
        H_K = 10;
        switch codeRate
            case 1/5
                N = 52;
                K = 10;
                M = 42;
            case 1/3
                N = 32;
                K = 10;
                M = 22;
            case 2/5
                N = 27;
                K = 10;
                M = 17;
            case 1/2
                N = 22;
                K = 10;
                M = 12;
            case 2/3
                N = 17;
                K = 10;
                M = 7;
            case 3/4
                N = 14;
                K = 9;
                M = 5;
        end
    otherwise
        disp('BG is error');
end

switch Z
    case{2, 4, 8, 16, 32, 64, 128, 256}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 2);
    case{3,6,12,24,48,96,192,384}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 3);
    case{5,10,20,40,80,160,320}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 4);
    case{7,14,28,56,112,224}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 5);
    case{9,18,36,72,144,288}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 6);
    case{11,22,44,88,176,352}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 7);
    case{13,26,52,104,208}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 8);
    case{15,30,60,120,240}
        bH_shift = xlsread(['R1-1711982_BG', num2str(BG), '.xlsx'], 9);
end

AI.BG = BG;
AI.codeRate = codeRate;
AI.Z = Z;
AI.N = N;
AI.M = M;
AI.H_K = H_K;
AI.K = K;
AI.numIter = numIter;
AI.alfa = alfa;
AI.betaMin = betaMin;
AI.numFrame = numFrame;
AI.H_matrix_BG = H_matrix_BG;
AI.bH_shift = bH_shift;


