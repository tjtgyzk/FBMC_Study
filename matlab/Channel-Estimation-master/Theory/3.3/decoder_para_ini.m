function [decoderPara] = decoder_para_ini(AI) 

    Z = AI.Z;
	H_N = AI.H_K + AI.M;
	H_K = AI.H_K;
	H_M = AI.M;
	M = H_M * Z;
    H_matrix_BG = AI.H_matrix_BG;
    bH_shift = AI.bH_shift;
    
    %//////////////////////// H matrix calculation ///////////////////////%
 
    H_BG = H_matrix_BG(1 : H_M, 1: H_N);
    H_rowWeight0 = sum(H_BG, 2);
    H_weight = sum(H_rowWeight0);
    maxRowWeight = max(H_rowWeight0);
    
    k = 1;
    H_columnIndex0 = zeros(1, H_weight);
    for i = 1 : 1 : H_M
        for j = 1 : 1 : H_N
            if(H_BG(i, j) ~= 0)
                H_columnIndex0(k) = j;
                k = k + 1;
            end
        end
    end
             
    %/////////////////// generate H matrix parameters ////////////////////%
    
    h = 1;
    H_shift = zeros(1, H_weight);
    for i = 1 : 1 : H_M
        for j = 1 : 1 : H_N
            if(bH_shift(i, j) == -1)
            else
                H_shift(h) = bH_shift(i, j);
                h = h + 1;
            end
        end
    end
    
    sumOfRowWeight = 0;
    H_idx = 0;
    H_columnIndex = zeros(1, H_weight * Z);
    
    for i = 1 : 1 : H_M
        for j = 1 : 1 : Z
            for k = 1 : 1: H_rowWeight0(i)
                H_idx = H_idx + 1;
                H_columnIndex(H_idx) = (H_columnIndex0(sumOfRowWeight + k) - 1) * Z + mod((H_shift(sumOfRowWeight + k) + j - 1), Z) + 1; 
            end
        end
        sumOfRowWeight = sumOfRowWeight + H_rowWeight0(i);
    end
    
    H_rowWeight = zeros(1, H_M * Z);
    for i = 1 : 1 : H_M
        for j = 1 : 1 : Z
            H_rowWeight((i-1) * Z + j) = H_rowWeight0(i);
        end
    end
    
    %/////////////////////// generate ctv parameters /////////////////////%
    
    ctvRowIdx = zeros(1, M);
    ctvRowIdx(1) = 1;
    numOnesH = H_rowWeight(1);
    for i = 2 : 1 : M
        numOnesH = numOnesH + H_rowWeight(i);
        ctvRowIdx(i) = ctvRowIdx(i - 1) + H_rowWeight(i - 1);
    end
    
    ctv = H_columnIndex;
    ctvRow = H_rowWeight;
    
    decoderPara.Z = Z;
    decoderPara.M = M;
    decoderPara.H_N = H_N;
	decoderPara.H_K = H_K;
    decoderPara.maxRowWeight = maxRowWeight;
    decoderPara.numIter = AI.numIter;
    decoderPara.alfa = AI.alfa;
    decoderPara.betaMin = AI.betaMin;
    decoderPara.numFrame = AI.numFrame;
    decoderPara.ctv = ctv;
    decoderPara.ctvRow = ctvRow;
    decoderPara.ctvRowIdx = ctvRowIdx;
    
    