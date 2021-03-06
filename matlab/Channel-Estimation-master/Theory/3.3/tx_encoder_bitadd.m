function [encoderDataOut] = tx_encoder_bitadd(encoderDataIn, encoderPara)

BG = encoderPara.BG;
codeRate = encoderPara.codeRate;
Z = encoderPara.Z;
H_K = (encoderPara.H_K - 2) * Z;
K = (encoderPara.K - 2) * Z;
M = encoderPara.M * Z;

dataSec0 = encoderDataIn(1 : K);
dataSec1 = encoderDataIn(K + 1 : K + M);

add_zero = zeros(1,2*Z);

switch BG
    case 1
        switch codeRate
            case 1/3
                dataSeco_add = [];
            case 1/2
                dataSeco_add = [];
            case 2/3
                dataSeco_add = [];
            case 3/4
                dataSeco_add = ones(1, H_K - K) * 32;
            case 5/6
                dataSeco_add = ones(1, H_K - K) * 32;
            case 8/9
                dataSeco_add = ones(1, H_K - K) * 32;
            case 11/21
                dataSeco_add = [];
        end
    case 2
        switch codeRate
            case 1/5
                dataSeco_add = [];
            case 1/3
                dataSeco_add = [];
            case 2/5
                dataSeco_add = [];
            case 1/2
                dataSeco_add = [];
            case 2/3
                dataSeco_add = [];
            case 3/4
                dataSeco_add = ones(1, H_K - K) * 32;
        end
    otherwise
        dataSeco_add = [];
end
 
encoderDataOut = [add_zero,dataSec0, dataSeco_add, dataSec1];