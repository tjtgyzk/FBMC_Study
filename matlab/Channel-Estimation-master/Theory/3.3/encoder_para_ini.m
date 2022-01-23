function [encoderPara] = encoder_para_ini(AI)

Z = AI.Z;
bH_shift = AI.bH_shift;

[m, n] = size(bH_shift);

for i = 1 : 1 : m
    for j = 1 : 1 : n
        if (bH_shift(i, j) == -1)
            H((i - 1) * Z + 1 : i * Z, (j - 1) * Z + 1 : j * Z) = zeros(Z);
        else
            I = eye(Z);
            H((i - 1) * Z + 1 : i * Z, (j - 1) * Z + 1 : j * Z) = circshift(I, [0, mod(bH_shift(i, j), Z)]);
        end
    end
end

encoderPara.BG = AI.BG;
encoderPara.codeRate = AI.codeRate;
encoderPara.Z = AI.Z;
encoderPara.H_K = AI.H_K;
encoderPara.numFrame = AI.numFrame;
encoderPara.H = H;
encoderPara.N = AI.N;
encoderPara.M = AI.M;
encoderPara.K = AI.K;







