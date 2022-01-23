function [encoderOut, codeBlock_out] = tx_encoder(encoderPara,in_data)

Z = encoderPara.Z;
% N = encoderPara.H_N;
K = encoderPara.H_K;
% M = N - K;
% P_Num = M;
N1 = encoderPara.N ;
M1 = encoderPara.M ;
K1 = encoderPara.K ;
numFrame = encoderPara.numFrame;

H = encoderPara.H;

row = K1 * Z * numFrame;
%codeBlock_in = unifrnd(0, 1, row, 1);
%for i = 1 : 1 : row
%	if (codeBlock_in(i, 1) >= 0.5) 
%       codeBlock_in(i, 1) = 1;
%    else
%        codeBlock_in(i, 1) = 0;
%    end
%end

codeBlock_out = in_data;

Hs1=H(1:4*Z,1:K1*Z);
Hs2=H(4*Z+1:M1*Z,1:K1*Z);
A=H(1:4*Z,K*Z+1:(K+4)*Z);
B=H(4*Z+1:M1*Z,K*Z+1:(K+4)*Z);

data_out = [];
encoderOut = [];
for j = 1 : 1 :numFrame
    S  = in_data((j - 1) * K1 * Z + 1 : j * K1 * Z);
    P1 = mod(inv(A) * Hs1 * S, 2);
    P2 = mod(Hs2 * S + B * P1, 2);
    P = [P1; P2];
%   P = p_all(1 : P_Num * Z, 1);
    data_out_temp = [S; P];%信息位与校验位所有的比特
%   data_out = [ data_out; data_out_temp];
    encoder_data_temp = data_out_temp( 2 * Z + 1 : end, 1);
    encoderOut = [encoderOut; encoder_data_temp];
end

