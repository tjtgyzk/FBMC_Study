clear all;
close all;
clc;
               %%%%%%%参数/滤波器设�?%%%%%%%
numFFT = 1024;                         %FFT点数(插�?�前)
numGuards = 212;                       %保护间隔
L = numFFT - 2*numGuards;              %有效子载波数
K = 4;                                 %扩展因子
QAMstep = 4;                           % QAM调制阶数
bitsPerSub = log2(QAMstep);            %每个子载波的数据�?
numBits = L/bitsPerSub;                %随机数生成数�?
numSymbols = 10;                      %FBMC符号�?
snr = 0:0.25:7;                        %信噪�?
KF = K*numFFT;
a = length(snr);

switch K                               %针对不同K的滤波器系数�?
    case 2
        HkOneSided = sqrt(2)/2;
    case 3
        HkOneSided = [0.911438 0.411438];
    case 4
        HkOneSided = [0.971960 sqrt(2)/2 0.235147];
    otherwise
        return
end
Hk = [fliplr(HkOneSided) 1 HkOneSided];  %滤波器实�?
inpData = zeros(L,1);                    %生成数矩�?
QAMData = zeros(L/bitsPerSub);           %QAM调制后矩�?
data = zeros(L/bitsPerSub*2,1);          %实虚分离后数据矩�?
dataUP = zeros(L/bitsPerSub*2*K,1);      %插�?�后矩阵
sumFBMCSpec = zeros(K*numFFT*2, 1);      %功率�?
dataREreal = zeros(L/log2(QAMstep),1);   %接收端实信号
dataREimag = zeros(L/log2(QAMstep),1);   %接收端虚信号
b=1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for snrdb = 0:0.25:7            
    
            %%%%%%%%%%%发射�?%%%%%%%%%%%
for i = 1:1:numSymbols
                    %%%QAM调制%%%
inpData = randi([0 1], L, 1);
inpDataMulti(:,i) = inpData; 
QAMData = qammod(inpData,QAMstep,'bin','InputType','bit','UnitAveragePower',true);
%QAMDataMulti(:,i) = QAMData;
                    %%%OQAM调制%%%
if rem(i,2)==1
    data(1:2:end) = real(QAMData);
    data(2:2:end) = imag(QAMData)*1i;
else
    data(1:2:end) = imag(QAMData)*1i;
    data(2:2:end) = real(QAMData);
end
datashow(:,i) = data;
                    %%%%%插�??%%%%%
dataUP(1:K:end) = data;
                   %%%加保护间�?%%%
dataUPG = [zeros(K*numGuards,1);dataUP;zeros(K*numGuards,1)];
                   %%%%过滤波器%%%%
X = filter(Hk,1,dataUPG);
%delaya = grpdelay(mean(Hk));
%delayb = grpdelay(Hk);
%delay = 3;
                   %%%%%群延�?%%%%%
X1 = [X(K:end);zeros(K-1,1)];
                    %%%%%IFFT%%%%%
sendResult = fftshift(ifft(X1));
sendResultMulti(:,i) = sendResult;
[specFBMC, fFBMC] = periodogram(sendResult, hann(KF, 'periodic'), KF*2, 1);
sumFBMCSpec = sumFBMCSpec + specFBMC;
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%信道%%%%%%%%%%%%
%%无信�?
%dataChannel = sendResultMulti;
%%高斯
for i = 1:1:numSymbols
dataChannel(:,i) = awgn (sendResultMulti(:,i),snrdb,'measured');
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%接收�?%%%%%%%%%%%
for i=1:1:numSymbols
                       %%%%FFT%%%%
%reFFT = fft(fftshift(sendResult))/sqrt(numFFT);
%reFFT = fftshift(fft(sendResult));
reFFT = fft(fftshift(dataChannel(:,i)));
                       %%%滤波�?%%%
reFilter = filter(Hk,1,reFFT);
                       %%%群延�?%%%
reFilter = [reFilter(K:end);zeros(K-1,1)];
                     %%%去保护间�?%%%
reMoveG = reFilter(numGuards*K+1:end-numGuards*K);
                       %%%下采�?%%%
reDown = reMoveG(1:K:end);
                      %%%OQAM解调%%%
R2 = real(reDown);
I2 = imag(reDown);
if rem(i,2)==1
    dataREreal(1:1:end) = R2(1:2:end);
    dataREimag(1:1:end) = I2(2:2:end);
else
    dataREreal(1:1:end) = R2(2:2:end);
    dataREimag(1:1:end) = I2(1:2:end);
end
                      %%%QAM解调%%%
RE = complex(dataREreal,dataREimag);
Receive = qamdemod(RE,QAMstep,'bin','OutputType','bit');
ReceiveMulti(:,i) = Receive; 
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%误码�?%%%%%%%%%%%
BBER = comm.ErrorRate;
bber = BBER(inpDataMulti(:),ReceiveMulti(:)); 
%[number,ber] = symerr(inpDataMulti,ReceiveMulti) 
BBB(b) = bber(1);
%BER(b) = ber;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = b+1;
end            
            %%%%%%%%%%%频谱�?%%%%%%%%%%%
sumFBMCSpec = sumFBMCSpec/mean(sumFBMCSpec(1+K+2*numGuards*K:end-2*numGuards*K-K));
figure(1);
plot(fFBMC-0.5,10*log10(sumFBMCSpec));
grid on
axis([-0.5 0.5 -180 10]);
xlabel('归一化频�?');
ylabel('功率谱密�? (dBW/Hz)')
title(['FBMC符号的功率谱密度分布']);
%set(gcf, 'Position', figposition([15 50 30 30]));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
%semilogy(snr,BER);
semilogy(snr,BBB);
grid on;
xlabel('snr/dB');
ylabel('BER');