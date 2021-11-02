clear all;
close all;
clc;
               %%%%%%%参数设置%%%%%%%
numFFT = 1024;                         %FFT点数
numGuards = 212;                       %保护间隔大小
L = numFFT - 2*numGuards;              %有效子载波数
K = 4;                                 %频率扩展点数
QAMstep = 4;                          %QAM阶数
bitsPerSub = log2(QAMstep);            %子载波传输数据数
numBits = L*bitsPerSub/2;              %生成随机数数量
numSymbols = 100;                      %FBMC符号数
snr = 0:0.25:10;                       %信噪比
KF = K*numFFT;
KL = K*L;
a = length(snr);

switch K                               %滤波器参数设置
    case 2
        HkOneSided = sqrt(2)/2;
    case 3
        HkOneSided = [0.911438 0.411438];
    case 4
        HkOneSided = [0.971960 sqrt(2)/2 0.235147];
    otherwise
        return
end
Hk = [fliplr(HkOneSided) 1 HkOneSided];        %滤波器
inpData = zeros(numBits,1);                    %输入数据
QAMData = zeros(numBits/bitsPerSub);           %QAM调制数据
data = zeros(2*length(QAMData),1);             %OQAM调制数据
dataUP = zeros(K*length(data),1);              %频率扩展后数据
sumFBMCSpec = zeros(KF*2, 1);                  %psd
dataREreal = zeros(numBits/bitsPerSub,1);      %接收数据实部
dataREimag = zeros(numBits/bitsPerSub,1);      %接收数据虚部
symBuf = complex(zeros(2*KF, 1));              %叠加器
b=1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for snrdb = 0:0.25:10            
    
            %%%%%%%%%%%发射机%%%%%%%%%%%
for i = 1:1:numSymbols
                    %%%QAM调制%%%
inpData = randi([0 1], numBits, 1);
inpDataMulti(:,i) = inpData; 
QAMData = qammod(inpData,QAMstep,'bin','InputType','bit','UnitAveragePower',true);
                    %%%OQAM调制%%%
data = OQAM_modulater(QAMData,i);
%datashow(:,i) = data;
                    %%%%%频率扩展%%%%%
dataUP(1:K:end) = data;
                   %%%加保护间隔%%%
dataUPG = [zeros(K*numGuards,1);dataUP;zeros(K*numGuards,1)];
                   %%%%过滤波器%%%%
X = filter(Hk,1,dataUPG);
%delaya = grpdelay(mean(Hk));
%delayb = grpdelay(Hk);
%delay = 3;
                   %%%%%群时延%%%%%
X1 = [X(K:end);zeros(K-1,1)];
                    %%%%%IFFT%%%%%
sendResult = fftshift(ifft(X1));
                    %%%叠加发送%%%
symBuf = [symBuf(numFFT/2+1:end); complex(zeros(numFFT/2,1))];
symBuf(KF+(1:KF)) = symBuf(KF+(1:KF)) + sendResult;
sendResultMulti(:,i) = complex(symBuf(1:KF));

[specFBMC, fFBMC] = periodogram(complex(symBuf(1:KF)), hann(KF, 'periodic'), KF*2, 1);
sumFBMCSpec = sumFBMCSpec + specFBMC;
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%信道%%%%%%%%%%%%
%%无信道?
%dataChannel = sendResultMulti;
%%高斯信道
for i = 1:1:numSymbols
dataChannel(:,i) = awgn (sendResultMulti(:,i),snrdb,'measured');
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%接收机%%%%%%%%%%%
for i=1:1:numSymbols
                       %%%%FFT%%%%
%reFFT = fft(fftshift(sendResult))/sqrt(numFFT);
%reFFT = fftshift(fft(sendResult));
reFFT = fft(fftshift(dataChannel(:,i)));
                       %%%滤波器%%%
reFilter = filter(Hk,1,reFFT);
                       %%%群时延%%%
reFilter = [reFilter(K:end);zeros(K-1,1)];
                     %%%去保护间隔%%%
reMoveG = reFilter(numGuards*K+1:end-numGuards*K);
                       %%%频率解扩%%%
reDown = reMoveG(1:K:end);
                      %%%OQAM解调%%%
RE = OQAM_demodulater(reDown,i);
                      %%%QAM解调%%%
Receive = qamdemod(RE,QAMstep,'bin','OutputType','bit');
ReceiveMulti(:,i) = Receive; 
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%误码率%%%%%%%%%%%
BER = comm.ErrorRate;
BER.ReceiveDelay = bitsPerSub*KL;
ber = BER(inpDataMulti(:),ReceiveMulti(:)); 
%[number,ber] = symerr(inpDataMulti,ReceiveMulti) 
Ber(b) = ber(1);
%BER(b) = ber;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = b+1;
end            
            %%%%%%%%%%%psd%%%%%%%%%%%
sumFBMCSpec = sumFBMCSpec/mean(sumFBMCSpec(1+K+2*numGuards*K:end-2*numGuards*K-K));
figure(1);
plot(fFBMC-0.5,10*log10(sumFBMCSpec));
grid on
axis([-0.5 0.5 -180 10]);
xlabel('归一化频率');
ylabel('功率谱密度(dBW/Hz)')
title(['FBMC符号功率谱密度']);
%set(gcf, 'Position', figposition([15 50 30 30]));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
%semilogy(snr,BER);
semilogy(snr,Ber);
grid on;
xlabel('snr/dB');
ylabel('BER');