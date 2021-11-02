clear all;
close all;
clc;
               %%%%%%%��������%%%%%%%
numFFT = 1024;                         %FFT����
numGuards = 212;                       %���������С
L = numFFT - 2*numGuards;              %��Ч���ز���
K = 4;                                 %Ƶ����չ����
QAMstep = 4;                          %QAM����
bitsPerSub = log2(QAMstep);            %���ز�����������
numBits = L*bitsPerSub/2;              %�������������
numSymbols = 100;                      %FBMC������
snr = 0:0.25:10;                       %�����
KF = K*numFFT;
KL = K*L;
a = length(snr);

switch K                               %�˲�����������
    case 2
        HkOneSided = sqrt(2)/2;
    case 3
        HkOneSided = [0.911438 0.411438];
    case 4
        HkOneSided = [0.971960 sqrt(2)/2 0.235147];
    otherwise
        return
end
Hk = [fliplr(HkOneSided) 1 HkOneSided];        %�˲���
inpData = zeros(numBits,1);                    %��������
QAMData = zeros(numBits/bitsPerSub);           %QAM��������
data = zeros(2*length(QAMData),1);             %OQAM��������
dataUP = zeros(K*length(data),1);              %Ƶ����չ������
sumFBMCSpec = zeros(KF*2, 1);                  %psd
dataREreal = zeros(numBits/bitsPerSub,1);      %��������ʵ��
dataREimag = zeros(numBits/bitsPerSub,1);      %���������鲿
symBuf = complex(zeros(2*KF, 1));              %������
b=1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for snrdb = 0:0.25:10            
    
            %%%%%%%%%%%�����%%%%%%%%%%%
for i = 1:1:numSymbols
                    %%%QAM����%%%
inpData = randi([0 1], numBits, 1);
inpDataMulti(:,i) = inpData; 
QAMData = qammod(inpData,QAMstep,'bin','InputType','bit','UnitAveragePower',true);
                    %%%OQAM����%%%
data = OQAM_modulater(QAMData,i);
%datashow(:,i) = data;
                    %%%%%Ƶ����չ%%%%%
dataUP(1:K:end) = data;
                   %%%�ӱ������%%%
dataUPG = [zeros(K*numGuards,1);dataUP;zeros(K*numGuards,1)];
                   %%%%���˲���%%%%
X = filter(Hk,1,dataUPG);
%delaya = grpdelay(mean(Hk));
%delayb = grpdelay(Hk);
%delay = 3;
                   %%%%%Ⱥʱ��%%%%%
X1 = [X(K:end);zeros(K-1,1)];
                    %%%%%IFFT%%%%%
sendResult = fftshift(ifft(X1));
                    %%%���ӷ���%%%
symBuf = [symBuf(numFFT/2+1:end); complex(zeros(numFFT/2,1))];
symBuf(KF+(1:KF)) = symBuf(KF+(1:KF)) + sendResult;
sendResultMulti(:,i) = complex(symBuf(1:KF));

[specFBMC, fFBMC] = periodogram(complex(symBuf(1:KF)), hann(KF, 'periodic'), KF*2, 1);
sumFBMCSpec = sumFBMCSpec + specFBMC;
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%�ŵ�%%%%%%%%%%%%
%%���ŵ�?
%dataChannel = sendResultMulti;
%%��˹�ŵ�
for i = 1:1:numSymbols
dataChannel(:,i) = awgn (sendResultMulti(:,i),snrdb,'measured');
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%���ջ�%%%%%%%%%%%
for i=1:1:numSymbols
                       %%%%FFT%%%%
%reFFT = fft(fftshift(sendResult))/sqrt(numFFT);
%reFFT = fftshift(fft(sendResult));
reFFT = fft(fftshift(dataChannel(:,i)));
                       %%%�˲���%%%
reFilter = filter(Hk,1,reFFT);
                       %%%Ⱥʱ��%%%
reFilter = [reFilter(K:end);zeros(K-1,1)];
                     %%%ȥ�������%%%
reMoveG = reFilter(numGuards*K+1:end-numGuards*K);
                       %%%Ƶ�ʽ���%%%
reDown = reMoveG(1:K:end);
                      %%%OQAM���%%%
RE = OQAM_demodulater(reDown,i);
                      %%%QAM���%%%
Receive = qamdemod(RE,QAMstep,'bin','OutputType','bit');
ReceiveMulti(:,i) = Receive; 
end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%������%%%%%%%%%%%
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
xlabel('��һ��Ƶ��');
ylabel('�������ܶ�(dBW/Hz)')
title(['FBMC���Ź������ܶ�']);
%set(gcf, 'Position', figposition([15 50 30 30]));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
%semilogy(snr,BER);
semilogy(snr,Ber);
grid on;
xlabel('snr/dB');
ylabel('BER');