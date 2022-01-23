clear;clc;
close all;
addpath('./Theory');

M_SNR_OFDM_dB =[0:5:30];

NrRepetitions = 500;
NrTime=50;
QAM_ModulationOrder = 16; % 4 16 64 128 ...
iteration = 4;

%% FBMCģ��
FBMC = Modulation.FBMC(...
    16,...                          % ���ز���
    8,...                           % FBMC������
    15e3,...                        % ���ز���� (Hz)
    15e3*14*12,...                  % ������ Sampling Rates(Samples/s)
    15e3*20,...                     % ��Ƶ��һ���ز� Intermediate frequency first subcarrier (Hz)
    false,...                       % ����ʵֵ�ź� Transmit real valued signal 
    'PHYDYAS-OQAM',...              % ԭ���˲��� (Hermite, PHYDYAS, RRC) and OQAM or QAM, 
    8, ...                          % �ص�ϵ�� (��Ӧԭ�͹���������)
    0, ...                          % ��ʼ����
    true ...                        % ���ദ��
    );

%% OFDMģ��
OFDM = Modulation.OFDM(...
    16,...                          % Number subcarriers
    4,...                           % Number OFDM Symbols
    15e3,...                        % Subcarrier spacing (Hz)
    15e3*14*12,...                  % Sampling rate (Samples/s)
    15e3*20,...                     % Intermediate frequency first subcarrier (Hz)
    false,...                       % Transmit real valued signal
    0, ...                          % Cyclic prefix length (s), LTE: 1/15e3/14
    (8-1/2)*1/15e3*1/2 ...          % Zero guard length (s)
    );

%% PAM��QAMģ��
PAM = Modulation.SignalConstellation(sqrt(QAM_ModulationOrder),'PAM');
QAM = Modulation.SignalConstellation(QAM_ModulationOrder,'QAM');

%% �ŵ�����ģ��
ChannelEstimation_FBMC = ChannelEstimation.PilotSymbolAidedChannelEstimation(...
    'Diamond',...                           % ��Ƶģʽ
    [...                                    % ��ʾ��Ƶģʽ�����ľ���
    FBMC.Nr.Subcarriers,...                 % ���ز���
    4; ...                                  % Ƶ���еĵ�Ƶ���
    FBMC.Nr.MCSymbols,...                   % FBMC/OFDM ������
    5 ...                                   % ʱ���еĵ�Ƶ���
    ],...                                   
    'linear'...                             % ��ֵ��Extrapolation���� 'linear','spline','FullAverage,'MovingBlockAverage',...
    );

%% �鲿��������ģ��
AuxiliaryMethod = ChannelEstimation.ImaginaryInterferenceCancellationAtPilotPosition(...
    'Auxiliary', ...                                    % ����������ʽ
    ChannelEstimation_FBMC.GetAuxiliaryMatrix(2), ...   % ��Ƶ����
    FBMC.GetFBMCMatrix, ...                             % �鲿���ž���
    16, ...                                             % ȡ��28������ĸ���Դ
    2 ...                                               % ��Ƶ�����ݹ���ƫ��
    );

BER_FBMC_Aux = nan(length(M_SNR_OFDM_dB),NrRepetitions);
MSE_FBMC_Aux = nan(length(M_SNR_OFDM_dB),NrRepetitions);
Time_FBMC_Aux = nan(length(M_SNR_OFDM_dB),NrRepetitions);
%save settings.mat AuxiliaryMethod ChannelEstimation_FBMC FBMC M_SNR_OFDM_dB NrRepetitions NrTime PAM QAM QAM_ModulationOrder
for i_rep = 1:NrRepetitions
for i_SNR = 1:length(M_SNR_OFDM_dB)

        SNR_OFDM_dB = M_SNR_OFDM_dB(i_SNR);
        Pn_time = FBMC.PHY.SamplingRate/(FBMC.PHY.SubcarrierSpacing*FBMC.Nr.Subcarriers)*10^(-SNR_OFDM_dB/10);
        Tp=0;
        Tpd=0;

    for t = 1:NrTime
        if mod(t,2)==1
            Tp=Tp+1;
            [BinaryDataStream_FBMC_Aux_signal(:,Tp),xP_FBMC(:,Tp),x_FBMC_Aux(:,:,t),s_FBMC_Aux(:,t)]= FBMC_signal(AuxiliaryMethod,FBMC,PAM,ChannelEstimation_FBMC);
            index(Tp)=t;
            xC_FBMC_Aux(:,Tp) = x_FBMC_Aux(ChannelEstimation_FBMC.PilotMatrix==1);
            %have pilots
        else
            Tpd=Tpd+1;
            [BinaryDataStream_FBMC_Aux_data(:,Tpd),x_FBMC_Aux(:,:,t),s_FBMC_Aux(:,t)]= FBMC_data(AuxiliaryMethod,FBMC,PAM);
        %pure data
        end
        
   end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %   Binarysignal(:,:,i_SNR,i_rep) = BinaryDataStream_FBMC_Aux_signal;
 %   Binarydata(:,:,i_SNR,i_rep) = BinaryDataStream_FBMC_Aux_data;
 %   Lastxp(:,:,i_SNR,i_rep) = xP_FBMC;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% �ŵ� (doubly flat fading and AWGN in accordance with our testbed measurements!)     
        [h,~] = Jakes_Flat(FBMC,NrTime);
        h=abs(h);
        h=h./norm(h,'inf');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    chennel(:,i_SNR,i_rep) = h;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        n_FBMC = sqrt(1/2)*sqrt(Pn_time/2)*(randn(size(s_FBMC_Aux))+1j*randn(size(s_FBMC_Aux)));
        
        Tp = 0;
    for t = 1:NrTime
        r_FBMC_Aux(:,t) = h(t).*s_FBMC_Aux(:,t)+n_FBMC(:,t);
      % r_FBMC_Aux(:,t) = s_FBMC_Aux(:,t);
        %FBMC�ŵ����
        y_FBMC_Aux(:,:,t) = FBMC.Demodulation(r_FBMC_Aux(:,t));

        %�ڵ�Ƶ����LS�ŵ�����BinaryDataStream_FBMC_Aux_signal
        if(mod(t,2)==1)
            y_FBMC_Aux_temp = y_FBMC_Aux(:,:,t);
            Tp = Tp+1;
            hP_LS_FBMC_Aux(Tp) =mean(y_FBMC_Aux_temp(ChannelEstimation_FBMC.PilotMatrix==1)./xC_FBMC_Aux(:,Tp)/...
                sqrt(AuxiliaryMethod.PilotToDataPowerOffset*AuxiliaryMethod.DataPowerReduction));
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     FirstY(:,:,:,i_SNR,i_rep) = y_FBMC_Aux;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% ʹ�ò�ֵ�������ŵ����Ʋ����� MSE
        tic;
        h_FBMC_Aux = interp1(index,hP_LS_FBMC_Aux,(1:NrTime),'linear','extrap');
        
        Time_FBMC_Aux(i_SNR,i_rep)=toc;
        MSE_FBMC_Aux(i_SNR,i_rep)=var(h-(h_FBMC_Aux).');
        
        %����λ�ô��ľ�����շ���
        Tp=0;Tpd=0;
        for t = 1:NrTime
            if mod(t,2)==1
                y_FBMC_Aux_temp=y_FBMC_Aux(:,:,t);
                y_EQ_FBMC_Aux_temp = real(y_FBMC_Aux_temp(AuxiliaryMethod.PilotMatrix==0)./h_FBMC_Aux(t)...
                    /sqrt(AuxiliaryMethod.DataPowerReduction));
                %��������
                Tp=Tp+1;
                DetectedBitStream_FBMC_Aux_signal(:,Tp) = PAM.Symbol2Bit(real(y_EQ_FBMC_Aux_temp(:)));
                BER_FBMC_Aux_signal(Tp)=mean(BinaryDataStream_FBMC_Aux_signal(:,Tp)~=DetectedBitStream_FBMC_Aux_signal(:,Tp));
            else
                y_FBMC_Aux_temp=y_FBMC_Aux(:,:,t);
                y_EQ_FBMC_Aux_temp = real(y_FBMC_Aux_temp./h_FBMC_Aux(t));
                %��������
                Tpd=Tpd+1;
                DetectedBitStream_FBMC_Aux_data(:,Tpd) = PAM.Symbol2Bit(real(y_EQ_FBMC_Aux_temp(:)));
                BER_FBMC_Aux_data(Tp)=mean(BinaryDataStream_FBMC_Aux_data(:,Tp)~=DetectedBitStream_FBMC_Aux_data(:,Tp));
             end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %   IterationOneUsedDetectedSignal(:,:,i_SNR,i_rep) = DetectedBitStream_FBMC_Aux_signal;
     %   IterationOneUsedDetectedData(:,:,i_SNR,i_rep) = DetectedBitStream_FBMC_Aux_data;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% ���� BER
        BER_FBMC_Aux(i_SNR,i_rep)=mean([BER_FBMC_Aux_data,BER_FBMC_Aux_signal]);
        
end
    if mod(i_rep,10)==0
       disp([int2str(i_rep/NrRepetitions*100) '%']);
    end
end
%save BerBefore.mat BER_FBMC_Aux
%save MSEBefore.mat MSE_FBMC_Aux
%save FirstRunData.mat Binarysignal Binarydata Lastxp chennel FirstY  IterationOneUsedDetectedSignal IterationOneUsedDetectedData
%% Plot MSE
figure();
semilogy(M_SNR_OFDM_dB,trimmean(MSE_FBMC_Aux',2),'red -o');
hold on;
xlabel('SNR for OFDM (dB)'); 
ylabel('MSE');
legend('Simulation: FBMC Auxiliary','Location','SouthWest');
grid on;
%% Plot BER and BEP
figure();
semilogy(M_SNR_OFDM_dB,trimmean(BER_FBMC_Aux',2),'red -o');
hold on;
xlabel('SNR for OFDM (dB)'); 
ylabel('BER');
legend('Simulation: FBMC Auxiliary','Location','SouthWest');
grid on;

%% Plot H
figure();
plot(h,'red -');
hold on;
plot(real(h_FBMC_Aux),'black :');
xlabel('Time'); 
ylabel('Magnitude');
legend('h','h-FBMC-Aux');
grid on;

%% Plot Pilot Pattern
figure();
ChannelEstimation_FBMC.PlotPilotPattern(AuxiliaryMethod.PilotMatrix)
title('������Ƶ');

%% Calculate and Plot Expected Transmit Power Over Time
[Power_FBMC_Aux,t_FBMC] = FBMC.PlotTransmitPower(AuxiliaryMethod.PrecodingMatrix*AuxiliaryMethod.PrecodingMatrix');
[Power_OFDM,t_OFDM] = OFDM.PlotTransmitPower;
figure();
plot(t_FBMC,Power_FBMC_Aux,'red');
hold on;
plot(t_OFDM,Power_OFDM,'black ');
legend({'FBMC Auxiliary','OFDM'});
ylabel('Transmit Power');
xlabel('Time(s)');

%% Calculate Power Spectral Density
[PSD_FBMC_Aux,t_FBMC] = FBMC.PlotPowerSpectralDensity(AuxiliaryMethod.PrecodingMatrix*AuxiliaryMethod.PrecodingMatrix');
[PSD_OFDM,t_OFDM] = OFDM.PlotPowerSpectralDensity;
figure();
plot(t_FBMC,10*log10(PSD_FBMC_Aux),'red');
hold on;
plot(t_OFDM,10*log10(PSD_OFDM),'black ');
legend({'FBMC Auxiliary','OFDM'});
ylabel('Power Spectral Density (dB)');
xlabel('Frequency (Hz)');

save('temp_linear_real_50.mat');
