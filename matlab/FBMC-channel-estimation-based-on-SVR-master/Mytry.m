clear;
clc;
load settings.mat
load FirstRunData.mat
NrRepetitions =10;
BER_FBMC_Aux = nan(length(M_SNR_OFDM_dB),NrRepetitions);
MSE_FBMC_Aux = nan(length(M_SNR_OFDM_dB),NrRepetitions);
Time_FBMC_Aux = nan(length(M_SNR_OFDM_dB),NrRepetitions);
for i_rep = 1:NrRepetitions
for i_SNR = 1:length(M_SNR_OFDM_dB)
        
        SNR_OFDM_dB = M_SNR_OFDM_dB(i_SNR);
        Pn_time = FBMC.PHY.SamplingRate/(FBMC.PHY.SubcarrierSpacing*FBMC.Nr.Subcarriers)*10^(-SNR_OFDM_dB/10);
        Tp=0;
        Tpd=0;
    for t = 1:NrTime
        if mod(t,2)==1
            Tp=Tp+1;
            [BinaryDataStream_FBMC_Aux_signal(:,Tp),xP_FBMC(:,Tp),x_FBMC_Aux(:,:,t),s_FBMC_Aux(:,t)]= FBMC_signaliteration(IterationOneUsedDetectedSignal(:,Tp,i_SNR,i_rep),Lastxp(:,Tp,i_SNR,i_rep),AuxiliaryMethod,FBMC,PAM,ChannelEstimation_FBMC);
            index(Tp)=t;
            xC_FBMC_Aux(:,Tp) = x_FBMC_Aux(ChannelEstimation_FBMC.PilotMatrix==1);
            %have pilots
        else
            Tpd=Tpd+1;
            [BinaryDataStream_FBMC_Aux_data(:,Tpd),x_FBMC_Aux(:,:,t),s_FBMC_Aux(:,t)]= FBMC_dataiteration(IterationOneUsedDetectedData(:,Tpd,i_SNR,i_rep),AuxiliaryMethod,FBMC,PAM);
        %pure data
        end
    end
    
   % n_FBMC = sqrt(1/2)*sqrt(Pn_time/2)*(randn(size(s_FBMC_Aux))+1j*randn(size(s_FBMC_Aux)));
    Tp = 0;
    
    for t = 1:NrTime
        %FBMC�ŵ����
        y_FBMC_Aux(:,:,t) = FBMC.Demodulation(s_FBMC_Aux(:,t));
        %�ڵ�Ƶ����LS�ŵ�����BinaryDataStream_FBMC_Aux_signal
        if(mod(t,2)==1)
            y_FBMC_Aux_temp = y_FBMC_Aux(:,:,t);%�ŵ����ƻ�������һ�ε�Y
           % y_FBMC_Aux_temp =FirstY(:,:,t,i_SNR,i_rep);
            Tp = Tp+1;
            hP_LS_FBMC_Aux(Tp) =mean(y_FBMC_Aux_temp(ChannelEstimation_FBMC.PilotMatrix==1)./xC_FBMC_Aux(:,Tp)/...
                sqrt(AuxiliaryMethod.PilotToDataPowerOffset*AuxiliaryMethod.DataPowerReduction));
        end
    end
    
    
     %% ʹ�ò�ֵ�������ŵ����Ʋ����� MSE
        tic;
        %%%
        h = chennel(:,i_SNR,i_rep);
        %%%
        h_FBMC_Aux = interp1(index,hP_LS_FBMC_Aux,(1:NrTime),'linear','extrap');
        Time_FBMC_Aux(i_SNR,i_rep)=toc;
        MSE_FBMC_Aux(i_SNR,i_rep)=var(h-(h_FBMC_Aux).');
        %����λ�ô��ľ�����շ��� �õ�һ��ѭ����Y
        Tp=0;Tpd=0;
        for t = 1:NrTime
            if mod(t,2)==1
                y_FBMC_Aux_temp=FirstY(:,:,t,i_SNR,i_rep);
                %y_FBMC_Aux_temp=y_FBMC_Aux(:,:,t);
                y_EQ_FBMC_Aux_temp = real(y_FBMC_Aux_temp(AuxiliaryMethod.PilotMatrix==0)./h_FBMC_Aux(t)...
                    /sqrt(AuxiliaryMethod.DataPowerReduction));
                %��������
                Tp=Tp+1;
                DetectedBitStream_FBMC_Aux_signal(:,Tp) = PAM.Symbol2Bit(real(y_EQ_FBMC_Aux_temp(:)));
                BER_FBMC_Aux_signal(Tp)=mean(Binarysignal(:,Tp,i_SNR,i_rep)~=DetectedBitStream_FBMC_Aux_signal(:,Tp));
            else
                y_FBMC_Aux_temp=FirstY(:,:,t,i_SNR,i_rep);
                %y_FBMC_Aux_temp=y_FBMC_Aux(:,:,t);
                y_EQ_FBMC_Aux_temp = real(y_FBMC_Aux_temp./h_FBMC_Aux(t));
                %��������
                Tpd=Tpd+1;
                DetectedBitStream_FBMC_Aux_data(:,Tpd) = PAM.Symbol2Bit(real(y_EQ_FBMC_Aux_temp(:)));
                BER_FBMC_Aux_data(Tp)=mean(Binarydata(:,Tp,i_SNR,i_rep)~=DetectedBitStream_FBMC_Aux_data(:,Tp));
             end
        end
        %% ���� BER �õ�һ�ε�Binary�Ա�
        BER_FBMC_Aux(i_SNR,i_rep)=mean([BER_FBMC_Aux_data,BER_FBMC_Aux_signal]);
        
end
    if mod(i_rep,10)==0
       disp([int2str(i_rep/NrRepetitions*100) '%']);
    end
end
save BerAfter.mat BER_FBMC_Aux
save MSEAfter.mat MSE_FBMC_Aux
    
    
    
    
    