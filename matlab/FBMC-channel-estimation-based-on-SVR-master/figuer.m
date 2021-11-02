clear;clc;
load bertest.mat
ber = BER_FBMC_Aux;
load linear.mat
%% Plot MSE
figure();
semilogy(M_SNR_OFDM_dB,trimmean(MSE_FBMC_Aux',2),'red -o');
hold on;
%semilogy(M_SNR_OFDM_dB_morePoints,BEP_perfect','black');
xlabel('SNR for OFDM (dB)'); 
ylabel('MSE');
legend('Simulation: FBMC Auxiliary','Location','SouthWest');
grid on;
%% Plot BER and BEP
figure();
hold on;
semilogy(M_SNR_OFDM_dB,trimmean(BER_FBMC_Aux',2),'red -o');
semilogy(M_SNR_OFDM_dB,trimmean(ber',2),'blue -o');

xlabel('SNR for OFDM (dB)'); 
ylabel('BER');
legend('Simulation: FBMC Auxiliary','Location','SouthWest');
grid on;

%% Plot H
figure();
%plot(abs(h),'red -');
plot(h,'red -');
hold on;
%plot(abs(h_FBMC_Aux),'green --');
%plot(angle(h),'blue -.');
%plot(angle(h_FBMC_Aux),'black :');
plot(real(h_FBMC_Aux),'black :');
xlabel('Time'); 
%ylabel('Magnitude and Phase');
ylabel('Magnitude');
%legend('h-Mag','h-FBMC-Aux-Mag',...
%    'h-Pha','h-FBMC-Aux-Pha','Location','SouthWest');
legend('h','h-FBMC-Aux');
grid on;

%% Plot Pilot Pattern
figure();
ChannelEstimation_FBMC.PlotPilotPattern(AuxiliaryMethod.PilotMatrix)
title('FBMC Auxiliary');

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
