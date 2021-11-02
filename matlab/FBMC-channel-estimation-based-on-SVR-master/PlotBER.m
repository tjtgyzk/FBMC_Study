M_SNR_OFDM_dB =[0:5:30];
load BerBefore;
before = BER_FBMC_Aux;
load BerAfter;
after = BER_FBMC_Aux;
load MSEBefore;
Mbefore = MSE_FBMC_Aux;
load MSEAfter;
Mafter = MSE_FBMC_Aux;
%% Plot BER and BEP
figure();
hold on;
semilogy(M_SNR_OFDM_dB,trimmean(before',2),'red -o');
semilogy(M_SNR_OFDM_dB,trimmean(after',2),'blue -o');
%semilogy(M_SNR_OFDM_dB,before,'red -o');
%semilogy(M_SNR_OFDM_dB,after,'blue -o');

xlabel('SNR for OFDM (dB)'); 
ylabel('BER');
legend('Simulation: FBMC Auxiliary','Location','SouthWest');
grid on;
%% Plot MSE
figure();
hold on;
semilogy(M_SNR_OFDM_dB,trimmean(Mbefore',2),'red -o');
semilogy(M_SNR_OFDM_dB,trimmean(Mafter',2),'blue -o');
xlabel('SNR for OFDM (dB)'); 
ylabel('MSE');
legend('Simulation: FBMC Auxiliary','Location','SouthWest');
grid on;