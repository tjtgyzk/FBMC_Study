   EqulizedRate = 10;
   semilogy(M_SNR_dB,mean(BER_FBMC_Aux_InterferenceCancellation(:,:,1)/EqulizedRate,2),'LineWidth',0.7,'Marker','square','LineStyle','-.')
   hold on
   grid on;
   semilogy(M_SNR_dB,mean(BER_FBMC_Aux_InterferenceCancellation(:,:,2)/EqulizedRate,2),'LineWidth',0.7,'Marker','diamond','LineStyle','--');
   semilogy(M_SNR_dB,mean(BER_FBMC_Aux_InterferenceCancellation(:,:,3)/EqulizedRate,2),'LineWidth',0.7,'Marker','x');
   semilogy(M_SNR_dB,mean(BER_FBMC_Aux_InterferenceCancellation(:,:,4)/EqulizedRate,2),'LineWidth',0.7,'Marker','o');
   % 取消以下行的注释以保留坐标区的 X 范围
   % xlim(axes1,[10 40]);
   % 取消以下行的注释以保留坐标区的 Y 范围
   % ylim(axes1,[0.00981127595093666 0.281460978842495]);
   legend({'无迭代','迭代1次','迭代两次','迭代三次'});