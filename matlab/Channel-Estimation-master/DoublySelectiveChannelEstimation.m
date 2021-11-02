% =========================================================================   
% (c) 2018 Ronald Nissel, ronald.nissel@gmail.com
% =========================================================================  
% Allows to reproduce Figure 2, 3, 4 and 5 of "Doubly-Selective Channel 
% Estimation in FBMC-OQAM and OFDM Systems", R. Nissel, et al, IEEE VTC
% Fall, 2018. In particular, this script simulates an FBMC and OFDM
% transmission over a doubly-selective channel, including doubly-selective
% MMSE channel estimation and interference cancellation. Note that, to 
% truly reproduce the figures, the lines 41-46 must be uncommented.


clear; close all;
addpath('./Theory');


%% 参数
% 仿真
M_SNR_dB                  = [10:5:40];              % 信噪比dB
NrRepetitions             = 25;                     % 蒙特卡罗重复次数（不同信道实现）             
ZeroThresholdSparse       = 8;                      % 将一些小于“10^（-ZeroThresholdSparse）”的矩阵值设置为零。
PlotIterationStepsSNRdB   = 35;                     % 绘制SNR为35dB的迭代步骤上的误码率。

% FBMC与OFDM参数
L                         = 12*2;                   % 子载波数，一个资源块由12个子载波组成（时间为0.5ms）
F                         = 15e3;                   % 子载波间隔（Hz）
SamplingRate              = F*12*2;                 % 采样率（采样数/秒）
NrSubframes               = 1;                      % 子帧的数目。F=15kHz时，一个子帧需要1ms。                             
QAM_ModulationOrder       = 4;                      % QAM信号星座顺序，4，16，64，256，1024，。。。

% 信道估计参数
PilotToDataPowerOffset    = 2;                      % OFDM的导频到数据功率偏移。在FBMC数据扩展中，功率偏移是该数字的两倍。 
PilotToDataPowerOffsetAux = 4.685;                  % FBMC的导频到数据功率偏移，辅助方法。
NrIterations              = 4;                      % 干扰消除方案的迭代次数。

% Channel
Velocity_kmh              = 500;                    % 速度单位为km/h。请注意 [mph]*1.6=[kmh] and [m/s]*3.6=[kmh]        
PowerDelayProfile         = 'VehicularA';           % 信道模型，字符串或向量：'Flat', 'AWGN', 'PedestrianA', 'PedestrianB', 'VehicularA', 'VehicularB', 'ExtendedPedestrianA', 'ExtendedPedestrianB', or 'TDL-A_xxns','TDL-B_xxns','TDL-C_xxns' (with xx the RMS delay spread in ns, e.g. 'TDL-A_30ns'), or [1 0 0.2] (Self-defined power delay profile which depends on the sampling rate) 


% ###########################################################################
% % In the paper:
% M_SNR_dB                = [10:2:40];
% PlotIterationStepsSNRdB = 32;
% NrRepetitions           = 1000;
% SamplingRate            = F*14*14;
% NrSubframes             = 2;
% ###########################################################################


%% FBMC对象
FBMC = Modulation.FBMC(...
    L,...                               % 子载波数
    30*NrSubframes,...                  % FBMC符号数
    F,...                               % 子载波间隔（Hz）
    SamplingRate,...                    % 采样率（采样数/秒）
    0,...                               % 中频第一副载波（Hz）
    false,...                           % 传输实值信号
    'Hermite-OQAM',...                  % 原型滤波器（Hermite、PHYDYAS、RRC）和OQAM或QAM， 
    8, ...                              % 重叠因子（还确定频域中的过采样）
    0, ...                              % 初始相移
    true ...                            % 多相实现
    );


%% OFDM Object (Add zeroes to the OFDM signal so that it fits the FBMC signal)
ZeroGuardTimeLength = ((FBMC.Nr.SamplesTotal-(round((1/15e3/14)*SamplingRate)+round(SamplingRate/15e3))*14*NrSubframes)/2)/SamplingRate;
OFDM = Modulation.OFDM(...
    L,...                           % Number Subcarriers
    14*NrSubframes,...              % Number OFDM Symbols
    F,...                           % Subcarrier spacing (Hz)
    SamplingRate,...                % Sampling rate (Samples/s)
    0,...                           % Intermediate frequency first subcarrier (Hz)
    false,...                       % Transmitreal valued signal
    1/15e3/14, ...                  % Cyclic prefix length (s)
    ZeroGuardTimeLength ...         % Zero guard length (s)
    );

%% 检查样本数量
if  OFDM.Nr.SamplesTotal~=FBMC.Nr.SamplesTotal
   error('Total number of samples must be the same for OFDM and FBMC.');
end
N = OFDM.Nr.SamplesTotal;


%% PAM和QAM对象
PAM = Modulation.SignalConstellation(sqrt(QAM_ModulationOrder),'PAM');
QAM = Modulation.SignalConstellation(QAM_ModulationOrder,'QAM');


%% 导频矩阵，0=“数据”，1=导频
PilotMatrix_OFDM = zeros(OFDM.Nr.Subcarriers,14);
PilotMatrix_OFDM(2:2*6:end,2:2*3.5:end)=1;
PilotMatrix_OFDM(5:2*6:end,6:2*3.5:end)=1;
PilotMatrix_OFDM(8:2*6:end,2:2*3.5:end)=1;
PilotMatrix_OFDM(11:2*6:end,6:2*3.5:end)=1;
PilotMatrix_OFDM = repmat(PilotMatrix_OFDM,[1 NrSubframes]);

PilotMatrix_FBMC =  zeros(FBMC.Nr.Subcarriers,30);
PilotMatrix_FBMC(2:12:end,3:16:end) = 1;
PilotMatrix_FBMC(5:12:end,11:16:end) = 1;
PilotMatrix_FBMC(8:12:end,3+1:16:end) = 1;
PilotMatrix_FBMC(11:12:end,11+1:16:end) = 1;
PilotMatrix_FBMC = repmat(PilotMatrix_FBMC,[1 NrSubframes]);


AuxilaryPilotMatrix_FBMC = PilotMatrix_FBMC;
[a,b] = find(PilotMatrix_FBMC);
for i_pilot = 1:length(a)
    AuxilaryPilotMatrix_FBMC(a(i_pilot)+1,b(i_pilot))=-1;
    AuxilaryPilotMatrix_FBMC(a(i_pilot)-1,b(i_pilot))=-1;
    AuxilaryPilotMatrix_FBMC(a(i_pilot),b(i_pilot)+1)=-1;
    AuxilaryPilotMatrix_FBMC(a(i_pilot),b(i_pilot)-1)=-1;
end

%% 消除导频位置对象处的假想干扰（预编码矩阵）
AuxiliaryMethod = ChannelEstimation.ImaginaryInterferenceCancellationAtPilotPosition(...
    'Auxiliary', ...                                    % 干扰消除方式
    AuxilaryPilotMatrix_FBMC, ...                       % 辅助导频矩阵
    FBMC.GetFBMCMatrix, ...                             % 虚干扰矩阵
    28, ...                                             % 消除最近28个干扰源
    PilotToDataPowerOffsetAux ...                       % 导频到数据功率偏移
    );
CodingMethod = ChannelEstimation.ImaginaryInterferenceCancellationAtPilotPosition(...
    'Coding', ...                                       % 干扰消除方式
    PilotMatrix_FBMC, ...                               % 导频矩阵
    FBMC.GetFBMCMatrix, ...                             % 虚干扰矩阵
    20, ...                                             % 消除最近28个干扰源
    2*PilotToDataPowerOffset ...                        % 导频到数据功率偏移
    );

NrPilotSymbols = sum(PilotMatrix_OFDM(:)==1);
NrDataSymbols_OFDM = sum(PilotMatrix_OFDM(:)==0);

PilotMapping_OFDM = zeros(numel(PilotMatrix_OFDM));
PilotMapping_OFDM(PilotMatrix_OFDM(:)==1,1:NrPilotSymbols) = sqrt(PilotToDataPowerOffset)*eye(NrPilotSymbols);
PilotMapping_OFDM(PilotMatrix_OFDM(:)==0,NrPilotSymbols+1:end) = eye(NrDataSymbols_OFDM);
PilotMapping_OFDM = PilotMapping_OFDM/sqrt(mean(diag(PilotMapping_OFDM*PilotMapping_OFDM')));
DataPowerReduction_OFDM = numel(PilotMatrix_OFDM)/(NrPilotSymbols*PilotToDataPowerOffset+NrDataSymbols_OFDM);

Kappa_Aux  = AuxiliaryMethod.PilotToDataPowerOffset*AuxiliaryMethod.DataPowerReduction;
Kappa_Cod  = CodingMethod.PilotToDataPowerOffset*CodingMethod.DataPowerReduction;
Kappa_OFDM = PilotToDataPowerOffset*DataPowerReduction_OFDM;

%% 仅评估中心附近的误码率。例如，忽略边缘 
ConsideredTimeFrequencyPositions_FBMC = zeros(size(PilotMatrix_FBMC));
ConsideredTimeFrequencyPositions_FBMC(5:end-4,11:end-10)= 1;

ConsideredTimeFrequencyPositions_OFDM = zeros(size(PilotMatrix_OFDM));
ConsideredTimeFrequencyPositions_OFDM(5:end-4,6:end-5)= 1;

for i_lk = 1: AuxiliaryMethod.NrDataSymbols
    xD_Temp = zeros(AuxiliaryMethod.NrDataSymbols,1);
    xD_Temp(i_lk) = 1;
    x_FBMC_Aux = AuxiliaryMethod.PrecodingMatrix*[zeros(NrPilotSymbols,1);xD_Temp];    
    ConsideredDataPositions_FBMC_Aux(i_lk) = sum(abs(x_FBMC_Aux(1==ConsideredTimeFrequencyPositions_FBMC.*not(AuxilaryPilotMatrix_FBMC))))>AuxiliaryMethod.DataPowerReduction*0.9;
end
for i_lk = 1: CodingMethod.NrDataSymbols
    xD_Temp = zeros(CodingMethod.NrDataSymbols,1);
    xD_Temp(i_lk) = 1;
    x_FBMC_Cod = CodingMethod.PrecodingMatrix*[zeros(NrPilotSymbols,1);xD_Temp];      
    ConsideredDataPositions_FBMC_Cod(i_lk) =not(any(x_FBMC_Cod(not(ConsideredTimeFrequencyPositions_FBMC))));
end
for i_lk = 1: NrDataSymbols_OFDM
    xD_Temp = zeros(NrDataSymbols_OFDM,1);
    xD_Temp(i_lk) = 1;
    x_OFDM = PilotMapping_OFDM*[zeros(NrPilotSymbols,1);xD_Temp];
    ConsideredDataPositions_OFDM(i_lk) = sum(abs(x_OFDM(1==ConsideredTimeFrequencyPositions_OFDM.*not(PilotMatrix_OFDM))))>DataPowerReduction_OFDM*0.9;
end

ConsideredBits_FBMC_Aux = reshape(repmat(ConsideredDataPositions_FBMC_Aux,log2(QAM_ModulationOrder)/2,1),[],1);
ConsideredBits_FBMC_Cod = reshape(repmat(ConsideredDataPositions_FBMC_Cod,log2(QAM_ModulationOrder)/2,1),[],1);
ConsideredBits_OFDM     = reshape(repmat(ConsideredDataPositions_OFDM,log2(QAM_ModulationOrder),1),[],1);


%% 信道模型对象
ChannelModel = Channel.FastFading(...
    SamplingRate,...                                   % 采样率（采样数/秒）
    PowerDelayProfile,...                              % 功率延迟配置文件，字符串或向量: 'Flat', 'AWGN', 'PedestrianA', 'PedestrianB', 'VehicularA', 'VehicularB', 'ExtendedPedestrianA', 'ExtendedPedestrianB', or 'TDL-A_xxns','TDL-B_xxns','TDL-C_xxns' (with xx the RMS delay spread in ns, e.g. 'TDL-A_30ns'), or [1 0 0.2] (取决于采样率的自定义功率延迟曲线)
    N,...                                              % 总样本数
    Velocity_kmh/3.6*2.5e9/2.998e8,...                 % 最大多普勒频移: Velocity_kmh/3.6*CarrierFrequency/2.998e8
    'Jakes',...                                        % 多普勒模型: 'Jakes', 'Uniform', 'Discrete-Jakes', 'Discrete-Uniform'. For "Discrete-", 我们假设一个离散的多普勒频谱来提高仿真时间。只有在样本数量和速度足够大的情况下，这种方法才能准确工作
    200,...                                            % WSSUS进程的路径数. 只和 'Jakes' 和 'Uniform' 多普勒频谱相关
    1,...                                              % 发射天线的数量
    1,...                                              % 接收天线的数量
    1 ...                                              % 如果通道的预定义延迟抽头不符合采样率，则发出警告。如果它们大致相同，这通常不是什么大问题。
    );
R_vecH = ChannelModel.GetCorrelationMatrix;


%% 预先计算发送和接收矩阵
G_FBMC = FBMC.GetTXMatrix;
Q_FBMC = (FBMC.GetRXMatrix)';

G_OFDM = OFDM.GetTXMatrix;
Q_OFDM = (OFDM.GetRXMatrix)';

GP_FBMC = G_FBMC(:,PilotMatrix_FBMC(:)==1);
GP_OFDM = G_OFDM(:,PilotMatrix_OFDM(:)==1);

QP_FBMC = Q_FBMC(:,PilotMatrix_FBMC(:)==1);
QP_OFDM = Q_OFDM(:,PilotMatrix_OFDM(:)==1);

G_Aux = G_FBMC*AuxiliaryMethod.PrecodingMatrix;
G_Cod = G_FBMC*CodingMethod.PrecodingMatrix;
G_OFDM_PilotMapping = G_OFDM*PilotMapping_OFDM;


%% 计算相关矩阵
disp('计算导频估计的相关矩阵 (无噪音，无干扰)...');
R_hP_FBMC  = nan(NrPilotSymbols,NrPilotSymbols);
R_hP_OFDM = nan(NrPilotSymbols,NrPilotSymbols);
for j_pilot = 1:NrPilotSymbols
    R_hP_FBMC(:,j_pilot) = sum((QP_FBMC'*reshape(R_vecH*kron(GP_FBMC(:,j_pilot).',QP_FBMC(:,j_pilot)')',N,N)).*(GP_FBMC.'),2);    
    R_hP_OFDM(:,j_pilot) = sum((QP_OFDM'*reshape(R_vecH*kron(GP_OFDM(:,j_pilot).',QP_OFDM(:,j_pilot)')',N,N)).*(GP_OFDM.'),2);
end


disp('计算导频估计的相关矩阵(无干扰)...');
R_hP_est_noNoise_FBMC_Aux = R_hP_FBMC;
R_hP_est_noNoise_FBMC_Cod = R_hP_FBMC;
R_hP_est_noNoise_OFDM = R_hP_OFDM;
for i_pilots = 1: NrPilotSymbols
    % FBMC辅助方法，类似于方程式（13），但计算效率更高
    Temp = kron(sparse(eye(N)),QP_FBMC(:,i_pilots)')/sqrt(Kappa_Aux);
    R_hP_est_noNoise_FBMC_Aux(i_pilots,i_pilots)=abs(sum(sum((G_Aux.'*(Temp*R_vecH*Temp')).*G_Aux',2)));     

    % FBMC编码
    Temp = kron(sparse(eye(N)),QP_FBMC(:,i_pilots)')/sqrt(Kappa_Cod);
    R_hP_est_noNoise_FBMC_Cod(i_pilots,i_pilots)=abs(sum(sum((G_Cod.'*(Temp*R_vecH*Temp')).*G_Cod',2)));     

    % OFDM
    Temp = kron(sparse(eye(N)),QP_OFDM(:,i_pilots)')/sqrt(Kappa_OFDM);
    R_hP_est_noNoise_OFDM(i_pilots,i_pilots)=abs(sum(sum((G_OFDM_PilotMapping.'*(Temp*R_vecH*Temp')).*G_OFDM_PilotMapping',2)));           
end 


disp('计算导频估计的相关矩阵...');
R_hP_est_FBMC_Aux = repmat(R_hP_est_noNoise_FBMC_Aux,[1 1 length(M_SNR_dB)]);
R_hP_est_FBMC_Cod = repmat(R_hP_est_noNoise_FBMC_Cod,[1 1 length(M_SNR_dB)]);
R_hP_est_OFDM = repmat(R_hP_est_noNoise_OFDM,[1 1 length(M_SNR_dB)]);
for i_SNR = 1:length(M_SNR_dB)
    SNR_dB = M_SNR_dB(i_SNR);
    Pn_time = SamplingRate/(F*L)*10^(-SNR_dB/10);   
    for i_pilots = 1: NrPilotSymbols
        R_hP_est_FBMC_Aux(i_pilots,i_pilots,i_SNR)=R_hP_est_noNoise_FBMC_Aux(i_pilots,i_pilots)+Pn_time*QP_FBMC(:,i_pilots)'*QP_FBMC(:,i_pilots)/(Kappa_Aux);     
        R_hP_est_FBMC_Cod(i_pilots,i_pilots,i_SNR)=R_hP_est_noNoise_FBMC_Cod(i_pilots,i_pilots)+Pn_time*QP_FBMC(:,i_pilots)'*QP_FBMC(:,i_pilots)/(Kappa_Cod);     
        R_hP_est_OFDM(i_pilots,i_pilots,i_SNR)=R_hP_est_noNoise_OFDM(i_pilots,i_pilots)+Pn_time*QP_OFDM(:,i_pilots)'*QP_OFDM(:,i_pilots)/Kappa_OFDM;           
    end    
end

R_hP_est_noInterference_FBMC_Aux = R_hP_est_FBMC_Aux-repmat((R_hP_est_noNoise_FBMC_Aux-R_hP_FBMC),[1 1 length(M_SNR_dB)]);
R_hP_est_noInterference_FBMC_Cod = R_hP_est_FBMC_Cod-repmat((R_hP_est_noNoise_FBMC_Cod-R_hP_FBMC),[1 1 length(M_SNR_dB)]);
R_hP_est_noInterference_OFDM = R_hP_est_OFDM-repmat((R_hP_est_noNoise_OFDM-R_hP_OFDM),[1 1 length(M_SNR_dB)]);


disp('计算传输矩阵D和导频估计之间的相关矩阵...');
R_Dij_hP_FBMC = sparse(size(G_FBMC,2)^2,NrPilotSymbols);
R_Dij_hP_OFDM = sparse(size(G_OFDM,2)^2,NrPilotSymbols);
for i_pilot = 1: NrPilotSymbols
    R_Dij_hP_FBMC_Temp = reshape(Q_FBMC'*reshape(R_vecH*kron(GP_FBMC(:,i_pilot).',QP_FBMC(:,i_pilot)')',N,N)*G_FBMC,[],1);
    R_Dij_hP_OFDM_Temp = reshape(Q_OFDM'*reshape(R_vecH*kron(GP_OFDM(:,i_pilot).',QP_OFDM(:,i_pilot)')',N,N)*G_OFDM,[],1);
    
    R_Dij_hP_FBMC_Temp(abs(R_Dij_hP_FBMC_Temp)<10^(-ZeroThresholdSparse))=0;
    R_Dij_hP_OFDM_Temp(abs(R_Dij_hP_OFDM_Temp)<10^(-ZeroThresholdSparse))=0;
     
    R_Dij_hP_FBMC(:,i_pilot) = R_Dij_hP_FBMC_Temp;
    R_Dij_hP_OFDM(:,i_pilot) = R_Dij_hP_OFDM_Temp;
end
clear R_vecH;

%% 计算导频位置的SIR
SIR_P_FBMC_Aux_dB = 10*log10(trace(abs(R_hP_FBMC))./trace(abs(R_hP_est_noNoise_FBMC_Aux-R_hP_FBMC)));
SIR_P_FBMC_Cod_dB = 10*log10(trace(abs(R_hP_FBMC))./trace(abs(R_hP_est_noNoise_FBMC_Cod-R_hP_FBMC)));
SIR_P_OFDM_dB     = 10*log10(trace(abs(R_hP_OFDM))./trace(abs(R_hP_est_noNoise_OFDM-R_hP_OFDM)));


%% 利用相关系数计算MMSE估计矩阵
disp('计算MMSE解 ...');
W_MMSE_FBMC_Aux = sparse(size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,length(M_SNR_dB));
W_MMSE_FBMC_Cod = sparse(size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,length(M_SNR_dB));
W_MMSE_OFDM     = sparse(size(G_OFDM,2)*size(G_OFDM,2)*NrPilotSymbols,length(M_SNR_dB));
for i_SNR = 1:length(M_SNR_dB)
    W_MMSE_FBMC_Aux_Temp = reshape(R_Dij_hP_FBMC*pinv(R_hP_est_FBMC_Aux(:,:,i_SNR)),size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,1);
    W_MMSE_FBMC_Cod_Temp = reshape(R_Dij_hP_FBMC*pinv(R_hP_est_FBMC_Cod(:,:,i_SNR)),size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,1);
    W_MMSE_OFDM_Temp     = reshape(R_Dij_hP_OFDM*pinv(R_hP_est_OFDM(:,:,i_SNR)),size(G_OFDM,2)*size(G_OFDM,2)*NrPilotSymbols,1);
 
    W_MMSE_FBMC_Aux_Temp(abs(W_MMSE_FBMC_Aux_Temp)<10^(-ZeroThresholdSparse))=0;
    W_MMSE_FBMC_Cod_Temp(abs(W_MMSE_FBMC_Cod_Temp)<10^(-ZeroThresholdSparse))=0;
    W_MMSE_OFDM_Temp(abs(W_MMSE_OFDM_Temp)<10^(-ZeroThresholdSparse))=0;
    
    W_MMSE_FBMC_Aux(:,i_SNR) = W_MMSE_FBMC_Aux_Temp;
    W_MMSE_FBMC_Cod(:,i_SNR) = W_MMSE_FBMC_Cod_Temp;
    W_MMSE_OFDM(:,i_SNR)     = W_MMSE_OFDM_Temp;
end

%% 计算导频估计无干扰的MMSE估计矩阵
disp('计算MMSE解决方案（无干扰）...');
W_MMSE_noInterference_FBMC_Aux = sparse(size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,length(M_SNR_dB));
W_MMSE_noInterference_FBMC_Cod = sparse(size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,length(M_SNR_dB));
W_MMSE_noInterference_OFDM     = sparse(size(G_OFDM,2)*size(G_OFDM,2)*NrPilotSymbols,length(M_SNR_dB));
for i_SNR = 1:length(M_SNR_dB)
    W_MMSE_noInterference_FBMC_Aux_Temp = reshape(R_Dij_hP_FBMC*pinv(R_hP_est_noInterference_FBMC_Aux(:,:,i_SNR)),size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,1);
    W_MMSE_noInterference_FBMC_Cod_Temp = reshape(R_Dij_hP_FBMC*pinv(R_hP_est_noInterference_FBMC_Cod(:,:,i_SNR)),size(G_FBMC,2)*size(G_FBMC,2)*NrPilotSymbols,1);
    W_MMSE_noInterference_OFDM_Temp = reshape(R_Dij_hP_OFDM*pinv(R_hP_est_noInterference_OFDM(:,:,i_SNR)),size(G_OFDM,2)*size(G_OFDM,2)*NrPilotSymbols,1);

    W_MMSE_noInterference_FBMC_Aux_Temp(abs(W_MMSE_noInterference_FBMC_Aux_Temp)<10^(-ZeroThresholdSparse))=0;
    W_MMSE_noInterference_FBMC_Cod_Temp(abs(W_MMSE_noInterference_FBMC_Cod_Temp)<10^(-ZeroThresholdSparse))=0;
    W_MMSE_noInterference_OFDM_Temp(abs(W_MMSE_noInterference_OFDM_Temp)<10^(-ZeroThresholdSparse))=0;
    
    W_MMSE_noInterference_FBMC_Aux(:,i_SNR) = W_MMSE_noInterference_FBMC_Aux_Temp;
    W_MMSE_noInterference_FBMC_Cod(:,i_SNR) = W_MMSE_noInterference_FBMC_Cod_Temp;
    W_MMSE_noInterference_OFDM(:,i_SNR)     = W_MMSE_noInterference_OFDM_Temp;    
end


%% 双平坦瑞利信道的理论BEP
M_SNR_dB_morePoints = min(M_SNR_dB):1:max(M_SNR_dB);
BitErrorProbability = BitErrorProbabilityDoublyFlatRayleigh(M_SNR_dB_morePoints,QAM.SymbolMapping,QAM.BitMapping);


%% 为Parfor预先分配
BER_FBMC_Aux_PerfectCSI_InterferenceCancellation        = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Aux_PerfectCSI_InterferenceCancellation_NoEdge = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Cod_PerfectCSI_InterferenceCancellation        = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Cod_PerfectCSI_InterferenceCancellation_NoEdge = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_OFDM_PerfectCSI_InterferenceCancellation            = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_OFDM_PerfectCSI_InterferenceCancellation_NoEdge     = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Aux_OneTapEqualizer_PerfectCSI                 = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Aux_OneTapEqualizer_PerfectCSI_NoEdge          = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Cod_OneTapEqualizer_PerfectCSI                 = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Cod_OneTapEqualizer_PerfectCSI_NoEdge          = nan(length(M_SNR_dB),NrRepetitions);
BER_OFDM_OneTapEqualizer_PerfectCSI                     = nan(length(M_SNR_dB),NrRepetitions);
BER_OFDM_OneTapEqualizer_PerfectCSI_NoEdge              = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Aux_InterferenceCancellation                   = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Cod_InterferenceCancellation                   = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_OFDM_InterferenceCancellation                       = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Aux_InterferenceCancellation_NoEdge            = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Cod_InterferenceCancellation_NoEdge            = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_OFDM_InterferenceCancellation_NoEdge                = nan(length(M_SNR_dB),NrRepetitions,NrIterations);
BER_FBMC_Aux_OneTapEqualizer                            = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Aux_OneTapEqualizer_NoEdge                     = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Cod_OneTapEqualizer                            = nan(length(M_SNR_dB),NrRepetitions);
BER_FBMC_Cod_OneTapEqualizer_NoEdge                     = nan(length(M_SNR_dB),NrRepetitions);
BER_OFDM_OneTapEqualizer                                = nan(length(M_SNR_dB),NrRepetitions);
BER_OFDM_OneTapEqualizer_NoEdge                         = nan(length(M_SNR_dB),NrRepetitions);

%% 开始模拟
tic
disp('Monte Carlo 仿真 ...');
for i_rep = 1:NrRepetitions
    %% 更新信道
    ChannelModel.NewRealization;
    
    %% 二进制数据
    BinaryDataStream_FBMC_Aux = randi([0 1],AuxiliaryMethod.NrDataSymbols*log2(PAM.ModulationOrder),1);
    BinaryDataStream_FBMC_Cod = randi([0 1],CodingMethod.NrDataSymbols*log2(PAM.ModulationOrder),1);    
    BinaryDataStream_OFDM     = randi([0 1],NrDataSymbols_OFDM*log2(QAM.ModulationOrder),1);
       
    %% 数据符号
    xD_FBMC_Aux = PAM.Bit2Symbol(BinaryDataStream_FBMC_Aux);
    xD_FBMC_Cod = PAM.Bit2Symbol(BinaryDataStream_FBMC_Cod);
    xD_OFDM     = QAM.Bit2Symbol(BinaryDataStream_OFDM);
       
    %% 导频符号
    xP_FBMC = PAM.SymbolMapping(randi(PAM.ModulationOrder,AuxiliaryMethod.NrPilotSymbols,1));
    xP_FBMC = xP_FBMC./abs(xP_FBMC);
    xP_OFDM = QAM.SymbolMapping(randi(QAM.ModulationOrder,NrPilotSymbols,1));
    xP_OFDM = xP_OFDM./abs(xP_OFDM);
    
    %% 传输的数据符号 (Map bin to symbol)
    x_FBMC_Aux = AuxiliaryMethod.PrecodingMatrix*[xP_FBMC;xD_FBMC_Aux];
    x_FBMC_Cod = CodingMethod.PrecodingMatrix*[xP_FBMC;xD_FBMC_Cod];
    x_OFDM     = PilotMapping_OFDM*[xP_OFDM;xD_OFDM];
                
    %% 发射信号 (时域)
    s_FBMC_Aux = G_FBMC*x_FBMC_Aux(:); % Same as "FBMC.Modulation(x_FBMC_Aux)" which is computationally more efficient. But G_FBMC is consistent with the paper.
    s_FBMC_Cod = G_FBMC*x_FBMC_Cod(:); 
    s_OFDM     = G_OFDM*x_OFDM(:);
       
    %% 信道
    ConvolutionMatrix = ChannelModel.GetConvolutionMatrix{1};
   
    r_FBMC_Aux_noNoise = ConvolutionMatrix*s_FBMC_Aux;
    r_FBMC_Cod_noNoise = ConvolutionMatrix*s_FBMC_Cod;
    r_OFDM_noNoise     = ConvolutionMatrix*s_OFDM;
    
    %% 传输矩阵
    D_FBMC = Q_FBMC'*ConvolutionMatrix*G_FBMC;
    D_OFDM = Q_OFDM'*ConvolutionMatrix*G_OFDM;
        
    %% 单抽头信道（已知完美信道信息）
    h_FBMC = diag(D_FBMC);
    h_OFDM = diag(D_OFDM);
           
    for i_SNR = 1:length(M_SNR_dB)
        %% 增加噪声
        SNR_dB  = M_SNR_dB(i_SNR);
        Pn_time = SamplingRate/(F*L)*10^(-SNR_dB/10);
        noise   = sqrt(Pn_time/2)*(randn(size(s_OFDM))+1j*randn(size(s_OFDM)));

        r_FBMC_Aux = r_FBMC_Aux_noNoise+noise; 
        r_FBMC_Cod = r_FBMC_Cod_noNoise+noise;
        r_OFDM     = r_OFDM_noNoise+noise;

        %% 解调FBMC信号
        y_FBMC_Aux         = Q_FBMC'*r_FBMC_Aux; % Same as "FBMC.Demodulation(r_FBMC_Aux)" 
        y_FBMC_Cod         = Q_FBMC'*r_FBMC_Cod; % Same as "FBMC.Demodulation(r_FBMC_Cod)" 
        y_FBMC_Cod_PostCod = CodingMethod.PrecodingMatrix'*y_FBMC_Cod;
        y_OFDM             = Q_OFDM'*r_OFDM; % Same as "OFDM.Demodulation(r_OFDM)" 

        %% 导频位置的信道估计
        hP_est_FBMC_Aux = y_FBMC_Aux(PilotMatrix_FBMC==1)./xP_FBMC/sqrt(Kappa_Aux);
        hP_est_FBMC_Cod = y_FBMC_Cod(PilotMatrix_FBMC==1)./xP_FBMC/sqrt(Kappa_Cod);
        hP_est_OFDM     = y_OFDM(PilotMatrix_OFDM==1)./xP_OFDM/sqrt(Kappa_OFDM);

        %% 估计传输矩阵
        D_FBMC_est_Aux = sum(bsxfun(@times,...
            reshape(full(W_MMSE_FBMC_Aux(:,i_SNR)),size(G_FBMC,2),size(G_FBMC,2),NrPilotSymbols),...
            reshape(hP_est_FBMC_Aux,1,1,[])),3);
        D_FBMC_est_Cod = sum(bsxfun(@times,...
            reshape(full(W_MMSE_FBMC_Cod(:,i_SNR)),size(G_FBMC,2),size(G_FBMC,2),NrPilotSymbols),...
            reshape(hP_est_FBMC_Cod,1,1,[])),3);     
        D_OFDM_est = sum(bsxfun(@times,...
            reshape(full(W_MMSE_OFDM(:,i_SNR)),size(G_OFDM,2),size(G_OFDM,2),NrPilotSymbols),...
            reshape(hP_est_OFDM,1,1,[])),3);     

        %% 单抽头均衡器
        h_est_FBMC_Aux = diag(D_FBMC_est_Aux);  
        x_est_OneTapEqualizer_FBMC_Aux = y_FBMC_Aux./h_est_FBMC_Aux;
        xD_est_OneTapEqualizer_FBMC_Aux = real(x_est_OneTapEqualizer_FBMC_Aux(AuxilaryPilotMatrix_FBMC(:)==0)./sqrt(AuxiliaryMethod.DataPowerReduction));
        DetectedBitStream_OneTapEqualizer_FBMC_Aux = PAM.Symbol2Bit(xD_est_OneTapEqualizer_FBMC_Aux);   
        BER_FBMC_Aux_OneTapEqualizer(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux~=DetectedBitStream_OneTapEqualizer_FBMC_Aux);    
        BER_FBMC_Aux_OneTapEqualizer_NoEdge(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux(ConsideredBits_FBMC_Aux)~=DetectedBitStream_OneTapEqualizer_FBMC_Aux(ConsideredBits_FBMC_Aux));

        h_est_FBMC_Cod = diag(D_FBMC_est_Cod); 
        x_est_OneTapEqualizer_FBMC_Cod = CodingMethod.PrecodingMatrix'*(y_FBMC_Cod./h_est_FBMC_Cod);
        xD_est_OneTapEqualizer_FBMC_Cod = real(x_est_OneTapEqualizer_FBMC_Cod(NrPilotSymbols+1:end))/CodingMethod.DataPowerReduction;
        DetectedBitStream_OneTapEqualizer_FBMC_Cod = PAM.Symbol2Bit(xD_est_OneTapEqualizer_FBMC_Cod);   
        BER_FBMC_Cod_OneTapEqualizer(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Cod~=DetectedBitStream_OneTapEqualizer_FBMC_Cod);    
        BER_FBMC_Cod_OneTapEqualizer_NoEdge(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Cod(ConsideredBits_FBMC_Cod)~=DetectedBitStream_OneTapEqualizer_FBMC_Cod(ConsideredBits_FBMC_Cod));

        h_est_OFDM = diag(D_OFDM_est);
        x_est_OneTapEqualizer_OFDM = y_OFDM./h_est_OFDM;
        xD_est_OneTapEqualizer_OFDM = x_est_OneTapEqualizer_OFDM(PilotMatrix_OFDM(:)==0)./sqrt(DataPowerReduction_OFDM);   
        DetectedBitStream_OneTapEqualizer_OFDM = QAM.Symbol2Bit(xD_est_OneTapEqualizer_OFDM);   
        BER_OFDM_OneTapEqualizer(i_SNR,i_rep) = mean(BinaryDataStream_OFDM~=DetectedBitStream_OneTapEqualizer_OFDM);
        BER_OFDM_OneTapEqualizer_NoEdge(i_SNR,i_rep) = mean(BinaryDataStream_OFDM(ConsideredBits_OFDM)~=DetectedBitStream_OneTapEqualizer_OFDM(ConsideredBits_OFDM));

        %% 单抽头均衡器, 完美已知信道信息
        x_est_OneTapEqualizer_FBMC_Aux_PerfectCSI = y_FBMC_Aux./h_FBMC;
        xD_est_OneTapEqualizer_FBMC_Aux_PerfectCSI = real(x_est_OneTapEqualizer_FBMC_Aux_PerfectCSI(AuxilaryPilotMatrix_FBMC(:)==0)./sqrt(AuxiliaryMethod.DataPowerReduction));
        DetectedBitStream_OneTapEqualizer_FBMC_Aux_PerfectCSI = PAM.Symbol2Bit(xD_est_OneTapEqualizer_FBMC_Aux_PerfectCSI);   
        BER_FBMC_Aux_OneTapEqualizer_PerfectCSI(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux~=DetectedBitStream_OneTapEqualizer_FBMC_Aux_PerfectCSI);    
        BER_FBMC_Aux_OneTapEqualizer_PerfectCSI_NoEdge(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux(ConsideredBits_FBMC_Aux)~=DetectedBitStream_OneTapEqualizer_FBMC_Aux_PerfectCSI(ConsideredBits_FBMC_Aux));    

        x_est_OneTapEqualizer_FBMC_Cod_PerfectCSI = CodingMethod.PrecodingMatrix'*(y_FBMC_Cod./h_FBMC);
        xD_est_OneTapEqualizer_FBMC_Cod_PerfectCSI = real(x_est_OneTapEqualizer_FBMC_Cod_PerfectCSI(NrPilotSymbols+1:end))/CodingMethod.DataPowerReduction;
        DetectedBitStream_OneTapEqualizer_FBMC_Cod_PerfectCSI = PAM.Symbol2Bit(xD_est_OneTapEqualizer_FBMC_Cod_PerfectCSI);   
        BER_FBMC_Cod_OneTapEqualizer_PerfectCSI(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Cod~=DetectedBitStream_OneTapEqualizer_FBMC_Cod_PerfectCSI);    
        BER_FBMC_Cod_OneTapEqualizer_PerfectCSI_NoEdge(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Cod(ConsideredBits_FBMC_Cod)~=DetectedBitStream_OneTapEqualizer_FBMC_Cod_PerfectCSI(ConsideredBits_FBMC_Cod));    

        x_est_OneTapEqualizer_OFDM_PerfectCSI = y_OFDM./h_OFDM;
        xD_est_OneTapEqualizer_OFDM_PerfectCSI = x_est_OneTapEqualizer_OFDM_PerfectCSI(PilotMatrix_OFDM(:)==0)./sqrt(DataPowerReduction_OFDM);   
        DetectedBitStream_OneTapEqualizer_OFDM_PerfectCSI = QAM.Symbol2Bit(xD_est_OneTapEqualizer_OFDM_PerfectCSI);   
        BER_OFDM_OneTapEqualizer_PerfectCSI(i_SNR,i_rep) = mean(BinaryDataStream_OFDM~=DetectedBitStream_OneTapEqualizer_OFDM_PerfectCSI);
        BER_OFDM_OneTapEqualizer_PerfectCSI_NoEdge(i_SNR,i_rep) = mean(BinaryDataStream_OFDM(ConsideredBits_OFDM)~=DetectedBitStream_OneTapEqualizer_OFDM_PerfectCSI(ConsideredBits_OFDM));

        %% 改进的信道估计和数据检测
        xD_est_FBMC_Aux_Temp = xD_est_OneTapEqualizer_FBMC_Aux; % 单抽头估计值初始化   
        xD_est_FBMC_Cod_Temp = xD_est_OneTapEqualizer_FBMC_Cod; % 单抽头估计值初始化     
        xD_est_OFDM_Temp     = xD_est_OneTapEqualizer_OFDM; % 单抽头估计值初始化 
        xD_est_FBMC_Aux_PerfectCSI_Temp = xD_est_OneTapEqualizer_FBMC_Aux_PerfectCSI; % 单抽头估计值初始化     
        xD_est_FBMC_Cod_PerfectCSI_Temp = xD_est_OneTapEqualizer_FBMC_Cod_PerfectCSI; % 单抽头估计值初始化     
        xD_est_OFDM_PerfectCSI_Temp     = xD_est_OneTapEqualizer_OFDM_PerfectCSI; % initialize with one tap estimates     
        D_FBMC_est_Aux_Temp  = D_FBMC_est_Aux;
        D_FBMC_est_Cod_Temp  = D_FBMC_est_Cod;
        D_OFDM_est_Temp      = D_OFDM_est;
        h_est_FBMC_Aux_Temp  = h_est_FBMC_Aux;
        h_est_FBMC_Cod_Temp  = h_est_FBMC_Cod;
        h_est_OFDM_Temp      = h_est_OFDM;
        for i_iteration = 1:NrIterations
            y_FBMC_Aux_InterferenceCancellation = (y_FBMC_Aux(:) - (D_FBMC_est_Aux_Temp-diag(h_est_FBMC_Aux_Temp))*AuxiliaryMethod.PrecodingMatrix*[xP_FBMC;PAM.SymbolQuantization(xD_est_FBMC_Aux_Temp)]);                
            y_FBMC_Cod_InterferenceCancellation = (y_FBMC_Cod - (D_FBMC_est_Cod_Temp-diag(h_est_FBMC_Cod_Temp))*CodingMethod.PrecodingMatrix*[xP_FBMC;PAM.SymbolQuantization(xD_est_FBMC_Cod_Temp)]);                   
            y_OFDM_InterferenceCancellation     = (y_OFDM - (D_OFDM_est_Temp-diag(h_est_OFDM_Temp))*PilotMapping_OFDM*[xP_OFDM;QAM.SymbolQuantization(xD_est_OFDM_Temp)]);        

            % 导频位置的新信道估计            
            hP_est_FBMC_Aux_Temp = y_FBMC_Aux_InterferenceCancellation(PilotMatrix_FBMC==1)./xP_FBMC/sqrt(Kappa_Aux);
            hP_est_FBMC_Cod_Temp = y_FBMC_Cod_InterferenceCancellation(PilotMatrix_FBMC==1)./xP_FBMC/sqrt(Kappa_Cod);
            hP_est_OFDM_Temp     = y_OFDM_InterferenceCancellation(PilotMatrix_OFDM==1)./xP_OFDM/sqrt(Kappa_OFDM);

            % 改进的信道估计（迭代）
            if i_iteration<=NrIterations/2
                D_FBMC_est_Aux_Temp = sum(bsxfun(@times,...
                    reshape(full(W_MMSE_FBMC_Aux(:,i_SNR)),size(G_FBMC,2),size(G_FBMC,2),NrPilotSymbols),...
                    reshape(hP_est_FBMC_Aux_Temp,1,1,[])),3);
                D_FBMC_est_Cod_Temp = sum(bsxfun(@times,...
                    reshape(full(W_MMSE_FBMC_Cod(:,i_SNR)),size(G_FBMC,2),size(G_FBMC,2),NrPilotSymbols),...
                    reshape(hP_est_FBMC_Cod_Temp,1,1,[])),3);     
                D_OFDM_est_Temp = sum(bsxfun(@times,...
                    reshape(full(W_MMSE_OFDM(:,i_SNR)),size(G_OFDM,2),size(G_OFDM,2),NrPilotSymbols),...
                    reshape(hP_est_OFDM_Temp,1,1,[])),3);          
            else
                D_FBMC_est_Aux_Temp = sum(bsxfun(@times,...
                    reshape(full(W_MMSE_noInterference_FBMC_Aux(:,i_SNR)),size(G_FBMC,2),size(G_FBMC,2),NrPilotSymbols),...
                    reshape(hP_est_FBMC_Aux_Temp,1,1,[])),3);
                D_FBMC_est_Cod_Temp = sum(bsxfun(@times,...
                    reshape(full(W_MMSE_noInterference_FBMC_Cod(:,i_SNR)),size(G_FBMC,2),size(G_FBMC,2),NrPilotSymbols),...
                    reshape(hP_est_FBMC_Cod_Temp,1,1,[])),3);     
                D_OFDM_est_Temp = sum(bsxfun(@times,...
                    reshape(full(W_MMSE_noInterference_OFDM(:,i_SNR)),size(G_OFDM,2),size(G_OFDM,2),NrPilotSymbols),...
                    reshape(hP_est_OFDM_Temp,1,1,[])),3);       
            end

            % 单抽头信道
            h_est_FBMC_Aux_Temp = diag(D_FBMC_est_Aux_Temp);
            h_est_FBMC_Cod_Temp = diag(D_FBMC_est_Cod_Temp);
            h_est_OFDM_Temp     = diag(D_OFDM_est_Temp);

            x_est_FBMC_Aux_Temp = y_FBMC_Aux_InterferenceCancellation(:)./h_est_FBMC_Aux_Temp;
            x_est_FBMC_Cod_Temp = CodingMethod.PrecodingMatrix'*(y_FBMC_Cod_InterferenceCancellation./h_est_FBMC_Cod_Temp);
            x_est_OFDM_Temp     =  y_OFDM_InterferenceCancellation./h_est_OFDM_Temp;

            xD_est_FBMC_Aux_Temp = real(x_est_FBMC_Aux_Temp(AuxilaryPilotMatrix_FBMC(:)==0)./sqrt(AuxiliaryMethod.DataPowerReduction));
            xD_est_FBMC_Cod_Temp = real(x_est_FBMC_Cod_Temp(NrPilotSymbols+1:end))/CodingMethod.DataPowerReduction;
            xD_est_OFDM_Temp     = x_est_OFDM_Temp(PilotMatrix_OFDM(:)==0)./sqrt(DataPowerReduction_OFDM); 

            DetectedBitStream_FBMC_Aux_Temp = PAM.Symbol2Bit(xD_est_FBMC_Aux_Temp); 
            DetectedBitStream_FBMC_Cod_Temp = PAM.Symbol2Bit(xD_est_FBMC_Cod_Temp); 
            DetectedBitStream_OFDM_Temp     = QAM.Symbol2Bit(xD_est_OFDM_Temp);

            BER_FBMC_Aux_InterferenceCancellation(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Aux~=DetectedBitStream_FBMC_Aux_Temp);
            BER_FBMC_Cod_InterferenceCancellation(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Cod~=DetectedBitStream_FBMC_Cod_Temp);   
            BER_OFDM_InterferenceCancellation(i_SNR,i_rep,i_iteration)     = mean(BinaryDataStream_OFDM~=DetectedBitStream_OFDM_Temp);

            BER_FBMC_Aux_InterferenceCancellation_NoEdge(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Aux(ConsideredBits_FBMC_Aux)~=DetectedBitStream_FBMC_Aux_Temp(ConsideredBits_FBMC_Aux));
            BER_FBMC_Cod_InterferenceCancellation_NoEdge(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Cod(ConsideredBits_FBMC_Cod)~=DetectedBitStream_FBMC_Cod_Temp(ConsideredBits_FBMC_Cod));   
            BER_OFDM_InterferenceCancellation_NoEdge(i_SNR,i_rep,i_iteration)     = mean(BinaryDataStream_OFDM(ConsideredBits_OFDM)~=DetectedBitStream_OFDM_Temp(ConsideredBits_OFDM));

            
            % 完美信道信息
            y_FBMC_Aux_InterferenceCancellation_PerfectCSI = (y_FBMC_Aux(:) - (D_FBMC-diag(h_FBMC))*AuxiliaryMethod.PrecodingMatrix*[xP_FBMC;PAM.SymbolQuantization(xD_est_FBMC_Aux_PerfectCSI_Temp)]);                
            y_FBMC_Cod_InterferenceCancellation_PerfectCSI = (y_FBMC_Cod - (D_FBMC-diag(h_FBMC))*CodingMethod.PrecodingMatrix*[xP_FBMC;PAM.SymbolQuantization(xD_est_FBMC_Cod_PerfectCSI_Temp)]);                   
            y_OFDM_InterferenceCancellation_PerfectCSI     = (y_OFDM - (D_OFDM-diag(h_OFDM))*PilotMapping_OFDM*[xP_OFDM;QAM.SymbolQuantization(xD_est_OFDM_PerfectCSI_Temp)]);        

            x_est_FBMC_Aux_PerfectCSI_Temp = y_FBMC_Aux_InterferenceCancellation_PerfectCSI./h_FBMC;
            xD_est_FBMC_Aux_PerfectCSI_Temp = real(x_est_FBMC_Aux_PerfectCSI_Temp(AuxilaryPilotMatrix_FBMC(:)==0)./sqrt(AuxiliaryMethod.DataPowerReduction));
            DetectedBitStream_FBMC_Aux_PerfectCSI_Temp = PAM.Symbol2Bit(xD_est_FBMC_Aux_PerfectCSI_Temp);   
            BER_FBMC_Aux_PerfectCSI_InterferenceCancellation(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Aux~=DetectedBitStream_FBMC_Aux_PerfectCSI_Temp);    
            BER_FBMC_Aux_PerfectCSI_InterferenceCancellation_NoEdge(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Aux(ConsideredBits_FBMC_Aux)~=DetectedBitStream_FBMC_Aux_PerfectCSI_Temp(ConsideredBits_FBMC_Aux));    

            x_est_FBMC_Cod_PerfectCSI_Temp = CodingMethod.PrecodingMatrix'*(y_FBMC_Cod_InterferenceCancellation_PerfectCSI./h_FBMC);
            xD_est_FBMC_Cod_PerfectCSI_Temp = real(x_est_FBMC_Cod_PerfectCSI_Temp(NrPilotSymbols+1:end))/CodingMethod.DataPowerReduction;
            DetectedBitStream_FBMC_Cod_PerfectCSI_Temp = PAM.Symbol2Bit(xD_est_FBMC_Cod_PerfectCSI_Temp);   
            BER_FBMC_Cod_PerfectCSI_InterferenceCancellation(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Cod~=DetectedBitStream_FBMC_Cod_PerfectCSI_Temp);    
            BER_FBMC_Cod_PerfectCSI_InterferenceCancellation_NoEdge(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_FBMC_Cod(ConsideredBits_FBMC_Cod)~=DetectedBitStream_FBMC_Cod_PerfectCSI_Temp(ConsideredBits_FBMC_Cod));    

            x_est_OFDM_PerfectCSI_Temp = y_OFDM_InterferenceCancellation_PerfectCSI./h_OFDM;
            xD_est_OFDM_PerfectCSI_Temp = x_est_OFDM_PerfectCSI_Temp(PilotMatrix_OFDM(:)==0)./sqrt(DataPowerReduction_OFDM);   
            DetectedBitStream_OFDM_PerfectCSI_Temp = QAM.Symbol2Bit(xD_est_OFDM_PerfectCSI_Temp);   
            BER_OFDM_PerfectCSI_InterferenceCancellation(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_OFDM~=DetectedBitStream_OFDM_PerfectCSI_Temp);
            BER_OFDM_PerfectCSI_InterferenceCancellation_NoEdge(i_SNR,i_rep,i_iteration) = mean(BinaryDataStream_OFDM(ConsideredBits_OFDM)~=DetectedBitStream_OFDM_PerfectCSI_Temp(ConsideredBits_OFDM));

        end    

    end
    TimeNeededSoFar = toc;
    disp([int2str(i_rep/NrRepetitions*100) '% 完成! 剩余时间: ' int2str(TimeNeededSoFar/i_rep*(NrRepetitions-i_rep)/60) '分钟, 对应于大约. '  int2str(TimeNeededSoFar/i_rep*(NrRepetitions-i_rep)/3600) '小时']);


    %% 画出结果
    % OFDM
    figure(2);
    Markersize = 4;
    hold off;
    semilogy(M_SNR_dB_morePoints, BitErrorProbability,'Color',[1 1 1]*0.75);
    hold on;
    semilogy(M_SNR_dB, nanmean(BER_OFDM_PerfectCSI_InterferenceCancellation(:,:,end),2),'-x black','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_OFDM_InterferenceCancellation(:,:,end),2),'-s magenta','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_OFDM_InterferenceCancellation_NoEdge(:,:,end),2),'-o blue','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_OFDM_OneTapEqualizer_PerfectCSI,2),'-x','Color',[1 1 0]*0.7,'Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_OFDM_OneTapEqualizer,2),'-s red','Markersize',Markersize);
    ylim([10^-2 0.5]);
    title(['OFDM技术及其实现 ' int2str(i_rep) '/'  int2str(NrRepetitions)])
    legend({'Doubly-Flat 理论','Cancellation (完美 CSI)','Cancellation','Cancellation (无边缘)','单抽头 (完美 CSI)','单抽头'});
    ylabel('误码率');
    xlabel('信噪比 [dB]');
    
    % FBMC, 辅助导频方案
    figure(3);
    hold off;
    semilogy(M_SNR_dB_morePoints, BitErrorProbability,'Color',[1 1 1]*0.75);
    hold on;
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_PerfectCSI_InterferenceCancellation(:,:,end),2),'-x black','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_InterferenceCancellation(:,:,end),2),'-s magenta','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_InterferenceCancellation_NoEdge(:,:,end),2),'-o blue','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_OneTapEqualizer_PerfectCSI,2),'-x','Color',[1 1 0]*0.7,'Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_OneTapEqualizer,2),'-s red','Markersize',Markersize);
    ylim([10^-2 0.5]);
    semilogy([PlotIterationStepsSNRdB PlotIterationStepsSNRdB],[10^-2 10^-1],'Color',[1 1 1]*0.5,'Linewidth',1);
    title(['FBMC 辅助导频实现 ' int2str(i_rep) '/'  int2str(NrRepetitions)])
    legend({'Doubly-Flat 理论','Cancellation (完美 CSI)','Cancellation','Cancellation (无边缘)','单抽头 (完美 CSI)','单抽头'});
    ylabel('误码率');
    xlabel('信噪比 [dB]');
    
    % FBMC, 数据传播方法
    figure(4);
    hold off;    
    semilogy(M_SNR_dB_morePoints, BitErrorProbability,'Color',[1 1 1]*0.75);
    hold on;
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Cod_PerfectCSI_InterferenceCancellation(:,:,end),2),'-x black','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Cod_InterferenceCancellation(:,:,end),2),'-s magenta','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Cod_InterferenceCancellation_NoEdge(:,:,end),2),'-o blue','Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Cod_OneTapEqualizer_PerfectCSI,2),'-x','Color',[1 1 0]*0.7,'Markersize',Markersize);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Cod_OneTapEqualizer,2),'-s red','Markersize',Markersize);
    ylim([10^-2 0.5]);
    title(['FBMC Data Spreading实现 ' int2str(i_rep) '/'  int2str(NrRepetitions)])
    legend({'Doubly-Flat 理论','Cancellation (完美 CSI)','Cancellation','Cancellation (无边缘)','单抽头 (完美 CSI)','单抽头'});
    ylabel('误码率');
    xlabel('信噪比 [dB]');
    
    % FBMC, 辅助导频方案, 迭代后BER
    figure(5);
    hold off;
    semilogy(0:NrIterations, repmat(BitErrorProbability(find(PlotIterationStepsSNRdB==M_SNR_dB_morePoints)),NrIterations+1,1),'Color',[1 1 1]*0.75);
    hold on;
    Index = find(PlotIterationStepsSNRdB==M_SNR_dB);
    semilogy(0:NrIterations, [nanmean(BER_FBMC_Aux_OneTapEqualizer_PerfectCSI(Index,:),2);squeeze(nanmean(BER_FBMC_Aux_PerfectCSI_InterferenceCancellation(Index,:,:),2))],'-x black','Markersize',Markersize);
    semilogy(0:NrIterations, [nanmean(BER_FBMC_Aux_OneTapEqualizer(Index,:),2);squeeze(nanmean(BER_FBMC_Aux_InterferenceCancellation(Index,:,:),2))],'-s magenta','Markersize',Markersize);
    semilogy(0:NrIterations, [nanmean(BER_FBMC_Aux_OneTapEqualizer_NoEdge(Index,:),2);squeeze(nanmean(BER_FBMC_Aux_InterferenceCancellation_NoEdge(Index,:,:),2))],'-o blue','Markersize',Markersize);
    semilogy(0:NrIterations, repmat(nanmean(BER_FBMC_Aux_OneTapEqualizer_PerfectCSI(Index,:),2),NrIterations+1,1),'-x','Color',[1 1 0]*0.7,'Markersize',Markersize);
    semilogy(0:NrIterations, repmat(nanmean(BER_FBMC_Aux_OneTapEqualizer(Index,:),2),NrIterations+1,1),'-s red','Markersize',Markersize);
    title(['FBMC辅助导频实现 ' int2str(i_rep) '/'  int2str(NrRepetitions)])
    legend({'Doubly-Flat 理论','Cancellation (完美 CSI)','Cancellation','Cancellation (无边缘)','单抽头 (完美 CSI)','单抽头'});
    set(gca, 'XTick',0:NrIterations);
    ylabel('误码率');
    xlabel('迭代次数');
    
    pause(0.01);

end


%% 画出附加信息
fprintf('=============================\n');
fprintf('========= Data Rate =========\n');
fprintf('OFDM       |%7.2f Mbit/s  | \n', length(BinaryDataStream_OFDM)     / (OFDM.PHY.TimeSpacing*OFDM.Nr.MCSymbols)/1e6   );
fprintf('FBMC, Aux. |%7.2f Mbit/s  | \n', length(BinaryDataStream_FBMC_Aux) / (OFDM.PHY.TimeSpacing*OFDM.Nr.MCSymbols)/1e6   );
fprintf('FBMC, Cod. |%7.2f Mbit/s  | \n', length(BinaryDataStream_FBMC_Cod) / (OFDM.PHY.TimeSpacing*OFDM.Nr.MCSymbols)/1e6   );
fprintf('=============================\n');

% 功率被归一化，以便平均发射功率为1
fprintf('================================================\n');
fprintf('============== Relative SNR Shift ==============\n');
fprintf('================================================\n');
fprintf('           |    SNR    |  Data SNR  | Pilot SNR |\n');
fprintf('OFDM       |   %2.1fdB   |   %2.1fdB   |   %2.1fdB   |\n', 0, 10*log10(DataPowerReduction_OFDM), 10*log10(Kappa_OFDM))
fprintf('FBMC, Aux. |   %2.1fdB   |   %2.1fdB   |   %2.1fdB   |\n', 0, 10*log10(AuxiliaryMethod.DataPowerReduction), 10*log10(Kappa_Aux/2))
fprintf('FBMC, Cod. |   %2.1fdB   |   %2.1fdB   |   %2.1fdB   |\n', 0, 10*log10(CodingMethod.DataPowerReduction), 10*log10(Kappa_Cod/2))
fprintf('================================================\n');


