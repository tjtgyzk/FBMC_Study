clear; close all;
addpath('./Theory');
addpath(genpath(pwd));
%% 参数
% 仿真
M_SNR_dB                  = [10:5:40];              % 信噪比dB
NrRepetitions             = 10000;                     % 蒙特卡罗重复次数（不同信道实现）             
ZeroThresholdSparse       = 10;                      % 将一些小于�??10^�?-ZeroThresholdSparse）�?�的矩阵值设置为零�??
PlotIterationStepsSNRdB   = 35;                     % 绘制SNR�?35dB的迭代步骤上的误码率�?

% FBMC与OFDM参数
L                         = 12*2;                   % 子载波数，一个资源块�?12个子载波组成（时间为0.5ms�?
F                         = 15e3;                   % 子载波间隔（Hz�?
SamplingRate              = F*12*2;                 % 采样率（采样�?/秒）
NrSubframes               = 1;                      % 子帧的数目�?�F=15kHz时，�?个子帧需�?1ms�?                             
QAM_ModulationOrder       = 4;                      % QAM信号星座顺序�?4�?16�?64�?256�?1024，�?��?��??

% 信道估计参数
PilotToDataPowerOffsetAux = 4.685;                  % FBMC的导频到数据功率偏移，辅助方法�??
NrIterations              = 4;                      % 干扰消除方案的迭代次数�??
% 信道编码参数
 codeRate = [11/21]; %1/3 1/2 2/3 3/4 5/6 8/9
 BG = 1;
 Z = 16;
 numFrame = 1;
 frameAmount = 1;
 AI = LDPC_para_ini(BG, Z, codeRate, numFrame);
 encoderPara = encoder_para_ini(AI);
 decoderPara = decoder_para_ini(AI);    
 errorFrameNum = zeros(1, length(M_SNR_dB(:)));

%% FBMC对象
FBMC = Modulation.FBMC(...
    L,...                               % 子载波数
    30*NrSubframes,...                  % FBMC符号�?
    F,...                               % 子载波间隔（Hz�?
    SamplingRate,...                    % 采样率（采样�?/秒）
    0,...                               % 中频第一副载波（Hz�?
    false,...                           % 传输实�?�信�?
    'PHYDYAS-OQAM',...                  % 原型滤波器（Hermite、PHYDYAS、RRC）和OQAM或QAM�? 
    8, ...                              % 重叠因子（还确定频域中的过采样）
    0, ...                              % 初始相移
    true ...                            % 多相实现
    );
N = FBMC.Nr.SamplesTotal;


%% PAM和QAM对象
PAM = Modulation.SignalConstellation(sqrt(QAM_ModulationOrder),'PAM');
QAM = Modulation.SignalConstellation(QAM_ModulationOrder,'QAM');

%% 导频矩阵�?0=“数据�?�，1=导频
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

%% 消除导频位置对象处的假想干扰（预编码矩阵�?
AuxiliaryMethod = ChannelEstimation.ImaginaryInterferenceCancellationAtPilotPosition(...
    'Auxiliary', ...                                    % 干扰消除方式
    AuxilaryPilotMatrix_FBMC, ...                       % 辅助导频矩阵
    FBMC.GetFBMCMatrix, ...                             % 虚干扰矩�?
    28, ...                                             % 消除�?�?28个干扰源
    PilotToDataPowerOffsetAux ...                       % 导频到数据功率偏�?
    );
NrPilotSymbols = sum(PilotMatrix_FBMC(:)==1);
Kappa_Aux  = AuxiliaryMethod.PilotToDataPowerOffset*AuxiliaryMethod.DataPowerReduction;
global rayChan0;


for i_rep = 1:NrRepetitions
    for i_SNR = 1:length(M_SNR_dB)
    reset (rayChan0);  %信道初始�?
    
    %% 二进制数�?
%     BinaryDataStream_FBMC_Aux = randi([0 1],352*2,1);
    BinaryDataStream_FBMC_Aux = randi([0 1],352,1);
    BinaryDataStream_FBMC_Aux_Coded = tx_encoder(encoderPara,BinaryDataStream_FBMC_Aux);
    %% 数据符号
     xD_FBMC_Aux = PAM.Bit2Symbol(BinaryDataStream_FBMC_Aux_Coded); 
  %  xD_FBMC_Aux = -2*BinaryDataStream_FBMC_Aux_Coded + 1 ;
    %% 导频符号
    xP_FBMC = PAM.SymbolMapping(randi(PAM.ModulationOrder,AuxiliaryMethod.NrPilotSymbols,1));
    xP_FBMC = xP_FBMC./abs(xP_FBMC);
   
    %% 传输的数据符�?
    x_FBMC_Aux = AuxiliaryMethod.PrecodingMatrix*[xP_FBMC;xD_FBMC_Aux];
    x_FBMC_Aux = reshape(x_FBMC_Aux,[L,30*NrSubframes]);            
    %% 发射信号 (时域)
    s_FBMC_Aux = FBMC.Modulation(x_FBMC_Aux); % Same as "FBMC.Modulation(x_FBMC_Aux)" which is computationally more efficient. But G_FBMC is consistent with the paper.
    
    %% 信道
    Chennel = comm.RayleighChannel('SampleRate',SamplingRate, ...
            'NormalizePathGains',true, ...
            'MaximumDopplerShift',20,...
            'RandomStream', 'mt19937ar with seed',...%'Global stream'  'mt19937ar with seed'
            'Seed',73);
    
    
        %% 增加噪声
        SNR_dB  = M_SNR_dB(i_SNR);
        %Pn_time = SamplingRate/(F*L)*10^(-SNR_dB/10);
        %noise   = sqrt(Pn_time/2)*(randn(size(s_FBMC_Aux))+1j*randn(size(s_FBMC_Aux)));
        r_FBMC_Aux_noNoise = Chennel(s_FBMC_Aux);
        r_FBMC_Aux = awgn(r_FBMC_Aux_noNoise,SNR_dB,'measured','db'); 
        %r_FBMC_Aux = r_FBMC_Aux_noNoise;
        %r_FBMC_Aux = s_FBMC_Aux;
        %% 解调FBMC信号
        y_FBMC_Aux = FBMC.Demodulation(r_FBMC_Aux); % Same as "FBMC.Demodulation(r_FBMC_Aux)"
        y_FBMC_Aux = reshape(y_FBMC_Aux,L*30*NrSubframes,1);
        %% 导频位置的信道估�?
        hP_est_FBMC_Aux = y_FBMC_Aux(PilotMatrix_FBMC==1)./xP_FBMC/sqrt(Kappa_Aux);
        %% 插�??
        index = [50 62 80 92 245 257 275 287 434 446 464 476 629 641 659 671];
        %plot(index,hP_est_FBMC_Aux')
        h_est_interp = interp1(index,hP_est_FBMC_Aux',(1:720),'pchip');
        h_est_interp = h_est_interp';
        %% 数据恢复
        x_est_OneTapEqualizer_FBMC_Aux = y_FBMC_Aux./h_est_interp;
        xD_est_OneTapEqualizer_FBMC_Aux = real(x_est_OneTapEqualizer_FBMC_Aux(AuxilaryPilotMatrix_FBMC(:)==0)./sqrt(AuxiliaryMethod.DataPowerReduction));
        DetectedBitStream_OneTapEqualizer_FBMC_Aux = PAM.Symbol2Bit(xD_est_OneTapEqualizer_FBMC_Aux);
%         data_demod = DetectedBitStream_OneTapEqualizer_FBMC_Aux';
        decoderInBitAdd = tx_encoder_bitadd(xD_est_OneTapEqualizer_FBMC_Aux', encoderPara);
        [dataOut] = Quantization(decoderInBitAdd);
        [decoderOut,message_out] = rx_decoder(dataOut, decoderPara);
      %  DetectedBitStream_OneTapEqualizer_FBMC_Aux_De = vitdec(DetectedBitStream_OneTapEqualizer_FBMC_Aux,trel,tblen,'cont','hard');
        
      %  BER_FBMC_Aux_OneTapEqualizer(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux~=DetectedBitStream_OneTapEqualizer_FBMC_Aux_De); 
      decoderOut = decoderOut(1 : AI.K * Z);
      decoderOut = decoderOut';
      BER_FBMC_Aux_OneTapEqualizer_C(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux(1:end) ~= decoderOut(1 : AI.K * Z)); 
        %% 迭代
%        NewBinary_Before_Coding = [DetectedBitStream_OneTapEqualizer_FBMC_Aux_De(tblen+1:end);zeros(tblen,1)];
%        NewBinary = convenc(NewBinary_Before_Coding,trel);
%        xD1_FBMC_Aux = PAM.Bit2Symbol(NewBinary);    
        
 %   xP1_FBMC = PAM.SymbolMapping(randi(PAM.ModulationOrder,AuxiliaryMethod.NrPilotSymbols,1));
 %   xP1_FBMC = xP1_FBMC./abs(xP1_FBMC);
    
%        x1_FBMC_Aux = AuxiliaryMethod.PrecodingMatrix*[xP_FBMC;xD1_FBMC_Aux];
%        x1_FBMC_Aux = reshape(x1_FBMC_Aux,[L,30*NrSubframes]);            
   
%        s1_FBMC_Aux = FBMC.Modulation(x1_FBMC_Aux);
%        y1_FBMC_Aux = FBMC.Demodulation(s1_FBMC_Aux);
%        y1_FBMC_Aux = reshape(y1_FBMC_Aux,L*30*NrSubframes,1);
     
%        hP1_est_FBMC_Aux = y_FBMC_Aux(PilotMatrix_FBMC==1)./xP_FBMC/sqrt(Kappa_Aux);
     
%        index = [50 62 80 92 245 257 275 287 434 446 464 476 629 641 659 671];
     
%        h1_est_interp = interp1(index,hP1_est_FBMC_Aux',(1:720),'pchip');
%        h1_est_interp = h1_est_interp';
     %% 数据恢复
%        x1_est_OneTapEqualizer_FBMC_Aux = y_FBMC_Aux./h1_est_interp;
%        xD1_est_OneTapEqualizer_FBMC_Aux = real(x1_est_OneTapEqualizer_FBMC_Aux(AuxilaryPilotMatrix_FBMC(:)==0)./sqrt(AuxiliaryMethod.DataPowerReduction));
%        DetectedBitStream1_OneTapEqualizer_FBMC_Aux = PAM.Symbol2Bit(xD1_est_OneTapEqualizer_FBMC_Aux);   
%        DetectedBitStream1_OneTapEqualizer_FBMC_Aux_De = vitdec(DetectedBitStream1_OneTapEqualizer_FBMC_Aux,trel,tblen,'cont','hard');
      %  BER1_FBMC_Aux_OneTapEqualizer(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux~=DetectedBitStream1_OneTapEqualizer_FBMC_Aux); 
%        BER1_FBMC_Aux_OneTapEqualizer_C(i_SNR,i_rep) = mean(BinaryDataStream_FBMC_Aux(1:end - tblen)~=DetectedBitStream1_OneTapEqualizer_FBMC_Aux_De(tblen+1:end));% - randi([0,2],1)*0.0001/i_SNR - randi([0,2],1)*0.000001; 
      %  BER1_FBMC_Aux_OneTapEqualizer_C(i_SNR,i_rep) = mean(NewBinary_Before_Coding(1:end - tblen)~=DetectedBitStream1_OneTapEqualizer_FBMC_Aux_De(tblen+1:end)); 
        
    end
    disp([int2str(i_rep/NrRepetitions*100) '% 完成! ']);
    figure(1);
    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_OneTapEqualizer_C(:,:),2),'red',M_SNR_dB, nanmean(BER_FBMC_Aux_OneTapEqualizer_C(:,:),2),'blue');
    load WithOutCoding.mat
    xlim([0 40]);
%    semilogy(M_SNR_dB, nanmean(BER_FBMC_Aux_OneTapEqualizer_C(:,:),2),'blue',M_SNR_dB,nanmean(BER_FBMC_Aux_OneTapEqualizer(:,:),2),'red');
    
end
 
 