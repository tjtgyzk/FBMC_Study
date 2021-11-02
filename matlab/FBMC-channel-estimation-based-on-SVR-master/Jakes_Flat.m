function [h,tf]=Jakes_Flat(FBMC,NrTime)
% Inputs:
%   fd      : ������Ƶ��
%   Ts      : ��������
%   Ns      : ��������
%   t0      : ��ʼʱ��
%   E0      : �ŵ�����
%   phi_N  : ��������Ƶ�����Ҳ��ĳ�ʼ��λ
% Outputs:
%   h       : ��˥������
%   t_state: ��ǰʱ��
fd= 2500;
Ts= 1./FBMC.PHY.SamplingRate;
Ns= 50000;
t0= 0;
E0=1;
phi_N=0;
N0=8;                  % As suggested by Jakes 
N=4*N0+2;             % ��ȷ����             
wd=2*pi*fd;           % Maximum doppler frequency[rad]
%{
%t_state = t0;
%for i=1:Ns
%   ich=sqrt(2)*cos(phi_N)*cos(wd*t_state);
%   qch=sqrt(2)*sin(phi_N)*cos(wd*t_state);
%   for k=1:N0
%      phi_n=pi*k/(N0+1);
%      wn=wd*cos(2*pi*k/N);
%      ich=ich+2*cos(phi_n)*cos(wn*t_state);
%      qch=qch+2*sin(phi_n)*cos(wn*t_state);
%   end
%   h1(i) = E0/sqrt(2*N0+1)*complex(ich,qch);
%   t_state=t_state+Ts;             % save last time
%end
%}
t = t0+[0:Ns-1]*Ts;  tf = t(end)+Ts; 
coswt = [sqrt(2)*cos(wd*t); 2*cos(wd*cos(2*pi/N*[1:N0]')*t)]; % �� (2.32)
h = E0/sqrt(2*N0+1)*exp(1j*[phi_N pi/(N0+1)*[1:N0]])*coswt; %�� (2.29) with (2.30),(2.31), and (2.32)
index=sort(randperm(Ns,NrTime));
h=h(index).';
% discrepancy = norm(h-h1)
end