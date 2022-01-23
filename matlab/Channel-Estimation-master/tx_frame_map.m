function data_out = tx_frame_map(data_in,map_flag)

switch map_flag
    case 1
        R = reshape(data_in,2,[]);    %�����ݷ�Ϊÿ2��һ��
        B2D=bi2de(R','left-msb')+1;   %ÿ��תΪ10����
        q=[1+1j 1-1j -1+1j -1-1j];    %����ͼ
        data_after_map=q(B2D(:))/sqrt(2);          % ���-����ӳ�� QPSK
        
     case 2
        R = reshape(data_in,3,[]);    %�����ݷ�Ϊÿ3��һ��
        B2D=bi2de(R','left-msb')+1;   %ÿ��תΪ10����
        a=cos(pi/8); b=sin(pi/8);
        q=[a+b*1j, b+a*1j, a-b*1j, b-a*1j, -a+b*1j, -b+a*1j, -a-b*1j, -b-a*1j];  %����ͼ
        data_after_map=q(B2D(:));          % ���-����ӳ�� 8PSK       

    case 3
        R = reshape(data_in,4,[]);    %�����ݷ�Ϊÿ4��һ��
        B2D=bi2de(R','left-msb')+1;   %ÿ��תΪ10����
        q=[3+3j 3+1j 1+3j 1+1j 3-3j 3-1j 1-3j 1-1j -3+3j -3+1j -1+3j -1+1j -3-3j -3-1j -1-3j -1-1j];   %����ͼ
        data_after_map=q(B2D(:))/sqrt(10);          % ���-����ӳ�� 16QAM
end  

data_out = data_after_map;
