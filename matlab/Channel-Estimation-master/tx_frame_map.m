function data_out = tx_frame_map(data_in,map_flag)

switch map_flag
    case 1
        R = reshape(data_in,2,[]);    %将数据分为每2个一组
        B2D=bi2de(R','left-msb')+1;   %每组转为10进制
        q=[1+1j 1-1j -1+1j -1-1j];    %星座图
        data_after_map=q(B2D(:))/sqrt(2);          % 查表-星座映射 QPSK
        
     case 2
        R = reshape(data_in,3,[]);    %将数据分为每3个一组
        B2D=bi2de(R','left-msb')+1;   %每组转为10进制
        a=cos(pi/8); b=sin(pi/8);
        q=[a+b*1j, b+a*1j, a-b*1j, b-a*1j, -a+b*1j, -b+a*1j, -a-b*1j, -b-a*1j];  %星座图
        data_after_map=q(B2D(:));          % 查表-星座映射 8PSK       

    case 3
        R = reshape(data_in,4,[]);    %将数据分为每4个一组
        B2D=bi2de(R','left-msb')+1;   %每组转为10进制
        q=[3+3j 3+1j 1+3j 1+1j 3-3j 3-1j 1-3j 1-1j -3+3j -3+1j -1+3j -1+1j -3-3j -3-1j -1-3j -1-1j];   %星座图
        data_after_map=q(B2D(:))/sqrt(10);          % 查表-星座映射 16QAM
end  

data_out = data_after_map;
