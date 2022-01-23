function  data_out = rx_qamdemod(data_in,map_flag) 

switch map_flag
    case 1
        rxDataSoft = zeros(1,2*length(data_in));
        rxDataSoft(1:2:end) = real(data_in);
        rxDataSoft(2:2:end) = imag(data_in);

    case 2
        rxDataSoft = zeros(1,3*length(data_in));
        rxDataSoft(1:3:end) = real(data_in);
        rxDataSoft(2:3:end) = imag(data_in);
        rxDataSoft(3:3:end) = abs(real(data_in))-abs(imag(data_in));
         
    case 3
        rxDataSoft = zeros(1,4*length(data_in));
        rxDataSoft(1:4:end) = real(data_in);
        rxDataSoft(2:4:end) = imag(data_in);
        rxDataSoft(3:4:end) = abs(real(data_in))-2/sqrt(10);
        rxDataSoft(4:4:end) = abs(imag(data_in))-2/sqrt(10);
end

data_out = rxDataSoft;

end
