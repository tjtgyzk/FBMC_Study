function RE = OQAM_demodulater(reDown,i)
dataREreal = zeros(length(reDown)/2,1);        
dataREimag = zeros(length(reDown)/2,1);        
R2 = real(reDown);
I2 = imag(reDown);
if rem(i,2)==1
    dataREreal(1:1:end) = R2(1:2:end);
    dataREimag(1:1:end) = I2(2:2:end);
else
    dataREreal(1:1:end) = R2(2:2:end);
    dataREimag(1:1:end) = I2(1:2:end);
end
RE = complex(dataREreal,dataREimag);
end