function OQAMdata = OQAM_modulater(QAMData,i)
OQAMdata = zeros(2*length(QAMData),1);
if rem(i,2)==1
    OQAMdata(1:2:end) = real(QAMData);
    OQAMdata(2:2:end) = imag(QAMData)*1i;
else
    OQAMdata(1:2:end) = imag(QAMData)*1i;
    OQAMdata(2:2:end) = real(QAMData);
end
end