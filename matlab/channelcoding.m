clear;
clc;
L = 7;
data = randi([0,1],100,1);
data = data';
tblen = 6*L;
trel = poly2trellis(L,[133,171]);
coded = convenc(data,trel);
x = zeros(1,2*tblen);
coded_zero = [coded,x];
decoded_zero = vitdec(coded_zero,trel,tblen,'cont','hard');
decoded = decoded_zero(tblen+1:end);
%decoded = decoded_zero(1:length(data));
[number,ratio] = biterr(decoded,data)