clear all;
EbNo = 1:10;
N = 1000;
M = 2;
L = 7;
trel = poly2trellis(L,[171,133]);
tblen = 6*L;
msg = randi([0,1],1,N);
msg1 = convenc(msg,trel);
x1 = pskmod(msg1,M);
for i = 1:length(EbNo)
    y = awgn(x1,EbNo(i)-3);
    y1 = pskdemod(y,M);
    y2 = vitdec(y1,trel,tblen,'cont','hard');
    [err, ber1(i)] = biterr(y2(tblen+1:end),msg(1:end - tblen));
    y3 = vitdec(real(y),trel,tblen,'cont','unquant');
    [err,ber2(i)] = biterr(y3(tblen+1:end),msg(1:end-tblen));
end
ber = berawgn(EbNo,'psk',2,'nodiff');
figure();
semilogy(EbNo,ber,EbNo,ber1,EbNo,ber2);
legend("BPSK理论","硬判决","软判决");
xlabel('EbNo');
ylabel("BER");  