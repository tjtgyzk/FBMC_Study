function s = dec_to_binc(x, N)

if(x >= 0)
    s = dec2bin(x, N);
else 
    s = dec2bin(2^N + x, N);
end