load decoded_data.txt;
load decoder_out.txt;

Z = 384;
section_num = 1;

comp_data = zeros(1, section_num);
for k = 1 : section_num
    A = decoded_data((k - 1) * 22 * Z + 1 : k * Z * 22);
    B = decoder_out(1 : Z * 22);
    comp_data(k) = sum(abs(A - B));
end

sum(comp_data)