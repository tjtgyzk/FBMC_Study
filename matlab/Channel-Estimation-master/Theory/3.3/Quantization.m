function [dataOut] = Quantization(dataIn)

 maxValue = 5;
%  maxValue = 512;
% addZero = zeros(1, 2 * 128);
% dataIn = [addZero,dataIn(2*128+1:end)];
resout = zeros(1, length(dataIn));
for i = 1 : 1 : length(dataIn)
    if(dataIn(i) > maxValue)
        resout(i) = maxValue;
    else
        if(dataIn(i) < -maxValue)
            resout(i) = -maxValue; 
        else
             resout(i) = dataIn(i);
        end
    end
end

dataOut = zeros(1, length(dataIn));
for i = 1 : 1 : length(resout)
    if (resout(i) == maxValue)
		dataOut(i) = floor((resout(i) / maxValue) * 64) - 1;
	else
		dataOut(i) = floor((resout(i) / maxValue) * 64);
    end
end
        


