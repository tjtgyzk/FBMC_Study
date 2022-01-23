function [decoderOut,message_out] = rx_decoder(decoderIn, decoderPara)

    numFrame = decoderPara.numFrame;
    Z = decoderPara.Z;
    H_N = decoderPara.H_N;
 
    for i = 1 : numFrame
%         decoderInOneFrame = decoderIn((H_N - 2) * Z * (i - 1) + 1 : (H_N - 2) * Z * i);
%         addZero = zeros(1, 2 * Z);
%         subDecoderIn = [addZero, decoderInOneFrame];
%         subDecoderOut = rx_subLDPC_decoder(subDecoderIn, decoderPara,Z);
%         message_in = zeros(1,length(subDecoderIn)) ;
%         for j = 1:1:10
%             data_in = subDecoderIn - message_in ;
            [subDecoderOut,message_out] = rx_subLDPC_decoder(decoderIn, decoderPara,Z);
%             message_in = message_out ;
%         end
        decoderOut(H_N * Z * (i - 1) + 1 : H_N * Z * i) = subDecoderOut(1 : H_N * Z);
    end
    
    
