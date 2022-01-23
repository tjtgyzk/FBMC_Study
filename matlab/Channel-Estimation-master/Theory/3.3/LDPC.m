    clear  ,clc;
    
    tic;
    
    codeRate = [11/21]; %1/3 1/2 2/3 3/4 5/6 8/9
    BG = 1;
    Z = 16;
    numFrame = 1;
    frameAmount = 1;
    snr = [3 3.2 3.4 3.6 3.8 4 ];
    
    mapflag = 1 ;
    
    for i = 1 : 1 : length(codeRate(:))
    
        AI = LDPC_para_ini(BG, Z, codeRate(i), numFrame);
        encoderPara = encoder_para_ini(AI);
        decoderPara = decoder_para_ini(AI);
        
        errorFrameNum = zeros(1, length(snr(:)));
    
        for j = 1 : 1 : length(snr(:))
    
            frameErrorBits = zeros(1, frameAmount);
    
            for frame = 1 : 1 : frameAmount
        
                [encoderData, encoderIn] = tx_encoder(encoderPara);
                
                decoderIn = tx_frame_map(encoderData,1);
%                 decoderIn = -2 * encoderData + 1;
%                decoderIn = 32 * (-2 * encoderData + 1);
                
                chDataOut = awgn(decoderIn, snr(j), 'measured');       
                
                data_demod = rx_qamdemod(chDataOut,1) ;
                   
                decoderInBitAdd = tx_encoder_bitadd(data_demod, encoderPara); 
             
                decoderInQuan = Quantization(decoderInBitAdd);

                [decoderOut,message_out] = rx_decoder(decoderInBitAdd, decoderPara);
                
                decodeFrameErrorBits = decoderOut(1 : AI.K * Z) - encoderIn;
                
                frameErrorBits(frame) = sum(abs(decodeFrameErrorBits));
                
                if (frameErrorBits(frame) ~= 0)
                    errorFrameNum(1, j) = errorFrameNum(1, j) + 1;
                end
                
                numFrameIdx = frame
            end
            
            errorFrameRate = errorFrameNum / frameAmount
            
            errorBits = sum(frameErrorBits);
            errorBitsRate(i, j) = errorBits / (AI.K * Z * numFrame * frameAmount) %#ok<*SAGROW,*NOPTS>
        
        end
    end
    
    toc;