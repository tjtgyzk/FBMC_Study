function [subDecoderOut,message_out] = rx_subLDPC_decoder(subDecoderIn, decoderPara,Z)
    
    M = decoderPara.M;
    maxRowWeight = decoderPara.maxRowWeight;
    numIter = decoderPara.numIter;
    alfa = decoderPara.alfa;
    
    betaMin = decoderPara.betaMin;
    ctv = decoderPara.ctv;
    ctvRow = decoderPara.ctvRow;
    ctvRowIdx = decoderPara.ctvRowIdx;
    
	codeWordLLR = subDecoderIn;
    decoder_Q1 = codeWordLLR;
    decoder_Q2 = codeWordLLR;
     
	decoder_qq1 = zeros(M, maxRowWeight);
	decoder_rr1 = zeros(M, maxRowWeight);
	decoder_rr0 = zeros(M, maxRowWeight); 
    
    %***********Õ‚–≈œ¢***************/
    message_out = zeros(1,length(subDecoderIn));
    %******************************/
    
        for iter = 1 : 1 : numIter
		for row = 1 : 1 : M
            rowFirstIdx = ctvRowIdx(row);
            rowWeight = ctvRow(row);
            column = 1 : rowWeight; 
            
            % update q
			decoder_qq1(row, column) = decoder_Q1(ctv(rowFirstIdx + column - 1)) - decoder_rr0(row, column);
            
            % update r
            beta = betaMin;
            beta2 = betaMin;
            
            decoder_qk = decoder_qq1(row, column);
            qkSignAll = mod(length(find(decoder_qk < 0)), 2); 
            
            qkSort = sort(abs(decoder_qk));
            if(qkSort(1) < beta)
                beta = qkSort(1);
            end
            if(qkSort(2) < beta2)
                beta2 = qkSort(2);
            end
            
            qkSign = zeros(1, rowWeight);
            qkSign(column) = qkSignAll;
            negNumIdx = find(sign(decoder_qk) == -1);
            qkSign(negNumIdx) = ~qkSignAll;
            
            rr1Beta = zeros(1, rowWeight);
            qkAbs = abs(decoder_qk);
%             rr1Beta(column) = fix(beta * alfa);
            rr1Beta(column) = (beta * alfa);
            qkBetaIdx = find(qkAbs == beta);
%             rr1Beta(qkBetaIdx) = fix(beta2 * alfa);
            rr1Beta(qkBetaIdx) = (beta2 * alfa);
            
            decoder_rr1(row, column) = rr1Beta(column);
            negNumIdx1 = find(qkSign == 1);
            decoder_rr1(row, negNumIdx1) = -rr1Beta(negNumIdx1);
            
            decoder_rr0 = decoder_rr1;
            
            decoder_Q1(ctv(rowFirstIdx + column - 1)) = decoder_qq1(row, column) + decoder_rr1(row, column);
            
        end
    end
    
    message_out = decoder_Q1 ;
    
%     for iter = 1 : 1 : numIter
%         if(iter<4)
%             start=1;
%             step=1;
%             end_layer=M/Z;
%         else
%             if(mod(iter,2)==1)
%                 start=1;
%                 step=2;
%                 end_layer=M/Z-1;
%             else 
%                 start=2;
%                 step=2;
%                 end_layer=M/Z;
%             end
%         end
        %%  odd  layer
% 		for row = start : step : end_layer
%             for j=1:1:Z
%                 rowFirstIdx = ctvRowIdx((row-1)*Z+j);
%                 rowWeight = ctvRow((row-1)*Z+j);
%                 column = 1 : rowWeight; 
%             
%             % update q
%                 decoder_qq1((row-1)*Z+j, column) = decoder_Q1(ctv(rowFirstIdx + column - 1)) - decoder_rr0((row-1)*Z+j, column);
%             
%             % update r
%                 beta = betaMin;
%                 beta2 = betaMin;
%             
%                 decoder_qk = decoder_qq1((row-1)*Z+j, column);
%                 qkSignAll = mod(length(find(decoder_qk < 0)), 2); 
%             
%                 qkSort = sort(abs(decoder_qk));
%                  if(qkSort(1) < beta)
%                     beta = qkSort(1)+1;
%                 end
%                 if(qkSort(2) < beta2)
%                     beta2 = qkSort(2)+1;
%             end
%             
%             qkSign = zeros(1, rowWeight);
%             qkSign(column) = qkSignAll;
%             negNumIdx = find(sign(decoder_qk) == -1);
%             qkSign(negNumIdx) = ~qkSignAll;
%             
%             rr1Beta = zeros(1, rowWeight);
%             qkAbs = abs(decoder_qk);
%             rr1Beta(column) = fix(beta * alfa);
%             qkBetaIdx = find(qkAbs == beta);
%             rr1Beta(qkBetaIdx) = fix(beta2 * alfa);
%             
%             decoder_rr1((row-1)*Z+j, column) = rr1Beta(column);
%             negNumIdx1 = find(qkSign == 1);
%             decoder_rr1((row-1)*Z+j, negNumIdx1) = -rr1Beta(negNumIdx1);
%             
%             decoder_rr0 = decoder_rr1;
%             
%             decoder_Q1(ctv(rowFirstIdx + column - 1)) = decoder_qq1((row-1)*Z+j, column) + decoder_rr1((row-1)*Z+j, column);
%             
%             end
%      
%             
%             
%             
% %         %%  even layer  
% %             for j=1:1:Z
% %                 rowFirstIdx = ctvRowIdx((row-1)*Z+j);
% %                 rowWeight = ctvRow((row-1)*Z+j);
% %                 column = 1 : rowWeight; 
% %             
% %             % update q
% %                 decoder_qq1((row-1)*Z+j, column) = decoder_Q2(ctv(rowFirstIdx + column - 1)) - decoder_rr0((row-1)*Z+j, column);
% %             
% %             % update r
% %                 beta = betaMin;
% %                 beta2 = betaMin;
% %             
% %                 decoder_qk = decoder_qq1((row-1)*Z+j, column);
% %                 qkSignAll = mod(length(find(decoder_qk < 0)), 2); 
% %             
% %                 qkSort = sort(abs(decoder_qk));
% %                  if(qkSort(1) < beta)
% %                     beta = qkSort(1);
% %                 end
% %                 if(qkSort(2) < beta2)
% %                     beta2 = qkSort(2);
% %             end
% %             
% %             qkSign = zeros(1, rowWeight);
% %             qkSign(column) = qkSignAll;
% %             negNumIdx = find(sign(decoder_qk) == -1);
% %             qkSign(negNumIdx) = ~qkSignAll;
% %             
% %             rr1Beta = zeros(1, rowWeight);
% %             qkAbs = abs(decoder_qk);
% %             rr1Beta(column) = fix(beta * alfa);
% %             qkBetaIdx = find(qkAbs == beta);
% %             rr1Beta(qkBetaIdx) = fix(beta2 * alfa);
% %             
% %             decoder_rr1((row-1)*Z+j, column) = rr1Beta(column);
% %             negNumIdx1 = find(qkSign == 1);
% %             decoder_rr1((row-1)*Z+j, negNumIdx1) = -rr1Beta(negNumIdx1);
% %             
% %             decoder_rr0 = decoder_rr1;
% %             
% %             decoder_Q2(ctv(rowFirstIdx + column - 1)) = decoder_qq1((row-1)*Z+j, column) + decoder_rr1((row-1)*Z+j, column);
% %             
% %             end
% %             
% %             
%         end
% 
%     end


   
    
    %////////////////////////// decoding decision ////////////////////////%
  
    cHat = decoder_Q1 < 0;
    
    %/////////////////////////// decoding output /////////////////////////%
   
    subDecoderOut = cHat;