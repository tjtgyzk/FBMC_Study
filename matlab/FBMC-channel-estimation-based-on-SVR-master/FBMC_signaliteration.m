%author:KAI
function [BinaryDataStream_FBMC_Aux,xP_FBMC,x_FBMC_Aux,s_FBMC_Aux]= FBMC_signaliteration(LastData,Last_xP,AuxiliaryMethod,FBMC,PAM,ChannelEstimation_FBMC)
    BinaryDataStream_FBMC_Aux = LastData;
    xD_FBMC_Aux = PAM.Bit2Symbol(BinaryDataStream_FBMC_Aux);    
   % xP_FBMC = Last_xP;
    xP_FBMC = PAM.SymbolMapping(randi(PAM.ModulationOrder,[ChannelEstimation_FBMC.NrPilotSymbols 1]));
    xP_FBMC = xP_FBMC./abs(xP_FBMC);
    x_FBMC_Aux = reshape(AuxiliaryMethod.PrecodingMatrix*[xP_FBMC;xD_FBMC_Aux],[FBMC.Nr.Subcarriers FBMC.Nr.MCSymbols]);
    s_FBMC_Aux = FBMC.Modulation(x_FBMC_Aux); 
end
