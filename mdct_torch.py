import numpy as np
import torch
from mdct import mdct
from imdct import imdct


def mdct_convert(audio_signal,batch):
    
    """Convert audio signal to mdct for pytorch

    Args:
        audio_signal (torch.tensor): Audio signal of type torch.tensor [B x 1 x 16384]
        batch (int): Batch size for training of audio

    Returns:
        torch.float32: Returns audio signal after mdct conversion [B x 1 x 8192]
    """
    
    # Audio signal is converted from torch.tensor to numpy.ndarray 
    audio_signal_np = audio_signal.cpu().detach().numpy()
    mdct_results = []
    
    for i in range(batch):
        audio_mdct = mdct(audio_signal_np[i][0])
        mdct_results.append(audio_mdct) # Out as list
        
    # List to array
    mdct_array = np.array(mdct_results)
    
    # reshape the array according to requirement
    reshaped_array = mdct_array.reshape(batch, 1, 8192) # [B x 1 x 8192]
    
    # Conversion to torch.tensor
    output = torch.from_numpy(reshaped_array)
    
    # Conversion to torch.float32
    output=output.to(torch.float32) #Datatype can be changed according to requirement
    
    return output


def imdct_convert(audio_mdct,batch):
    """Convert audio mdct to audio signal for pytorch

    Args:
        audio_mdct (torch.float32): Audio mdct of type torch.tensor [B x 1 x 8192]
        batch (int): Batch size for training of audio

    Returns:
        torch.float32: Returns audio signal after inverse mdct conversion [B x 1 x 16384]
    """
    audio_mdct_np = audio_mdct.cpu().detach().numpy()
    imdct_results = []
    
    for i in range(batch):
        audio_mdct = imdct(audio_mdct_np[i][0])
        imdct_results.append(audio_mdct) # Out as list
        
    # List to array
    imdct_array = np.array(imdct_results)
    
    # reshape the array according to requirement
    reshaped_array = imdct_array.reshape(batch, 1, 16384) # [B x 1 x 16384]
    
    # Conversion to torch.tensor
    output = torch.from_numpy(reshaped_array)
    
    # Conversion to torch.float32
    output=output.to(torch.float32) #Datatype can be changed according to requirement
    
    return output