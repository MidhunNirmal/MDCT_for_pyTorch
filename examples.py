"""
Example demonstrating the mdct and inverse mdct conversion.

When 16,384 values are transformed using the Modified Discrete Cosine Transform (MDCT),
the result is reduced to 8,192 values due to the transformation process. 
However, applying the inverse MDCT to these 8,192 values restores the original 16,384 values, 
completing the round-trip transformation without loss of information. 
This process is commonly used in audio and image compression techniques to 
efficiently represent and reconstruct signals while minimizing data size.
"""

import torch
from mdct_torch import mdct_convert, imdct_convert

# Example usage of mdct_convert function
# Assume audio_signal is a torch.tensor of shape [batch_size, num_samples]

# Generate a random audio signal tensor (for demonstration purposes)
batch_size = 2
num_samples = 16384
audio_signal = torch.randn(batch_size, 1, num_samples)

# Convert audio signal to MDCT using batch processing
batch_mdct = mdct_convert(audio_signal, batch_size)

# Print the shape of the output MDCT tensor
print("MDCT Output Shape:", batch_mdct.shape)

# Example usage of imdct_convert function
# Assume audio_mdct is a torch.tensor of shape [batch_size, 1, mdct_length]

# Convert MDCT back to audio signal using batch processing
batch_imdct = imdct_convert(batch_mdct, batch_size)

# Print the shape of the output iMDCT tensor
print("iMDCT Output Shape:", batch_imdct.shape)
