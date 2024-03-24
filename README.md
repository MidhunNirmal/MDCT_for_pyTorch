# PyTorch MDCT and Inverse MDCT Conversion

This repository provides an implementation of the **Modified Discrete Cosine Transform (MDCT)** and its inverse using **PyTorch tensors**, along with example scripts demonstrating their usage for audio signal processing.

## Files Overview

- `examples.py`: Contains examples demonstrating the MDCT and inverse MDCT conversion processes using PyTorch tensors.
- `mdct_torch.py`: Includes functions for converting audio signals to MDCT and vice versa using PyTorch tensors, tailored for PyTorch-specific operations.
- `mdct.py` and `imdct.py`: Implement the MDCT and inverse MDCT algorithms, respectively, compatible with PyTorch.

## How It Works

### MDCT Conversion
1. `mdct_convert`: Takes an audio signal as a PyTorch tensor (audio_signal) and batch size (batch). It converts the audio signal to MDCT format using the MDCT algorithm implemented in mdct.py.
   - Converts the input audio signal tensor to a numpy array for processing.
   - Performs MDCT transformation using the MDCT algorithm.
   - Reshapes the MDCT results to match the required output shape of [batch x 1 x 8192].
   - Returns the MDCT output as a PyTorch tensor of type torch.float32.

### Inverse MDCT Conversion
2. `imdct_convert`: Takes an MDCT representation tensor (audio_mdct) and batch size (batch) as inputs. It performs the inverse MDCT conversion to reconstruct the original audio signal.
   - Converts the input MDCT tensor to a numpy array for processing.
   - Utilizes the inverse MDCT algorithm implemented in imdct.py to perform the conversion.
   - Reshapes the reconstructed audio signal to match the original shape of [batch x 1 x 16384].
   - Returns the reconstructed audio signal as a PyTorch tensor of type torch.float32.

## Example Usage
The `examples.py` script demonstrates how to use the `mdct_convert` and `imdct_convert` functions with randomly generated audio signals within a PyTorch environment.

1. Generates a random audio signal tensor using PyTorch.
2. Converts the audio signal to MDCT format using mdct_convert.
3. Prints the shape of the MDCT output.
4. Converts the MDCT back to the audio signal using imdct_convert.
5. Prints the shape of the reconstructed audio signal.

## Dependencies
- `numpy`: Used for numerical computations and array operations.
- `scipy`: Required for FFT and inverse FFT operations within the MDCT and inverse MDCT algorithms.
- `torch`: Essential for tensor operations and compatibility with PyTorch functionalities.

## Conclusion
This implementation provides a way to perform MDCT and inverse MDCT transformations efficiently using PyTorch tensors, which is beneficial for audio compression and signal processing tasks. Users can adjust batch sizes and data types based on specific requirements while leveraging the power of PyTorch for tensor computations.

## References

- MDCT & Inverse MDCT ([dhroth](https://github.com/dhroth/pytorch-mdct)) 

<!-- ## License

Specify the license under which your project is distributed. For example:
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. -->

## Contributors

- [Midhuhn Nirmal](https://github.com/MidhunNirmal)
- [Avinash K B](https://github.com/avinash-panikkan)
- [Diljith P A](https://github.com/dilji)
- [Jithu Johan Jose](https://github.com/RoyalewidCheese)

