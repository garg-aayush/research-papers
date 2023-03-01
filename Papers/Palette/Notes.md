# Palette: Image-to-Image Diffusion Models

## Introduction
1. **Single-image super-resolution**
    - The image-to-image translation task of generating a high-resolution image that is consistent with an input low-resolution image.
    -  It is a challenging problem because:
       - Multiple output images may be consistent with a single input image.
       - The conditional distribution of output images given the input typically does not conform well to simple parametric distributions sucn as multivariate Gaussian.
  
2. **SR3 (Super-Resolution via Repeated Refinement)**
   - It is a new approach to conditional image generation based on DDPMs and and denoising score matching.
   - It works by learning to transform a standard normal distribution into an empirical data distribution through a sequence of refinement steps, resembling Langevin dynamics. 
   - It uses a U-Net architecture that is trained with a denoising objective to iteratively remove various levels of noise from the output. 
   - It adapt DDPMs to conditional image generation by a change in U-Net architecture.
   - It minimizes a well-defined loss function and uses a constant number of inference steps regardless of output resolution.
   - It can be cascaded, e.g., going from 64×64 to 256×256, and then to 1024×1024. Cascading allows to independently train a few small models rather than a single large model with a high magnification factor.
   - It is effective on both faces and natural images.


3. **Evaluation Metrics**
   - Image quality scores like PSNR and SSIM do not reflect human preference well when the input resolution is low and the magnification ratio is large. These quality scores often penalize synthetic high-frequency details, such as hair texture, because synthetic details do not perfectly align with the reference details. 
   - They use human evaluation to compare the quality of super-resolution methods and calculate the fool rate.

