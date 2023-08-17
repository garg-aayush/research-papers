# Research papers
This repository houses my personal summaries and notes on a variety of academic papers/blogs I have read. These summaries are intended to provide a brief overview of the papers' main points, methodologies, findings, and implications, thereby serving as quick references for myself and anyone interested.


## Papers
### 1. Denoising Diffusion Probabilistic Models, Ho et. al.
It presents a generative model that employs denoising diffusion processes to learn and generate realistic images. By iteratively adding noise and removing it, the model learns a diffusion process that captures the underlying distribution of complex image data, enabling high-quality image synthesis.

- [[`Archive link`](https://arxiv.org/abs/2006.11239)] [[`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=W-O7AZNzbzQ)] [[`Basic annotated implementation`](https://nn.labml.ai/diffusion/ddpm/index.html)]
- [**`Summary notes`**](Summaries/DDPM.md)

### 2. Improved Denoising Diffusion Probabilistic Models, Nichol A. and Dhariwal P.


### 3. Diffusion Models Beat GANs on Image Synthesis, Dhariwal P. and Nichol A.

### 4. Denoising Diffusion Implicit Models, Song et. al.
It presents a more efficient alternative sampling (DDIM) in comparison to DDPMs for high-quality image generation. By constructing non-Markovian diffusion processes, DDIMs achieve faster sampling, enabling trade-offs between computation and sample quality, and facilitating meaningful image interpolation in the latent space.

### 5. High-Resolution Image Synthesis with Latent Diffusion Models, Rombach et. al.

### 6. Prompt-to-Prompt Image Editing with Cross Attention Control, Hertz et. al.

### 7. Null-text Inversion for Editing Real Images using Guided Diffusion Models, Mokady et. al.
The paper introduces an accurate inversion technique for text-guided diffusion models, enabling intuitive and versatile text-based image modification without tuning model weights. The proposed method demonstrates high-fidelity editing of real images through pivotal inversion and NULL-text optimization, showcasing its efficacy in prompt-based editing scenarios.
num

### 8. Adding Conditional Control to Text-to-Image Diffusion Models, Lvmin Zhang and Maneesh Agarwala et. al.
The authors propose an architecture called ControlNet that enhances control over the image generation process in the diffusion/stable diffusion process, enabling the generation of specific and desired images. This is achieved by incorporating conditional inputs, such as edge maps, segmentation maps, and keypoints, into the diffusion model.

### 9. DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion, Karras et. al.

### 10. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, Podell et. al.
The paper introduces an enhanced stable diffusion model that surpasses the generating capabilities of previous versions. This is achieved by incorporating a larger UNet backbone and introducing novel conditioning schemes in the training stage.




