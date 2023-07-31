# Research papers
This repository houses my personal summaries and notes on a variety of academic papers/blogs I have read. These summaries are intended to provide a brief overview of the papers' main points, methodologies, findings, and implications, thereby serving as quick references for myself and anyone interested.


## Papers
### 1. Attention Is All You Need, Vaswani et. al.

The paper introduces the `Transformer` model, a neural network architecture that solely relies on self-attention mechanisms, eliminating the need for recurrent or convolutional layers. This approach achieves SOTA results in number of NLP taks, revolutionizing the field using the power of attention mechanisms.

- [[`Archive link`](https://arxiv.org/abs/1706.03762)] [[`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=iDulhoQ2pro&t=2s)] [[`Basic annotated implementation`](http://nlp.seas.harvard.edu/annotated-transformer/)]
- [**`Summary notes`**](Summaries/Attention_Is_All_You_Need.md)
   
### 2. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy et. al.

The paper introduces the concept of using Transformers, originally designed for NLP, for image recognition tasks. By dividing images into patches and leveraging self-attention mechanisms, this approach achieves competitive results on large-scale image recognition benchmarks, challenging the traditional convolutional neural network paradigm.

### 3. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, Liu et. al.

The paper proposes a hierarchical vision Transformer architecture that uses shifted windows to capture both local and global information in images. By leveraging hierarchical representations and efficient computation, Swin Transformer achieves strong performance on various vision tasks, surpassing previous Transformer-based models while maintaining computational efficiency.

### 4. Denoising Diffusion Probabilistic Models, Ho et. al.

It presents a generative model that employs denoising diffusion processes to learn and generate realistic images. By iteratively adding noise and removing it, the model learns a diffusion process that captures the underlying distribution of complex image data, enabling high-quality image synthesis.

- [[`Archive link`](https://arxiv.org/abs/2006.11239)] [[`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=W-O7AZNzbzQ)] [[`Basic annotated implementation`](https://nn.labml.ai/diffusion/ddpm/index.html)]
- [**`Summary notes`**](Summaries/DDPM.md)]

### 5. Denoising Diffusion Implicit Models, Song et. al.

It presents a more efficient alternative sampling (DDIM) in comparison to DDPMs for high-quality image generation. By constructing non-Markovian diffusion processes, DDIMs achieve faster sampling, enabling trade-offs between computation and sample quality, and facilitating meaningful image interpolation in the latent space.



### 6. High-Resolution Image Synthesis with Latent Diffusion Models, Rombach. et. al.

It introduces the approach behind the `Stable Diffusion`. It proposes in the latent space of pretrained autoencoders, enabling near-optimal complexity reduction, detail preservation, and flexible generation for various conditioning inputs with improved visual fidelity.


### 7. Adding Conditional Control to Text-to-Image Diffusion Models, Lvmin Zhang and Maneesh Agarwala et. al.

The authors propose an architecture called ControlNet that enhances control over the image generation process in the diffusion/stable diffusion process, enabling the generation of specific and desired images. This is achieved by incorporating conditional inputs, such as edge maps, segmentation maps, and keypoints, into the diffusion model.


### 8. Null-text Inversion for Editing Real Images using Guided Diffusion Models, Mokady et. al.

The paper introduces an accurate inversion technique for text-guided diffusion models, enabling intuitive and versatile text-based image modification without tuning model weights. The proposed method demonstrates high-fidelity editing of real images through pivotal inversion and NULL-text optimization, showcasing its efficacy in prompt-based editing scenarios.
num
   
### 9. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, Podell et. al.

The paper introduces an enhanced stable diffusion model that surpasses the generating capabilities of previous versions. This is achieved by incorporating a larger UNet backbone and introducing novel conditioning schemes in the training stage.

### 10. Photoswap: Personalized Subject Swapping in Images, Gu et. al.

The paper discusses a novel approach that leverages pre-trained diffusion models for personalized subject swapping in images, allowing users to seamlessly replace subjects while preserving the composition. The approach revolves around swapping and manipulating the UNets attention maps in a training-free manner

### 11. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations, Meng et. al.

11. Barbershop: GAN-based Image Compositing using Segmentation Masks, Zhu et. al.
   - [[`Archive link`](https://arxiv.org/abs/2106.01505)] [[`Github repository`](https://github.com/lllyasviel/ControlNet)] [[`Huggingface blog`](https://huggingface.co/blog/controlnet)] 
   - [[`Very very short summary`](#barbershop)] 
   - [[`Summary notes`](Summaries/Barbershop.md)]

12.   Barbershop: GAN-based Image Compositing using Segmentation Masks, Zhu et. al.
   - [[`Archive link`](https://arxiv.org/abs/2204.11823)] [[`Github repository`](https://github.com/stylegan-human/StyleGAN-Human)] [[`Project page`](https://stylegan-human.github.io/)]
   - [[`Very very short summary`](#stylegan-human)] 
   - [[`Summary notes`](Summaries/StyleGAN-Human.md)]
   



