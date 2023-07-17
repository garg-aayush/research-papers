# Research papers
This repository houses my personal summaries and notes on a variety of academic papers/blogs I have read. These summaries are intended to provide a brief overview of the papers' main points, methodologies, findings, and implications, thereby serving as quick references for myself and anyone interested.


## Papers
1. Attention Is All You Need, Vaswani et. al.
   - [[`Archive link`](https://arxiv.org/abs/1706.03762)] [[`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=iDulhoQ2pro&t=2s)] [[`Basic annotated implementation`](http://nlp.seas.harvard.edu/annotated-transformer/)]
   - [[`Very very short summary`](#attention-is-all-you-need)] 
   - [[`Summary notes`](Summaries/Attention_Is_All_You_Need.md)]

2. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy et. al.

3. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, Liu et. al.


4. Denoising Diffusion Probabilistic Models, Ho et. al.
  - [[`Archive link`](https://arxiv.org/abs/2006.11239)] [[`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=W-O7AZNzbzQ)] [[`Basic annotated implementation`](https://nn.labml.ai/diffusion/ddpm/index.html)]
   - [[`Very very short summary`](#ddpm)] 
   - [[`Summary notes`](Summaries/DDPM.md)]


5. Denoising Diffusion Implicit Models, Song et. al.

6. High-Resolution Image Synthesis with Latent Diffusion Models, Rombach. et. al.


7. Adding Conditional Control to Text-to-Image Diffusion Models, Lvmin Zhang and Maneesh Agarwala et. al.
  - [[`Archive link`](https://arxiv.org/abs/2302.05543)] [[`Github repository`](https://github.com/ZPdesu/Barbershop)] [[`Paper explanation video: Original authors`](https://www.youtube.com/watch?v=zk_NhOuAYmo&t=7s)
  - [[`Very very short summary`](#controlnet)] 
  - [[`Summary notes`](Summaries/ControlNet.md)]


7. Null-text Inversion for Editing Real Images using Guided Diffusion Models, Mokady et. al.
   
8. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, Podell et. al.

9. Photoswap: Personalized Subject Swapping in Images, Gu et. al.

10. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations, Meng et. al.

11. Barbershop: GAN-based Image Compositing using Segmentation Masks, Zhu et. al.
   - [[`Archive link`](https://arxiv.org/abs/2106.01505)] [[`Github repository`](https://github.com/lllyasviel/ControlNet)] [[`Huggingface blog`](https://huggingface.co/blog/controlnet)] 
   - [[`Very very short summary`](#barbershop)] 
   - [[`Summary notes`](Summaries/Barbershop.md)]

12.   Barbershop: GAN-based Image Compositing using Segmentation Masks, Zhu et. al.
   - [[`Archive link`](https://arxiv.org/abs/2204.11823)] [[`Github repository`](https://github.com/stylegan-human/StyleGAN-Human)] [[`Project page`](https://stylegan-human.github.io/)]
   - [[`Very very short summary`](#stylegan-human)] 
   - [[`Summary notes`](Summaries/StyleGAN-Human.md)]
   


## Very very short summaries
### Attention Is All You Need
- A Transformer model with only attention mechanism as driving component, no RNN components
- Encoder-decoder kind architecture with resnet-like connections
- Contains MLP for increasing the learning even more complexity
- Uses layer normalization instead of BatchNorm
- Excellent hyperparemeters according to Andrej Karpathy (example, size of feed-forward layer (ffw_size=4) which kinda remain same upto now for many architectures)
- A possible contender for being a general-purpose architecture
- Ability to learn long-range dependencies
- Can be parallelized
- Implements multiple head of attention in parallel

### DDPM
- Lorem ipsum

### ControlNet
- Lorem Ipsum

### Barbershop
- Lorem Ipsum

### StyleGAN-Human
- Lorem Ipsum
