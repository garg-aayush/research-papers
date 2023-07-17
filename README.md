# Research papers
This repository houses my personal summaries and notes on a variety of academic papers/blogs I have read. These summaries are intended to provide a brief overview of the papers' main points, methodologies, findings, and implications, thereby serving as quick references for myself and anyone interested.


## Papers
1. Attention Is All You Need, Vaswani et. al.
   - [[`Archive link`](https://arxiv.org/abs/1706.03762)] [[`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=iDulhoQ2pro&t=2s)] [[`Basic annotated implementation`](http://nlp.seas.harvard.edu/annotated-transformer/)]
   - [[`Very very short summary`](#attention-is-all-you-need)] 
   - [[`Summary and notes`](#fmfmf)]



### T1
- [ ] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [ ] [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [ ] [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)
- [ ] [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [ ] [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [ ] [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)
- [ ] [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)
- [ ] [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [ ] [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- [ ] [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142)
- [ ] [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/abs/2301.00704)
- [ ] [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)
- [ ] [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392)
- [ ] [eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/abs/2211.01324)

### T1
- [ ] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [ ] [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [ ] [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)
- [ ] [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [ ] [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [ ] [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)
- [ ] [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)
- [ ] [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [ ] [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- [ ] [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142)
- [ ] [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/abs/2301.00704)
- [ ] [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)
- [ ] [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392)
- [ ] [eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/abs/2211.01324)

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
