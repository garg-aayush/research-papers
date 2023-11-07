# Research papers
This repository houses my personal summaries and notes on a variety of academic papers/blogs I have read. These summaries are intended to provide a brief overview of the papers' main points, methodologies, findings, and implications, thereby serving as quick references for myself and anyone interested.


## Diffusion Papers
### 1. Denoising Diffusion Probabilistic Models, Ho et. al.
- Introduces a generative modeling using a continuous-time diffusion process, offering an alternative to adversarial and maximum likelihood methods
- Produces image samples of quality comparable or superior to leading GANs and VAEs
- Provides a theoretical foundation for diffusion models, linking them to other generative techniques
    | [**`Summary notes`**](Summaries/Diffusion/DDPM.md) |  [`Paper explanation video: Yanic Kilcher`](https://www.youtube.com/watch?v=W-O7AZNzbzQ) |
    |---|---|
    |  [**`Archive link`**](https://arxiv.org/abs/2006.11239) |  [**`Basic annotated implementation`**](https://nn.labml.ai/diffusion/ddpm/index.html) |


### 2. Denoising Diffusion Implicit Models, Song et. al.
- Present DDIMS which are implicit probabilistic models and can produce high quality samples **10X** to **50X** faster (in about 50 steps) in comparison to DDPM
- Generalizes DDPMs by using a class of non-Markovian diffusion process that lead to "short" generative Markov chains that can simulate image generation in a small number of steps
- The training objective in DDIM is similar to DDPM, one can use any pretrained DDPM model with DDIM or other generative processes that can generative images in least steps
    | [**`Summary notes`**](Summaries/Diffusion/DDIM.md) |  [`Archive link`](https://arxiv.org/abs/2010.02502) |  [`Github repo`](https://github.com/ermongroup/ddim) |
    |---|---|---|


### 3. Prompt-to-Prompt Image Editing with Cross Attention Control, Hertz et. al.
- Introduces a textual editing method to semantically edit images in pre-trained text-conditioned diffusion models via Prompt-to-Prompt manipulations
- Approach allows for editing the image while preserving the original composition of the image and addressing the content of the new prompt.  
- The key idea is that onr can edit images by injecting the cross-attention maps during the diffusion process, controlling which pixels attend to which tokens of the prompt text during which diffusion steps. 
    | [**`Summary notes`**](Summaries/Diffusion/Prompt-to-prompt.md) |  [`Archive link`](https://arxiv.org/abs/2208.01626) | [`Github repo`](https://github.com/google/prompt-to-prompt/) |
    |---|---|---|


### 4. Null-text Inversion for Editing Real Images using Guided Diffusion Models, Mokady et. al.
- Introduces an accurate inversion scheme for **real input images**,  enabling intuitive and versatile text-based image modification without tuning model weights.
- It achieving near-perfect reconstruction, while retaining the rich text-guided editing capabilities of the original model
- The approach consists of two novel ideas, pivotal inversion (using DDIM inversion trajactory as the anchor noise vector) and null-text optimization (optimizing only the null-text embeddings)
    | [**`Summary notes`**](Summaries/Diffusion/Null-TextInversion.md) |  [`Archive link`](https://arxiv.org/abs/2211.09794) |
    |---|---|
    | [**`Paper walkthrough video: Original author`**](https://www.youtube.com/watch?v=qzTlzrMWU2M&t=52s) | [**`Github repo`**](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)  |


### 5. Adding Conditional Control to Text-to-Image Diffusion Models, Lvmin Zhang and Maneesh Agarwala et. al.
- Allows additional control for the pre-trained large diffusion models, such as Stable diffusion, by providing the facility of input visual conditions such as edge maps, segment masks, depth masks, etc.
- Learns task-specific conditions in an end-to-end way
- Training is as fast as fine-tuning a diffusion model, and for small dataset (<50k), it can be trained to produce robust results even on desktop-grade personal GPUs.
- Multiple controlnets can be combinded at inference time to have multiple control visual conditions
    | [**`Summary notes`**](Summaries/Diffusion/ControlNet.md) |  [`Archive link`](https://arxiv.org/abs/2302.05543) | [`Github repo`](https://github.com/lllyasviel/ControlNet)|
    |---|---|---|
    |  [**`HF usage example`**](https://huggingface.co/blog/controlnet) |[**`Controlnet SD1.5 1.0 and 1.1 ckpts`**](https://huggingface.co/lllyasviel) |  [**`Controlnet SDXL ckpts`**](https://huggingface.co/models?other=stable-diffusion-xl&other=controlnet) |



### 6. DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion, Karras et. al.
-  An image-and-pose conditioned diffusion method based upon Stable Diffusion to turn fashion photographs into realistic, animated videos
-  Introduces a pose conditioning approach that greatly improves temporal consistency across frames
-  Uses an image CLIP and VAE encoder, instead of text encoder, that increases the output fidelity to the conditioning image
    | [**`Summary notes`**](Summaries/Diffusion/DreamPose.md) |  [`Archive link`](https://arxiv.org/abs/2304.06025) | [`Github repo`](https://github.com/johannakarras/DreamPose)|
    |---|---|---|


### 7. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, Podell et. al.
- Introduces an enhanced stable diffusion model that surpasses the generating capabilities of previous versions
- Uses a larger UNet backbone and introducing novel conditioning schemes in the training stage
- Probably, the best public domain open-source text-to-image model at this moment (Aug, 2023)
    | [**`Summary notes`**](Summaries/Diffusion/SDXL.md) |  [`Archive link`](https://arxiv.org/abs/2307.01952) | 
    |---|---|
    | [**`Paper walkthrough video: Two minute papers`**](https://www.youtube.com/watch?v=kkYaikeLJdc) | [**`HF usage example`**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)  |
    
### 8. ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models, He et. al.
- Directly sampling an image with a resolution beyond the training image sizes of pre-trained diffusion models models usually result in severe object repetition issues and unreasonable object structures.
- The paper explores the use of pre-trained diffusion models to generate images at resolutions higher than the models were trained on, specifically targeting the generation of images with arbitrary aspect ratios and higher resolution.
- Probably, the best public domain open-source text-to-image model at this moment (Aug, 2023)
    | [**`Summary notes`**](Summaries/Diffusion/ScaleCrafter.md) |  [`Archive link`](https://arxiv.org/abs/2310.07702) | 
    |---|---|
    | [**`Project page`**](https://yingqinghe.github.io/scalecrafter/) | [**`Github repo`**](https://github.com/YingqingHe/ScaleCrafter)  |

## Transformers Papers
### 1. Attention Is All You Need, Vaswani et. al.
- Introduces the `Transformer` model, which relies solely on attention mechanisms for sequence modelling and transduction tasks. It dispenses the recurrence and convolutions networks entirely.
- It is a breakthrough paper that has lead to major advances in NLP, CV and multi-modal machine learning
  | [**`Summary notes`**](Summaries/Transformers/Attention.md) |  [`Archive link`](https://arxiv.org/abs/1706.03762) |
    |---|---|
    | [**`Paper explanation video: Yanic Kilcher`**](https://www.youtube.com/watch?v=iDulhoQ2pro&t=3s) | [**`Annotated Implementation`**](http://nlp.seas.harvard.edu/annotated-transformer/) |
    
## GANs Papers
### 1. Barbershop: GAN-based Image Compositing using Segmentation Masks
- Proposes a novel solution to image blending, particularly for the problem of hairstyle transfer, based on GAN-inversion
- introduces a latent space for image blending which is better at preserving detail and encoding spatial information
- explains a new GAN-embedding algorithm which is able to slightly modify images to conform to a common segmentation mask
    | [**`Summary notes`**](Summaries/GANs/Barbershop.md) |  [`Archive link`](https://arxiv.org/abs/2106.01505) | [`Github repo`](https://github.com/ZPdesu/Barbershop)
    |---|---|---|