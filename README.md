# CureCraft_DPO

For the probem setting and the proposed solution, please refer to our slides:
- [Phase 1 presentation](https://docs.google.com/presentation/d/1F37v0YhErD4I0jQaxYGCs2vIpPUFNLYZsfXV1nRePig/edit?usp=sharing)
- [Phase 2 presentation](https://docs.google.com/presentation/d/1c5X76advAqP4fJTnn09mhfICY6A6hYeWd6XgevumBIk/edit?usp=sharing)
- [Phase 3 presentation](https://docs.google.com/presentation/d/19Nnc5rVUEb4nSX9QAg0_deAGQi9OyPhm2GnpTnaWli4/edit?usp=sharing)

A brief overview of our code repository. Explains the components and functionality of each file.

- ddpm.py
  - Diffusion Probabilistic Models (DDPM) for Image Generation
  - Implementation of Diffusion Probabilistic Models (DDPM) for unconditional image generation using PyTorch

- ddpm_conditional.py
  - Conditional Diffusion Probabilistic Models (DDPM) are implemented for image generation using PyTorch.
  - Training involves utilizing a conditional UNet architecture with stochastic gradient descent and AdamW optimizer.
    
- ddpm_dpo.py
  -  implements a conditional version of Diffusion Probabilistic Models (DDPM) for image generation using PyTorch.
  -  The UNet model is trained using a custom loss function based on the Difference of Expected Losses (DPO) algorithm, optimizing for image reconstruction fidelity and latent space stability.
  
- loss_fns.py
  - DPOloss: This loss function implements the Difference of Expected Losses (DPO) algorithm, which aims to optimize the difference between losses computed by the model and a reference model, promoting stability in latent space during training.
  - KTOloss: This loss function stands for Knowledge Transfer Optimization (KTO) loss, which potentially leverages knowledge distillation techniques to transfer knowledge from a reference model to the current model being trained.
    
- modules.py
  - EMA (Exponential Moving Average): A class for maintaining an exponential moving average of model parameters during training, aiding in stable optimization.
  - UNet and UNet_conditional: Neural network architectures for image segmentation tasks, featuring a U-shaped architecture with skip connections, downsampling, upsampling, and self-attention mechanisms, with UNet_conditional also incorporating conditional information through label embeddings. 
