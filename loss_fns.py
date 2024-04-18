import torch
import torch.nn.functional as F



def DPOloss(beta, diffusion, timesteps, labels, model, ref_model , noisy_latents, noise):
    predicted_noise = model(noisy_latents, timesteps, labels)

    model_losses = (predicted_noise - noise).pow(2).mean()

    model_losses_w, model_losses_l = model_losses.chunk(2)

    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
            
    model_diff = model_losses_w - model_losses_l

    with torch.no_grad(): # Get the reference policy (unet) prediction
        ref_pred = ref_model(noisy_latents, timesteps, labels).sample.detach()
        ref_losses = (predicted_noise - noise).pow(2).mean()
        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
        ref_diff = ref_losses_w - ref_losses_l
        raw_ref_loss = ref_losses.mean()    
                
    scale_term = -0.5 * beta
    inside_term = scale_term * (model_diff - ref_diff)
    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    loss = -1 * F.logsigmoid(inside_term).mean()

    return loss


def KTOloss(beta, diffusion, timesteps, labels, model, ref_model , noisy_latents, noise):
    predicted_noise = model(noisy_latents, timesteps, labels)
    