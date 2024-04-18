import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
from diffusers import AutoencoderKL


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_latents(self, x, t, noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    # setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)

    # Add a vae encoder to get into latent space
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    vae = AutoencoderKL.from_single_file(url)
    
    
    # Sample noise that we'll add to the latents
    

    # if args.input_perturbation: # haven't tried yet
    #     new_noise = noise + args.input_perturbation * torch.randn_like(noise)



    model = UNet_conditional(num_classes=args.num_classes).to(device)
    ref_model = UNet_conditional(num_classes=args.num_classes).to(device)


    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # mse = nn.MSELoss()
    # diffusion = Diffusion(img_size=args.image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    # l = len(dataloader)
    # ema = EMA(0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)

            timesteps = diffusion.sample_timesteps(images.shape[0]).to(device)

            # if args.train_method == 'dpo': # make timesteps and noise same for pairs in DPO
            #     timesteps = timesteps.chunk(2)[0].repeat(2)
            #     noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

            noisy_latents = diffusion.noise_latents(latents, timesteps, noise)
            
            if np.random.random() < 0.1:
                labels = None
            # Need to modify unet
            predicted_noise = model(noisy_latents, timesteps, labels)

            model_losses = (predicted_noise - noise).pow(2).mean()
            # Modify for DPO
            # model_losses_w, model_losses_l = model_losses.chunk(2)

            # raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    
            # model_diff = model_losses_w - model_losses_l

            # with torch.no_grad(): # Get the reference policy (unet) prediction
            #     ref_pred = ref_model(noisy_latents, timesteps, labels).sample.detach()
            #     ref_losses = (predicted_noise - noise).pow(2).mean()
            #     ref_losses_w, ref_losses_l = ref_losses.chunk(2)
            #     ref_diff = ref_losses_w - ref_losses_l
            #     raw_ref_loss = ref_losses.mean()    
                        
            scale_term = -0.5 * args.beta_dpo
            inside_term = scale_term * (model_diff - ref_diff)
            implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
            loss = -1 * F.logsigmoid(inside_term).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("DPO loss", loss.item(), global_step=epoch * l + i)

        # if epoch % 10 == 0:
        #     labels = torch.arange(10).long().to(device)
        #     sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        #     ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        #     plot_images(sampled_images)
        #     save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #     save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
        #     torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        #     torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        #     torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 14
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
    args.device = "cpu"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    # launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    vae = AutoencoderKL.from_single_file(url)
    
    image = torch.randn((3, 512, 512))
    latents = vae.encode(image).latent_dist.sample()
    print(latents.shape)
