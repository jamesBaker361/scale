#Rapidata/Animals-10 has animals 

import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from data_helpers import AnimalData

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
from accelerate.utils import set_seed
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from torch.utils.data import random_split, DataLoader
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from unet_helpers import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--epochs",type=int,default=2)
parser.add_argument("--limit",type=int,default=10)
parser.add_argument("--save_dir",type=str,default="weights")
parser.add_argument("--training_type",type=str,default="noise",help="scale or noise or scale_noise")
parser.add_argument("--power2_dim",type=int,default=7,help="power of 2 for image dimension")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--load_hf",action="store_true")
parser.add_argument("--n_test",type=int,default=4)
parser.add_argument("--num_inference_steps",type=int,default=10)

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    with accelerator.autocast():
        set_seed(123)
        device=accelerator.device
        state = PartialState()
        print(f"Rank {state.process_index} initialized successfully")
        if accelerator.is_main_process or state.num_processes==1:
            accelerator.print(f"main process = {state.process_index}")
        if accelerator.is_main_process or state.num_processes==1:
            try:
                accelerator.init_trackers(project_name=args.project_name,config=vars(args))

                api=HfApi()
                api.create_repo(args.name,exist_ok=True)
            except HfHubHTTPError:
                print("hf hub error!")
                time.sleep(random.randint(5,120))
                accelerator.init_trackers(project_name=args.project_name,config=vars(args))

                api=HfApi()
                api.create_repo(args.name,exist_ok=True)

        


        torch_dtype={
            "no":torch.float32,
            "fp16":torch.float16,
            "bf16":torch.bfloat16
        }[args.mixed_precision]

        pipe=DiffusionPipeline.from_pretrained("Lykon/DreamShaper")
        unet_config=pipe.unet.config
        unet=UNet2DConditionModel.from_config(unet_config).to(device) #,torch_dtype)
        unet.requires_grad_(True)
        text_encoder=pipe.text_encoder.to(device) #,torch_dtype)
        tokenizer=pipe.tokenizer
        vae=pipe.vae.to(device) #,torch_dtype)
        scheduler=pipe.scheduler
        step=scheduler.config.num_train_timesteps//args.power2_dim
        scale_noise_steps=[int(x*step) for x in range(args.power2_dim+1)]
        accelerator.print("noise_steps",scale_noise_steps)
        image_processor=pipe.image_processor
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        
        params=[p for p in unet.parameters()]
        
        optimizer=torch.optim.AdamW(params,args.lr)

        #dataset=??????
        
        
        dataset=AnimalData(image_processor,tokenizer,dim=2**args.power2_dim)
        accelerator.print("image size ",2**args.power2_dim)

        test_size=len(dataset)//10
        train_size=len(dataset)-test_size

        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(42)

        # Split the dataset
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        train_loader,test_loader,optimizer,unet,vae,text_encoder=accelerator.prepare(train_loader,test_loader,optimizer,unet,vae,text_encoder)

        save_subdir=os.path.join(args.save_dir,args.name)
        os.makedirs(save_subdir,exist_ok=True)

        WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
        CONFIG_NAME="config.json"
        
        save_path=os.path.join(save_subdir,WEIGHTS_NAME)
        config_path=os.path.join(save_subdir,CONFIG_NAME)

        start_epoch=1
        try:
            if args.load_hf:
                pretrained_weights_path=api.hf_hub_download(args.name,WEIGHTS_NAME,force_download=True)
                pretrained_config_path=api.hf_hub_download(args.name,CONFIG_NAME,force_download=True)
                unet.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
                with open(pretrained_config_path,"r") as f:
                    data=json.load(f)
                start_epoch=data["start_epoch"]+1
        except Exception as e:
            accelerator.print(e)
            
        if args.training_type=="scale_noise":
            set_metadata_embedding(unet,1)


        def save(e:int,state_dict):
            #state_dict=???
            print("state dict len",len(state_dict))
            torch.save(state_dict,save_path)
            with open(config_path,"w+") as config_file:
                data={"start_epoch":e}
                json.dump(data,config_file, indent=4)
                pad = " " * 2048  # ~1KB of padding
                config_file.write(pad)
            print(f"saved {save_path}")
            try:
                api.upload_file(path_or_fileobj=save_path,
                                        path_in_repo=WEIGHTS_NAME,
                                        repo_id=args.name)
                api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                        repo_id=args.name)
                print(f"uploaded {args.name} to hub")
            except Exception as e:
                accelerator.print("failed to upload")
                accelerator.print(e)

        for e in range(start_epoch,args.epochs+1):
            start=time.time()
            loss_buffer=[]
            train_loss=0.0
            for b,batch in enumerate(train_loader):
                if b==args.limit:
                    break

                with accelerator.accumulate(params):
                    
                    images=batch["image"].to(device) #,dtype=torch_dtype)
                    text=batch["text"]['input_ids'].to(device)#,dtype=torch_dtype)
                    encoder_hidden_states = text_encoder(text, return_dict=False)[0] #.to(dtype=torch_dtype)
                    
                    
                    '''if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)'''
                    
                    if e==start_epoch and b==0:
                        accelerator.print("images",images.size(),images.shape)
                    bsz = images.shape[0]
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    metadata=None
                    if args.training_type=="noise":
                        

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()
                        # Sample a random timestep for each image
                        unet_input = scheduler.add_noise(latents, noise, timesteps)
                        target=noise
                        
                    elif args.training_type=="scale":
                        
                        scales=[random.randint(0,len(scale_noise_steps))]*bsz
                        if e==start_epoch and b==0:
                            accelerator.print("scales",scales)
                            accelerator.print("scale noise steps",scale_noise_steps)
                            accelerator.print("scale nosie steps for s in scales",[scale_noise_steps[s] for s in scales])
                        timesteps=torch.tensor([scale_noise_steps[s] for s in scales]).long().to(device=device)
                        
                        input_list=[]
                        target_list=[]
                        for img,scale in zip(latents,scales):
                            img=img.unsqueeze(0)
                            size=img.size()[-1]
                            initial_size=size
                            for step in range(0,scale):
                                size=size//2
                            input_img=F.interpolate(img,[size,size])
                            input_img=F.interpolate(input_img,[initial_size,initial_size])
                            target_img=F.interpolate(img,[2*size,2*size])
                            target_img=F.interpolate(target_img,[initial_size,initial_size])
                            input_list.append(input_img)
                            target_list.append(target_img)
                            
                        unet_input=torch.concat(input_list).to(device=device)
                        target=torch.concat(target_list).to(device=device)
                        
                    elif args.training_type=="scale_noise":
                        scales=[random.randint(0,len(scale_noise_steps))]*bsz
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        
                        input_list=[]
                        target_list=[]
                        for img,scale in zip(latents,scales):
                            img=img.unsqueeze(0)
                            size=img.size()[-1]
                            initial_size=size
                            for step in range(0,scale):
                                size=size//2
                            input_img=F.interpolate(img,[size,size])
                            input_img=F.interpolate(input_img,[initial_size,initial_size])
                            target_img=F.interpolate(img,[2*size,2*size])
                            target_img=F.interpolate(target_img,[initial_size,initial_size])
                            input_list.append(input_img)
                            target_list.append(target_img)
                            
                        unet_input=torch.concat(input_list).to(device=device)
                        noise=torch.concat(target_list).to(device=device)
                        
                        unet_input=scheduler.add_noise(latents, noise, timesteps)
                        target=noise
                        
                        metadata=prepare_metadata(scales)
                        
                        
                    # Predict the noise residual and compute loss
                    #model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0]
                    if b==0 and e==start_epoch:
                        accelerator.print("unet_input",unet_input.device,unet_input.dtype)
                        accelerator.print("timesteps ",timesteps.device, timesteps.dtype)
                        accelerator.print("encoder", encoder_hidden_states.dtype,encoder_hidden_states.device)
                    #with accelerator.autocast():
                    model_pred=forward_with_metadata(unet,unet_input, timesteps, encoder_hidden_states, metadata=metadata,return_dict=False)[0]

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                    train_loss += avg_loss.cpu().detach().item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    '''if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)'''
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss})
                    train_loss = 0.0


            end=time.time()
            accelerator.print(f"epoch {e} elapsed {end-start}")
        #inference
        for b,batch in enumerate(test_loader):
            text=batch["text"]['input_ids'].to(device)#,dtype=torch_dtype)
            encoder_hidden_states = text_encoder(text, return_dict=False)[0] #.to(dtype=torch_dtype)
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler, args.num_inference_steps, device
            )
            if args.training_type=="noise":
            
                noise = torch.randn_like(latents)
                
                for i,t in enumerate(timesteps):
                    noise=scheduler.scale_model_input(noise, t)
                    
                    model_pred=forward_with_metadata(unet,noise, t, encoder_hidden_states, metadata=None,return_dict=False)[0]
                    
                    noise=scheduler.step(model_pred,t,noise,return_dict=False)[0]
                    
            image = vae.decode(noise / vae.config.scaling_factor, return_dict=False)[0]
            image=image_processor.postprocess(image,"pil",[True])[0]
            accelerator.log({
                f"test_{b}":wandb.Image(image)
            })
                
                
            

            
            
            

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")