#Rapidata/Animals-10 has animals 

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.saving import save_state_dict
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from data_helpers import AnimalData
from torch.optim.lr_scheduler import LambdaLR

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
from image_utils import concat_images_horizontally
from torcheval.metrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="scale")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--repo_id",type=str,default="jlbaker361/model",help="name on hf")
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
parser.add_argument("--val_interval",type=int,default=10)
parser.add_argument("--warmup_steps",type=int,default=2000)

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
                api.create_repo(args.repo_id,exist_ok=True)
            except HfHubHTTPError:
                print("hf hub error!")
                time.sleep(random.randint(5,120))
                accelerator.init_trackers(project_name=args.project_name,config=vars(args))

                api=HfApi()
                api.create_repo(args.repo_id,exist_ok=True)

        


        torch_dtype={
            "no":torch.float32,
            "fp16":torch.float16,
            "bf16":torch.bfloat16
        }[args.mixed_precision]

        pipe=DiffusionPipeline.from_pretrained("Lykon/DreamShaper")
        unet_config=pipe.unet.config
        unet=UNet2DConditionModel.from_config(unet_config).to(device) #,torch_dtype)
        if args.training_type=="scale":
            unet.conv_in=torch.nn.Conv2d(3,
                                         unet.conv_in.out_channels,
                                         unet.conv_in.kernel_size,
                                         unet.conv_in.stride,
                                         unet.conv_in.padding)
            unet.conv_out=torch.nn.Conv2d(unet.conv_out.in_channels,
                                         3,
                                         unet.conv_out.kernel_size,
                                         unet.conv_out.stride,
                                         unet.conv_out.padding)
        unet.requires_grad_(True)
        text_encoder=pipe.text_encoder.to(device) #,torch_dtype)
        tokenizer=pipe.tokenizer
        vae=pipe.vae.to(device) #,torch_dtype)
        scheduler=pipe.scheduler
        step=scheduler.config.num_train_timesteps//(args.power2_dim)
        scale_noise_steps=[int(x*step) for x in range(args.power2_dim+1)][:-1]+[1000]
        scale_steps=[]
        s=1
        for _ in scale_noise_steps:
            scale_steps.append(s)
            s=s*2
        scale_steps=scale_steps[::-1]
        accelerator.print("noise_steps",scale_noise_steps)
        accelerator.print("scale steps",scale_steps)
        image_processor=pipe.image_processor
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        
        params=[p for p in unet.parameters()]
        
        optimizer=torch.optim.AdamW(params,args.lr)
        
        def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
            def lr_lambda(step):
                # warmup
                if step < warmup_steps:
                    return step / warmup_steps
                
                # cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

            return LambdaLR(optimizer, lr_lambda)

        
        #dataset=??????
        
        
        dataset=AnimalData(image_processor,tokenizer,dim=2**args.power2_dim)
        accelerator.print("image size ",2**args.power2_dim)

        test_size=int(len(dataset)//10)
        train_size=int(len(dataset)-(test_size*2))

        
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(42)

        # Split the dataset
        train_dataset, test_dataset,val_dataset = random_split(dataset, [train_size, test_size,test_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader=DataLoader(val_dataset,batch_size=1,shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        total_steps=len(train_loader)*args.epochs
        
        lr_scheduler=cosine_schedule_with_warmup(optimizer,args.warmup_steps,total_steps)
        
        train_loader,test_loader,val_loader,optimizer,unet,vae,text_encoder,lr_scheduler=accelerator.prepare(train_loader,test_loader,val_loader,optimizer,unet,vae,text_encoder,lr_scheduler)

        save_subdir=os.path.join(args.save_dir,args.repo_id)
        os.makedirs(save_subdir,exist_ok=True)

        WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
        CONFIG_NAME="config.json"
        
        save_path=os.path.join(save_subdir,WEIGHTS_NAME)
        config_path=os.path.join(save_subdir,CONFIG_NAME)

        start_epoch=1
        try:
            if args.load_hf:
                pretrained_weights_path=api.hf_hub_download(args.repo_id,WEIGHTS_NAME,force_download=True)
                pretrained_config_path=api.hf_hub_download(args.repo_id,CONFIG_NAME,force_download=True)
                unet.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
                with open(pretrained_config_path,"r") as f:
                    data=json.load(f)
                start_epoch=data["start_epoch"]+1
        except Exception as e:
            accelerator.print(e)
            
        def inference(d_loader,label):
            fid=FrechetInceptionDistance()
            real=[]
            fake=[]
            for b,batch in enumerate(d_loader):
                
                if b==args.limit:
                    break
                text=batch["text"]['input_ids'].to(device)#,dtype=torch_dtype)
                text_str=batch["text_str"]
                intermediate_list=[]
                images=batch["image"]
                real.append(images)
                if b==0:
                    
                    latents = vae.encode(images).latent_dist.sample()
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
                        
                        intermediate_list.append(noise)
                        
                if args.training_type=="scale" or args.training_type=="scale_vae":
                    #if args.training_type=="scale":
                    noise=torch.ones_like(images)* random.uniform(-1,1)
                    if args.training_type=="scale_vae":
                        noise=vae.encode(noise).latent_dist.sample()
                        noise=noise * vae.config.scaling_factor
                    
                    for i,t in enumerate(timesteps):
                        if i==0 and b==0:
                            accelerator.print("noise",noise.size())
                            accelerator.print("text_str",text_str)
                        
                        noise=forward_with_metadata(unet,noise, t, encoder_hidden_states, metadata=None,return_dict=False)[0]
                        
                        intermediate_list.append(noise)
                        
                if args.training_type=="scale_vae" or args.training_type=="noise":    
                    image = vae.decode(noise / vae.config.scaling_factor, return_dict=False)[0]
                else:
                    image=noise
                fake.append(image)
                image=image_processor.postprocess(image.detach().cpu(),"pil",[True]*image.size()[0])
                for n,i in enumerate(image):
                    accelerator.log({
                        f"{label}_{text_str[n]}":wandb.Image(i)
                    })
                    '''accelerator.log({
                        f"{label}_{b}":wandb.Image(i)
                    })'''
                    concat_images=torch.stack([intermediate_list[i][n] for i,t in enumerate(timesteps)])
                    #accelerator.print("concat ",concat_images.size())
                    if args.training_type=="scale_vae" or args.training_type=="noise":
                        concat_images=torch.cat([vae.decode(img.unsqueeze(0)).sample/vae.config.scaling_factor for img in concat_images])
                        #accelerator.print("concat decoded",concat_images.size())
                    concat_images=image_processor.postprocess(concat_images.detach().cpu(),"pil",[True]*concat_images.size()[0])
                    
                    
                    '''accelerator.log({
                        f"concat_{label}_{b}":wandb.Image(concat_images_horizontally(concat_images))
                    })'''
            real=image_processor.denormalize(torch.cat(real))
            fake=image_processor.denormalize(torch.cat(fake))
            fid.update(real,True)
            fid.update(fake,False)
            fid_score = fid.compute().item()
            accelerator.log({
                f"{label}_fid":fid_score
            })
                
            
        if args.training_type=="scale_noise":
            set_metadata_embedding(unet,1)

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
                    
                    metadata=None
                    if args.training_type=="noise":
                        latents = vae.encode(images).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()
                        # Sample a random timestep for each image
                        unet_input = scheduler.add_noise(latents, noise, timesteps)
                        target=noise
                        
                    elif args.training_type=="scale" or args.training_type=="scale_vae":
                        
                        scales=[random.randint(0,len(scale_noise_steps)-1) for _ in range(bsz)]
                        if e==start_epoch and b==0:
                            accelerator.print("scales",scales)
                            accelerator.print("scale noise steps",scale_noise_steps)
                            accelerator.print("scale nosie steps for s in scales",[scale_noise_steps[s] for s in scales])
                        try:
                            timesteps=torch.tensor([scale_noise_steps[s] for s in scales]).long().to(device=device)
                        except IndexError:
                            accelerator.print(scales)
                            accelerator.print(scale_noise_steps)
                            raise IndexError()
                        
                        input_list=[]
                        target_list=[]
                        for img,scale in zip(images,scales):
                            img=img.unsqueeze(0)
                            size=img.size()[-1]
                            initial_size=size
                            size=scale_steps[scale]
                            try:
                                input_img=F.interpolate(img,[size,size])
                                input_img=F.interpolate(input_img,[initial_size,initial_size])
                                target_img=F.interpolate(img,[2*size,2*size])
                                target_img=F.interpolate(target_img,[initial_size,initial_size])
                            except RuntimeError:
                                accelerator.print("runtime error, scale:", scale,"size", size,"img size",initial_size)
                                raise RuntimeError()
                            if args.training_type=="scale_vae":
                                input_img=vae.encode(input_img).latent_dist.sample()
                                input_img=input_img * vae.config.scaling_factor
                                target_img=vae.encode(target_img).latent_dist.sample()
                                target_img=target_img * vae.config.scaling_factor
                            input_list.append(input_img)
                            target_list.append(target_img)
                            
                        unet_input=torch.concat(input_list).to(device=device)
                        target=torch.concat(target_list).to(device=device)
                        
                    elif args.training_type=="scale_noise":
                        scales=[random.randint(0,len(scale_noise_steps))]*bsz
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        
                        input_list=[]
                        target_list=[]
                        for img,scale in zip(images,scales):
                            img=img.unsqueeze(0)
                            size=img.size()[-1]
                            initial_size=size
                            size=scale_steps[scale]
                            try:
                                input_img=F.interpolate(img,[size,size])
                                input_img=F.interpolate(input_img,[initial_size,initial_size])
                                target_img=F.interpolate(img,[2*size,2*size])
                                target_img=F.interpolate(target_img,[initial_size,initial_size])
                            except RuntimeError:
                                accelerator.print("runtime error, scale:", scale,"size", size,"img size",initial_size)
                                raise RuntimeError()
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
                        accelerator.print("unet_input",unet_input.device,unet_input.dtype,unet_input.size())
                        accelerator.print("timesteps ",timesteps.device, timesteps.dtype,timesteps.size())
                        accelerator.print("encoder", encoder_hidden_states.dtype,encoder_hidden_states.device,encoder_hidden_states.size())
                    #with accelerator.autocast():
                    model_pred=forward_with_metadata(unet,unet_input, timesteps, encoder_hidden_states, metadata=metadata,return_dict=False)[0]

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                    train_loss += avg_loss.cpu().detach().item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    loss_buffer.append(loss.cpu().detach().numpy())

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss})
                    train_loss = 0.0


            end=time.time()
            accelerator.print(f"epoch {e} elapsed {end-start} avg loss {np.mean(loss_buffer)}")
            accelerator.log({
                "avg_loss":np.mean(loss_buffer)
            })
            save_state_dict(unet.state_dict(),e,save_path,config_path,repo_id=args.repo_id,api=api,accelerator=accelerator)
            if e%args.val_interval==0:
                with torch.no_grad():
                    inference(val_loader,"val")
        #inference
        with torch.no_grad():
            inference(test_loader,"test")
                
                
            

            
            
            

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