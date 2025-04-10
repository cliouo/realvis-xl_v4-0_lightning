import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = '1'
from huggingface_hub import snapshot_download
import torch
import base64
import random
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler

class InferlessPythonModel:
    def initialize(self):
        repo_id = "SG161222/RealVisXL_V4.0_Lightning"
        snapshot_download(repo_id=repo_id,allow_patterns=["*.safetensors"])
        self.pipe_fast = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0_Lightning", 
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        self.pipe_fast.scheduler = DPMSolverSinglestepScheduler.from_config(
            self.pipe_fast.scheduler.config,
            use_karras_sigmas=True
        )
        self.pipe_fast.to("cuda")
        
    def image_to_base64(self, image):
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        return img_str
    
    def infer(self, inputs):
        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, ugly, disgusting, blurry, amputation,(face asymmetry, eyes asymmetry, deformed eyes, open mouth)")
        steps = inputs.get("steps", 5)
        randomize_seed = inputs.get("randomize_seed", True)
        seed = inputs.get("seed", 2404)
        width = inputs.get("width", 1024)
        height = inputs.get("height", 1024)
        guidance_scale = inputs.get("guidance_scale", 1.0)
                
        image = self.pipe_fast(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
        output_image_base64 = self.image_to_base64(image)
        
        return {
            "generated_image_base64": output_image_base64,
        }
    
    def finalize(self):
        self.pipe_fast = None
