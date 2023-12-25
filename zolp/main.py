from typing import List, Optional, Tuple, Union

from diffusers import AutoPipelineForText2Image
from diffusers.utils import DIFFUSERS_CACHE, logging
from diffusers.utils.torch_utils import randn_tensor
# from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from pipeline_inversable_stable_diffusion import StableDiffusionInvPipeline
from diffusers.schedulers import DDIMInverseScheduler, EulerDiscreteScheduler

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def set_random_seeds(seed=42):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set the seed for CUDA operations (if using GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior when using CuDNN (if using GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AutoPipelineForText2ImageInv(AutoPipelineForText2Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        


    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "use_auth_token": use_auth_token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        # config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)

        # orig_class_name = config["_class_name"]

        # if "controlnet" in kwargs:
        #     orig_class_name = config["_class_name"].replace("Pipeline", "ControlNetPipeline")

        #text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, orig_class_name)
        
        text_2_image_cls = StableDiffusionInvPipeline

        kwargs = {**load_config_kwargs, **kwargs}
        pipe = text_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)
        
        return pipe


class EulerDiscreteInverseScheduler(EulerDiscreteScheduler):
    def __init__(
        self,
        n_sample_steps: int = 1,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
    ):
        
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]
        self.prediction_type = prediction_type
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        # sigmas = sigmas[
        #     self.steps_subset[1 : self.n_sample_steps] + self.steps_subset[-1:]
        # ]
        #timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()

        sigmas = torch.from_numpy(sigmas[::-1].copy()).to(dtype=torch.float32)
        #timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        timesteps = torch.tensor(
            self.steps_subset[1 : self.n_sample_steps] + self.steps_subset[-1:],
            dtype=torch.float32
            )
        timesteps_cleaner = torch.tensor(
            self.steps_subset[0 : self.n_sample_steps],
            dtype=torch.float32
        )

        # setable values
        self.num_inference_steps = None

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if timestep_type == "continuous" and prediction_type == "v_prediction":
            self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas])
        else:
            self.timesteps = timesteps
            self.timesteps_cleaner = timesteps_cleaner

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.is_scale_input_called = False
        self.use_karras_sigmas = use_karras_sigmas

        self._step_index = None    
        self.sigmas = torch.flip(self.sigmas, dims=[0])
        #self.timesteps = torch.flip(self.timesteps, dims=[0])  
        #self.timesteps_cleaner = torch.flip(self.timesteps_cleaner, dims=[0])  

    def set_timesteps(self):
        raise # this should never work
        
    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        timestep_cleaner: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )
        sigma = self.sigmas[int(timestep)]
        sigma_cleaner = self.sigmas[int(timestep_cleaner)]

        assert self.prediction_type == "epsilon"
        prev_sample = sample + (sigma - sigma_cleaner) * model_output
        if not return_dict:
            return (prev_sample,)
        else:
            raise("return_dict should be False")

    def scale_model_input_inv(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep+1).nonzero() # Note, +1 is added to prevent index out of range error

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

def main():

    pipe = AutoPipelineForText2ImageInv.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    
    if False:
        inv_scheduler = DDIMInverseScheduler(
            beta_end = 0.012,
            beta_schedule = 'scaled_linear', #squaredcos_cap_v2
            beta_start = 0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset = 1, #CHECK
            trained_betas = None,        
        )
    else:
        inv_scheduler = EulerDiscreteInverseScheduler(n_sample_steps=4)

    num_inversion_steps = 10
    pipe.set_invscheduler(inv_scheduler, num_inversion_steps)
    
    pipe.to("cuda")

    shape = (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
    
    set_random_seeds(42) # fix the seed for reproducibility
    latents = randn_tensor(shape, dtype=torch.float16)

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    
    output_original, output_latents = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, latents=latents)
    image = output_original.images[0]
    

    #image.show()
    latents_recon = pipe.exact_inversion(
        prompt=prompt,
        guidance_scale=0.0,
        latents=output_latents,
        verbose=True,
    )
    # latents_scaledT = latents * pipe.scheduler.init_noise_sigma.cpu()
    print(f"TOT norm : {((latents - (latents_recon/pipe.scheduler.init_noise_sigma).cpu()).norm() / latents.norm()).item()}" )

    latents_image = pipe.vae.decode(latents.cuda() / pipe.vae.config.scaling_factor)[0]
    latents_image = pipe.image_processor.postprocess(latents_image.detach(), output_type='pil', do_denormalize=[True])[0]
    #latents_image.show()

    recon_image = pipe.vae.decode(latents_recon.clone() / pipe.vae.config.scaling_factor)[0]
    recon_image = pipe.image_processor.postprocess(recon_image.detach(), output_type='pil', do_denormalize=[True])[0]
    #recon_image.show()

    output2_original, output2_latents = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, latents=latents_recon)
    image2 = output2_original.images[0]
    #image2.show()

    fig, axes = plt.subplots(2, 2)

    # Display each image in a subplot
    axes[0, 1].imshow(image)
    axes[0, 0].axis("off")
    axes[0, 0].imshow(latents_image)
    axes[0, 1].axis("off")
    axes[1, 0].imshow(recon_image)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(image2)
    axes[1, 1].axis("off")
    # Ensure a tight layout
    plt.tight_layout()

    # Show the plot
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_directory = "/home/icl2/khlee/generative-models/zolp/images/"
    plt.show()
    #plt.savefig(os.path.join(save_directory, f"plot_{timestamp}.png"))


if __name__ == "__main__":
    main()