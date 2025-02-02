from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMInverseScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class StableDiffusionInvPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super(StableDiffusionInvPipeline, self).__init__(
                vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                image_encoder,
                requires_safety_checker)
    
    def set_invscheduler(self, scheduler, num_inference_steps):
        
        self.inv_scheduler = scheduler
        self.inv_scheduler_type = type(self.inv_scheduler).__name__ 
        self.num_inference_steps = num_inference_steps
        if self.inv_scheduler_type=='DDIMInverseScheduler':
            self.inv_scheduler.set_timesteps(num_inference_steps)

        sigmas = np.array(((1 - self.inv_scheduler.alphas_cumprod) / self.inv_scheduler.alphas_cumprod) ** 0.5)
        sigmas = torch.from_numpy(sigmas[::1].copy()).to(dtype=torch.float32)
        
        self.sigmas = torch.cat([torch.zeros(1, device=sigmas.device),sigmas])

        
        # DDIMInverseScheduler

    # This function is the same to the original but has additional output latents
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
        """           
        # output_original = super(StableDiffusionInvPipeline, self).__call__(
        #     prompt,
        #     height,
        #     width,
        #     num_inference_steps,
        #     timesteps,
        #     guidance_scale,
        #     negative_prompt,
        #     num_images_per_prompt,
        #     eta,
        #     generator,
        #     latents,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     ip_adapter_image,
        #     output_type,
        #     return_dict,
        #     cross_attention_kwargs,
        #     guidance_rescale,
        #     clip_skip,
        #     callback_on_step_end,
        #     callback_on_step_end_tensor_inputs,
        #     **kwargs,
        # )  
        output_for_inv = (latents)
 
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator, # is None
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                print(f"generation : {t}, {timestep_cond}")

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        output_original = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
       
        return output_original, latents
    
    # The main contribution 1: exact inversion
    #@torch.inference_mode()
    def exact_inversion(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inversion_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: torch.FloatTensor = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        verbose: bool = False,
        **kwargs,
    ):
        # the inputs are the same as StableDiffusionPipeline.__call__()

        # in our code, we use only prompt, num_inference_steps=1, guidance_scale=0.0.
        with torch.no_grad():
            assert guidance_scale==0.0
            assert self.do_classifier_free_guidance == False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )        

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None:
                image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
                if self.do_classifier_free_guidance:
                    image_embeds = torch.cat([negative_image_embeds, image_embeds])

            # 4. Prepare timesteps
            #timesteps, num_inversion_steps = retrieve_timesteps(self.scheduler, num_inversion_steps, device, timesteps)

            # 5. Prepare latent variables: please don't do this

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. inv-Denoising loop
            final_latents = latents
            # 7-case A. Naive DDIM inversion 
            if self.inv_scheduler_type=='DDIMInverseScheduler':
                inv_timesteps = self.inv_scheduler.timesteps-1 # [1, 101, 201, ... , 901]
                # inv_timesteps_noiser = inv_timesteps + 
                with self.progress_bar(total=num_inversion_steps) as progress_bar:
                    for i, timestep in enumerate(inv_timesteps):
                        noiser_timestep = 999 if i==inv_timesteps.__len__()-1 else inv_timesteps[i+1]
                        t = timestep
                        s = noiser_timestep
                        print(f"{t}, {s}")
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, s)
                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            s,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                        latents = self.inv_scheduler.step(noise_pred, s, latents, return_dict=False)[0]

                        # 7-case A- case: correction
                        if True: #noiser_timestep == 999:
                            latents = self.forward_step_method(
                                latents, 
                                final_latents, 
                                s, 0,
                                prompt_embeds, 
                                timestep_cond, 
                                added_cond_kwargs,
                                extra_step_kwargs,
                                verbose=verbose,
                                )

                        #print(f"iteration {i}, mean of latents {latents.mean()}, std of latents {latents.std()}")

            elif(self.inv_scheduler_type=='EulerDiscreteInverseScheduler'):
                inv_timesteps = self.inv_scheduler.timesteps-1 # [1, 101, 201, ... , 901]
                # inv_timesteps_noiser = inv_timesteps + 
                with self.progress_bar(total=num_inversion_steps) as progress_bar:
                    for i, timestep in enumerate(inv_timesteps):
                        
                        if(i==0):
                            t = 0
                            s  = inv_timesteps[0]
                        else:
                            t = inv_timesteps[i-1]
                            s = inv_timesteps[i]

                        # #if(i==9) : break
                        # noiser_timestep = 999 if i==inv_timesteps.__len__()-1 else inv_timesteps[i+1]
                        # t = timestep
                        # s = noiser_timestep
                        print(f"{t}, {s}")
                        
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, s)
                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            s,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                        latents = self.inv_scheduler.scale_model_input_inv(latents, s)
                        latents = self.inv_scheduler.inv_step(noise_pred, s, 0, latents, return_dict=False)[0]

                        # 7-case A- case: correction
                        if False: #noiser_timestep == 999:
                            latents = self.forward_step_method(
                                latents, 
                                final_latents, 
                                s, t,
                                prompt_embeds, 
                                timestep_cond, 
                                added_cond_kwargs,
                                extra_step_kwargs,
                                verbose=verbose,
                                )
                        else:
                            torch.set_grad_enabled(True)
                            latents = self.gradient_method(
                                latents, 
                                final_latents, 
                                s, t,
                                prompt_embeds, 
                                timestep_cond, 
                                added_cond_kwargs,
                                extra_step_kwargs,
                                verbose=verbose,
                                )
                            torch.set_grad_enabled(False)

                        temp_image = self.vae.decode(latents.cuda() / self.vae.config.scaling_factor)[0]
                        temp_image = self.image_processor.postprocess(temp_image.detach(), output_type='pil', do_denormalize=[True])[0]

        return latents
    
    # The main contribution 2-1: forward step method
    @torch.inference_mode()
    def forward_step_method(self, latents, final_latents, s, t, prompt_embeds, timestep_cond, added_cond_kwargs,extra_step_kwargs, verbose=False):
        # initialize 
        latents_s = latents

        for i in range(100):
            latent_s_model_input = torch.cat([latents_s] * 2) if self.do_classifier_free_guidance else latents_s
            latent_s_model_input = self.scheduler.scale_model_input(latent_s_model_input, s)
            noise_pred = self.unet(
                latent_s_model_input,
                s,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            alpha_prod_s = self.scheduler.alphas_cumprod[int(s)]
            alpha_prod_t = self.scheduler.alphas_cumprod[0]
            beta_prod_s = 1 - alpha_prod_s

            pred_original_sample = (latents_s - beta_prod_s ** (0.5) * noise_pred) / alpha_prod_s ** (0.5)
            pred_epsilon = noise_pred

            prev_sample_diretion = (1 - alpha_prod_t) ** (0.5) * pred_epsilon

            prev_sample = alpha_prod_t ** (0.5) * pred_original_sample + prev_sample_diretion
            latents_t = prev_sample
        
            if t == 999 and i > 20:
                latents_s = latents_s - 0.01 * (latents_t - final_latents)    
            else:
                latents_s = latents_s - 0.1 * (latents_t - final_latents)
            if verbose:
                print(i, (latents_t - final_latents).norm()/final_latents.norm() )
        
        print(f"while forward, alphas used : {alpha_prod_s}, {alpha_prod_t}")
        print(f"while forward, timesteps used : {s}, {t}")
        return latents_s
    
    # The main contribution 2-2: gradient method
    def gradient_method(self, latents, final_latents, s, t, prompt_embeds, timestep_cond, added_cond_kwargs, extra_step_kwargs, verbose=False):
        # Initialize latent variables
        #latents_s = torch.clone(latents).requires_grad_()  # Clone the initial latents and enable gradient tracking
        
        latents_s = latents.clone()
        latents_s.requires_grad_(True)
        final_latents = final_latents.clone()

        optimizer = torch.optim.SGD([latents_s], lr=0.1)  # Define optimizer with a learning rate
        loss_function = torch.nn.MSELoss(reduction='sum')

        model = copy.deepcopy(self.unet)

        for i in range(100):
            latent_s_model_input = torch.cat([latents_s] * 2) if self.do_classifier_free_guidance else latents_s
            latent_s_model_input = self.scheduler.scale_model_input(latent_s_model_input, s)
            noise_pred = model(
                latent_s_model_input,
                s,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            alpha_prod_s = self.scheduler.alphas_cumprod[int(s)]
            alpha_prod_t = self.scheduler.alphas_cumprod[0]
            beta_prod_s = 1 - alpha_prod_s

            pred_original_sample = (latents_s - beta_prod_s ** (0.5) * noise_pred) / alpha_prod_s ** (0.5)
            pred_epsilon = noise_pred

            prev_sample_direction = (1 - alpha_prod_t) ** (0.5) * pred_epsilon

            prev_sample = alpha_prod_t ** (0.5) * pred_original_sample + prev_sample_direction
            latents_t = prev_sample

            # Calculate your loss based on the difference between latents_t and final_latents
            loss = loss_function(latents_t, final_latents)  # Adjust this based on your actual objective function
            
            optimizer.zero_grad()  # Clear accumulated gradients
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update latents_s based on gradients and the learning rate
            
            if verbose:
                print(i, (loss/final_latents.norm()).item())  # Print loss for tracking
            
        return latents_s.detach()  # Return the optimized latent variables without gradients