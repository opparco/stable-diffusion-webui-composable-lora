#
# Composable-Diffusion with Lora
#
import torch
import gradio as gr

import composable_lora
import modules.scripts as scripts
from modules import script_callbacks, prompt_parser
from modules.processing import StableDiffusionProcessing


def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora


if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

torch.nn.Linear.forward = composable_lora.lora_Linear_forward
torch.nn.Conv2d.forward = composable_lora.lora_Conv2d_forward

script_callbacks.on_script_unloaded(unload)


if not hasattr(prompt_parser, 'get_learned_conditioning_prompt_schedules_before_lora'):
    prompt_parser.get_learned_conditioning_prompt_schedules_before_lora = prompt_parser.get_learned_conditioning_prompt_schedules
prompt_parser.get_learned_conditioning_prompt_schedules = composable_lora.lora_get_learned_conditioning_prompt_schedules


class ComposableLoraScript(scripts.Script):
    def title(self):
        return "Composable Lora"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Composable Lora", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                opt_uc_text_model_encoder = gr.Checkbox(value=False, label="Use Lora in uc text model encoder")
                opt_uc_diffusion_model = gr.Checkbox(value=False, label="Use Lora in uc diffusion model")

        return [enabled, opt_uc_text_model_encoder, opt_uc_diffusion_model]

    def process(self, p: StableDiffusionProcessing, enabled: bool, opt_uc_text_model_encoder: bool, opt_uc_diffusion_model: bool):
        composable_lora.enabled = enabled
        composable_lora.opt_uc_text_model_encoder = opt_uc_text_model_encoder
        composable_lora.opt_uc_diffusion_model = opt_uc_diffusion_model

        composable_lora.num_batches = p.batch_size

        prompt = p.all_prompts[0]
        composable_lora.load_prompt_loras(prompt)

    def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        composable_lora.reset_counters()
