#
# Composable-Diffusion with Lora
#
import torch
import gradio as gr

import composable_lora
import modules.scripts as scripts
from modules import script_callbacks
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

class ComposableLoraScript(scripts.Script):
    def title(self):
        return "Composable Lora"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Composable Lora", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                opt_composable_with_step = gr.Checkbox(value=False, label="Composable LoRA with step")
                opt_uc_text_model_encoder = gr.Checkbox(value=False, label="Use Lora in uc text model encoder")
                opt_uc_diffusion_model = gr.Checkbox(value=False, label="Use Lora in uc diffusion model")
                opt_plot_lora_weight = gr.Checkbox(value=False, label="plot the LoRA weight in all steps")

        return [enabled, opt_composable_with_step, opt_uc_text_model_encoder, opt_uc_diffusion_model, opt_plot_lora_weight]

    def process(self, p: StableDiffusionProcessing, enabled: bool, opt_composable_with_step: bool, opt_uc_text_model_encoder: bool, opt_uc_diffusion_model: bool, opt_plot_lora_weight: bool):
        composable_lora.enabled = enabled
        composable_lora.opt_uc_text_model_encoder = opt_uc_text_model_encoder
        composable_lora.opt_uc_diffusion_model = opt_uc_diffusion_model
        composable_lora.opt_composable_with_step = opt_composable_with_step
        composable_lora.opt_plot_lora_weight = opt_plot_lora_weight

        composable_lora.num_batches = p.batch_size
        composable_lora.num_steps = p.steps

        if composable_lora.should_reload() and enabled:
            torch.nn.Linear.forward = composable_lora.lora_Linear_forward
            torch.nn.Conv2d.forward = composable_lora.lora_Conv2d_forward

        composable_lora.reset_step_counters()

        prompt = p.all_prompts[0]
        composable_lora.load_prompt_loras(prompt)
        if opt_composable_with_step:
            print("Loading LoRA step controller...")

    def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        composable_lora.reset_counters()

    def postprocess(self, p, processed, *args):
        if composable_lora.enabled and composable_lora.opt_plot_lora_weight:
            processed.images.extend([composable_lora.plot_lora()])
