# stable-diffusion-webui-composable-lora
This extension replaces the built-in LoRA forward procedure.

## Features
1. Compatible with Composable-Diffusion
By associating LoRA's insertion position in the prompt with "AND" syntax, LoRA's scope of influence is limited to a specific subprompt.
Built-in LoRA ignores the LoRA insert position.

2. Eliminate the impact on negative prompts
With the built-in LoRA, negative prompts are always affected by LoRA. This often has a negative impact on the output.
So this extension offers options to eliminate the negative effects.

## How to use
### Enabled
When checked, Composable LoRA is enabled.

### Use Lora to uc text model encoder
Enable LoRA for uncondition (negative prompt) text model encoder.
With this disabled, you can expect better output.
Built-in LoRA is always enabled.

### Use Lora to uc diffusion model
Enable LoRA for uncondition (negative prompt) diffusion model (denoiser).
With this disabled, you can expect better output.
Built-in LoRA is always enabled.
