# Composable LoRA
This extension replaces the built-in LoRA forward procedure.

## Features
### Compatible with Composable-Diffusion
By associating LoRA's insertion position in the prompt with "AND" syntax, LoRA's scope of influence is limited to a specific subprompt.

### Eliminate the impact on negative prompts
With the built-in LoRA, negative prompts are always affected by LoRA. This often has a negative impact on the output.
So this extension offers options to eliminate the negative effects.

## How to use
### Enabled
When checked, Composable LoRA is enabled.

### Use Lora in uc text model encoder
Enable LoRA for uncondition (negative prompt) text model encoder.
With this disabled, you can expect better output.

### Use Lora in uc diffusion model
Enable LoRA for uncondition (negative prompt) diffusion model (denoiser).
With this disabled, you can expect better output.

## compatibilities
--always-batch-cond-uncond must be enabled  with --medvram or --lowvram
