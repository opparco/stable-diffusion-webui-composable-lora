from typing import List, Dict
import re
import torch

from modules import extra_networks, prompt_parser, shared

re_AND = re.compile(r"\bAND\b")


def load_prompt_loras(prompt: str):
    prompt_loras.clear()
    subprompts = re_AND.split(prompt)
    for i, subprompt in enumerate(subprompts):
        loras = {}
        _, extra_network_data = extra_networks.parse_prompt(subprompt)
        for params in extra_network_data['lora']:
            name = params.items[0]
            multiplier = float(params.items[1]) if len(params.items) > 1 else 1.0
            loras[name] = multiplier

        prompt_loras.append(loras)


def reset_counters():
    global get_learned_conditioning_prompt_schedules_counter
    global text_model_encoder_counter
    global diffusion_model_counter

    get_learned_conditioning_prompt_schedules_counter = 0
    text_model_encoder_counter = 0
    diffusion_model_counter = 0


def debug(*values: object, lora_layer_name: str):
    if verbose:
        if lora_layer_name.startswith("transformer_"):
            if lora_layer_name.endswith("_11_mlp_fc2"):
                print(*values)
        elif lora_layer_name.startswith("diffusion_model_"):
            if lora_layer_name.endswith("_11_1_proj_out"):
                print(*values)


def lora_forward(compvis_module, input, res):
    global get_learned_conditioning_prompt_schedules_counter
    global text_model_encoder_counter
    global diffusion_model_counter

    import lora

    if len(lora.loaded_loras) == 0:
        return res

    lora_layer_name: str | None = getattr(compvis_module, 'lora_layer_name', None)
    if lora_layer_name is None:
        return res

    num_prompts = len(prompt_loras)

    # debug(f"lora.forward lora_layer_name={lora_layer_name} in.shape={input.shape} res.shape={res.shape} num_batches={num_batches} num_prompts={num_prompts}", lora_layer_name=lora_layer_name)

    for lora in lora.loaded_loras:
        module = lora.modules.get(lora_layer_name, None)
        if module is None:
            continue

        if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
            patch = module.up(module.down(res))
        else:
            patch = module.up(module.down(input))

        alpha = module.alpha / module.up.weight.shape[1] if module.alpha else 1.0

        # debug(f"lora.name={lora.name} lora.mul={lora.multiplier} alpha={alpha} pat.shape={patch.shape}")

        if enabled:
            if lora_layer_name.startswith("transformer_"):  # "transformer_text_model_encoder_"
                #
                if get_learned_conditioning_prompt_schedules_counter != 1 and 0 <= text_model_encoder_counter < len(prompt_loras):
                    # c
                    loras = prompt_loras[text_model_encoder_counter]
                    multiplier = loras.get(lora.name, 0.0)
                    if multiplier != 0.0:
                        debug(f"c #{text_model_encoder_counter} lora.name={lora.name} mul={multiplier}", lora_layer_name=lora_layer_name)
                        res += multiplier * alpha * patch
                else:
                    # uc
                    if opt_uc_text_model_encoder and lora.multiplier != 0.0:
                        debug(f"uc #{text_model_encoder_counter} lora.name={lora.name} lora.mul={lora.multiplier}", lora_layer_name=lora_layer_name)
                        res += lora.multiplier * alpha * patch

                if get_learned_conditioning_prompt_schedules_counter != 1 and lora_layer_name.endswith("_11_mlp_fc2"):  # last lora_layer_name of text_model_encoder
                    text_model_encoder_counter += 1
                    # c1 c2 c1 c2 ..
                    if text_model_encoder_counter == len(prompt_loras):
                        text_model_encoder_counter = 0

            elif res.shape[0] == num_batches * num_prompts + num_batches:  # "diffusion_model_"
                # tensor.shape[1] == uncond.shape[1]
                tensor_off = 0
                uncond_off = num_batches * num_prompts
                for b in range(num_batches):
                    # c
                    for p, loras in enumerate(prompt_loras):
                        multiplier = loras.get(lora.name, 0.0)
                        if multiplier != 0.0:
                            debug(f"tensor #{b}.{p} lora.name={lora.name} mul={multiplier}", lora_layer_name=lora_layer_name)
                            res[tensor_off] += multiplier * alpha * patch[tensor_off]
                        tensor_off += 1

                    # uc
                    if opt_uc_diffusion_model and lora.multiplier != 0.0:
                        debug(f"uncond lora.name={lora.name} lora.mul={lora.multiplier}", lora_layer_name=lora_layer_name)
                        res[uncond_off] += lora.multiplier * alpha * patch[uncond_off]
                    uncond_off += 1
            else:  # "diffusion_model_"
                # tensor.shape[1] != uncond.shape[1]
                if 0 <= diffusion_model_counter < len(prompt_loras):
                    # c
                    loras = prompt_loras[diffusion_model_counter]
                    multiplier = loras.get(lora.name, 0.0)
                    if multiplier != 0.0:
                        debug(f"c #{diffusion_model_counter} lora.name={lora.name} mul={multiplier}", lora_layer_name=lora_layer_name)
                        res += multiplier * alpha * patch
                else:
                    # uc
                    if opt_uc_diffusion_model and lora.multiplier != 0.0:
                        debug(f"uc {lora_layer_name} lora.name={lora.name} lora.mul={lora.multiplier}", lora_layer_name=lora_layer_name)
                        res += lora.multiplier * alpha * patch

                if lora_layer_name.endswith("_11_1_proj_out"):  # last lora_layer_name of diffusion_model
                    diffusion_model_counter += 1
                    # c1 c2 .. uc
                    if diffusion_model_counter == len(prompt_loras) + 1:
                        diffusion_model_counter = 0
        else:
            # default
            if lora.multiplier != 0.0:
                debug(f"DEFAULT {lora_layer_name} lora.name={lora.name} lora.mul={lora.multiplier}", lora_layer_name=lora_layer_name)
                res += lora.multiplier * alpha * patch

    return res


def lora_Linear_forward(self, input):
    return lora_forward(self, input, torch.nn.Linear_forward_before_lora(self, input))


def lora_Conv2d_forward(self, input):
    return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora(self, input))


def lora_get_learned_conditioning_prompt_schedules(prompts, steps):
    global get_learned_conditioning_prompt_schedules_counter
    #
    # order: uc c
    #

    prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules_before_lora(prompts, steps)

    get_learned_conditioning_prompt_schedules_counter += 1

    return prompt_schedules


enabled = False
opt_uc_text_model_encoder = False
opt_uc_diffusion_model = False
verbose = False

num_batches: int = 0
prompt_loras: List[Dict[str, float]] = []
get_learned_conditioning_prompt_schedules_counter: int = 0
text_model_encoder_counter: int = 0
diffusion_model_counter: int = 0
