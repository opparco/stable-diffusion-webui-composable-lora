from typing import List, Dict
import re
import torch

from modules import extra_networks, shared

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


def reset_text_model_encoder_counter():
    global text_model_encoder_counter
    global diff_model_layer_counter

    # reset counter to uc head
    text_model_encoder_counter = -1
    diff_model_layer_counter = 0


def lora_forward(compvis_module, input, res):
    global text_model_encoder_counter
    global diff_model_layer_counter

    import lora

    if len(lora.loaded_loras) == 0:
        return res

    lora_layer_name: str | None = getattr(compvis_module, 'lora_layer_name', None)
    if lora_layer_name is None:
        return res

    num_loras = len(lora.loaded_loras)
    if text_model_encoder_counter == -1:
        text_model_encoder_counter = len(prompt_loras) * num_loras

    # print(f"lora.forward lora_layer_name={lora_layer_name} in.shape={input.shape} res.shape={res.shape}")

    for lora in lora.loaded_loras:
        module = lora.modules.get(lora_layer_name, None)
        if module is None:
            continue

        if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
            patch = module.up(module.down(res))
        else:
            patch = module.up(module.down(input))

        alpha = module.alpha / module.up.weight.shape[1] if module.alpha else 1.0

        num_prompts = len(prompt_loras)

        # print(f"num_batches={num_batches} num_prompts={num_prompts} lora.name={lora.name} lora.mul={lora.multiplier} alpha={alpha} pat.shape={patch.shape}")

        if enabled:
            if lora_layer_name.startswith("transformer_"):  # "transformer_text_model_encoder_"
                #
                # This code evaluated twice in get_learned_conditioning.
                #
                if 0 <= text_model_encoder_counter // num_loras < len(prompt_loras):
                    # c
                    loras = prompt_loras[text_model_encoder_counter // num_loras]
                    multiplier = loras.get(lora.name, 0.0)
                    if multiplier != 0.0:
                        # print(f"c #{text_model_encoder_counter // 2} lora.name={lora.name} mul={multiplier}")
                        res += multiplier * alpha * patch
                else:
                    # uc
                    if opt_uc_text_model_encoder and lora.multiplier != 0.0:
                        # print(f"uc #{text_model_encoder_counter // 2} lora.name={lora.name} lora.mul={lora.multiplier}")
                        res += lora.multiplier * alpha * patch

                if lora_layer_name.endswith("_11_mlp_fc2"):  # last
                    text_model_encoder_counter += 1
                    # c1 c1 c2 c2 .. .. uc uc
                    if text_model_encoder_counter == (len(prompt_loras) + 1) * num_loras:
                        text_model_encoder_counter = 0

            elif lora_layer_name.startswith("diffusion_model_"): # "diffusion_model_"

                if res.shape[0] == num_batches * num_prompts + num_batches:
                    #
                    #
                    #
                    tensor_off = 0
                    uncond_off = num_batches * num_prompts
                    for b in range(num_batches):
                        # c
                        for p, loras in enumerate(prompt_loras):
                            multiplier = loras.get(lora.name, 0.0)
                            if multiplier != 0.0:
                                print(f"tensor #{b}.{p} lora.name={lora.name} mul={multiplier}")
                                res[tensor_off] += multiplier * alpha * patch[tensor_off]
                            tensor_off += 1

                        # uc
                        if opt_uc_diffusion_model and lora.multiplier != 0.0:
                            # print(f"uncond lora.name={lora.name} lora.mul={lora.multiplier}")
                            res[uncond_off] += lora.multiplier * alpha * patch[uncond_off]
                        uncond_off += 1
                else:
                    cur_num_prompts = int(res.shape[0])
                    base = (diff_model_layer_counter // cur_num_prompts) // num_loras * cur_num_prompts
                    if 0 <= base < len(prompt_loras):
                        # c
                        for off in range(cur_num_prompts):
                            loras = prompt_loras[base + off]
                            multiplier = loras.get(lora.name, 0.0)
                            if multiplier != 0.0:
                                res[off] += multiplier * alpha * patch[off]
                    else:
                        # uc
                        if opt_uc_diffusion_model and lora.multiplier != 0.0:
                            res += lora.multiplier * alpha * patch

                    if lora_layer_name.endswith("_output_blocks_11_1_proj_out"):
                        diff_model_layer_counter += cur_num_prompts
                        if diff_model_layer_counter >= (len(prompt_loras) + 1) * num_loras:
                            diff_model_layer_counter = 0

            else:
                # default
                if lora.multiplier != 0.0:
                    # print(f"default {lora_layer_name} lora.name={lora.name} lora.mul={lora.multiplier}")
                    res += lora.multiplier * alpha * patch
        else:
            # default
            if lora.multiplier != 0.0:
                # print(f"DEFAULT {lora_layer_name} lora.name={lora.name} lora.mul={lora.multiplier}")
                res += lora.multiplier * alpha * patch

    return res


def lora_Linear_forward(self, input):
    return lora_forward(self, input, torch.nn.Linear_forward_before_lora(self, input))


def lora_Conv2d_forward(self, input):
    return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora(self, input))


enabled = False
opt_uc_text_model_encoder = False
opt_uc_diffusion_model = False

num_batches: int = 0
prompt_loras: List[Dict[str, float]] = []
text_model_encoder_counter: int = 0
diff_model_layer_counter: int = 0
