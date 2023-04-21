from typing import List, Dict
import re
import torch
import composable_lora_step
import plot_helper
from modules import extra_networks, shared

re_AND = re.compile(r"\bAND\b")

def load_prompt_loras(prompt: str):
    global is_single_block
    global full_controllers
    global first_log_drawing
    prompt_loras.clear()
    prompt_blocks.clear()
    lora_controllers.clear()
    drawing_data.clear()
    full_controllers.clear()
    drawing_lora_names.clear()

    subprompts = re_AND.split(prompt)
    tmp_prompt_loras = []
    tmp_prompt_blocks = []
    for i, subprompt in enumerate(subprompts):
        loras = {}
        _, extra_network_data = extra_networks.parse_prompt(subprompt)
        for params in extra_network_data['lora']:
            name = params.items[0]
            multiplier = float(params.items[1]) if len(params.items) > 1 else 1.0
            loras[name] = multiplier

        tmp_prompt_loras.append(loras)
        tmp_prompt_blocks.append(subprompt)
    is_single_block = (len(tmp_prompt_loras) == 1)
    prompt_loras.extend(tmp_prompt_loras * num_batches)
    tmp_lora_controllers = composable_lora_step.parse_step_rendering_syntax(prompt)
    lora_controllers.extend(tmp_lora_controllers * num_batches)
    prompt_blocks.extend(tmp_prompt_blocks * num_batches)
    for controller_it in tmp_lora_controllers:
        full_controllers += controller_it
    first_log_drawing = False

def reset_counters():
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    global should_print

    # reset counter to uc head
    text_model_encoder_counter = -1
    diffusion_model_counter = 0
    step_counter += 1
    should_print = True
    
def reset_step_counters():
    global step_counter
    global should_print

    should_print = True
    step_counter = 0

def add_step_counters(): 
    global step_counter
    global should_print

    should_print = True
    step_counter += 1

    if step_counter > num_steps:
        step_counter = 0
    else:
        if opt_plot_lora_weight:
            log_lora()

def log_lora():
    import lora
    tmp_data : List[float] = []
    for m_lora in lora.loaded_loras:
        current_lora = m_lora.name
        multiplier = m_lora.multiplier
        if opt_composable_with_step:
            multiplier = composable_lora_step.check_lora_weight(full_controllers, current_lora, step_counter, num_steps)
        index = -1
        if current_lora in drawing_lora_names:
            index = drawing_lora_names.index(current_lora)
        else:
            index = len(drawing_lora_names)
            drawing_lora_names.append(current_lora)
        if index >= len(tmp_data):
            for i in range(len(tmp_data), index):
                tmp_data.append(0.0)
            tmp_data.append(multiplier)
        else:
            tmp_data[index] = multiplier
    drawing_data.append(tmp_data)

def plot_lora():
    max_size = -1
    drawing_data.insert(0, drawing_lora_first_index)
    for datalist in drawing_data:
        datalist_len = len(datalist)
        if datalist_len > max_size:
            max_size = datalist_len
    for i, datalist in enumerate(drawing_data):
        datalist_len = len(datalist)
        if datalist_len < max_size:
            drawing_data[i].extend([0.0]*(max_size - datalist_len))
    return plot_helper.plot_lora_weight(drawing_data, drawing_lora_names)

def lora_forward(compvis_module, input, res):
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    global should_print
    global first_log_drawing
    global drawing_lora_first_index
    import lora

    if not first_log_drawing:
        first_log_drawing = True
        print("Composable LoRA load successful.")
        if opt_plot_lora_weight:
            log_lora()
            drawing_lora_first_index = drawing_data[0]

    if len(lora.loaded_loras) == 0:
        return res

    lora_layer_name_loading : str | None = getattr(compvis_module, 'lora_layer_name', None)
    if lora_layer_name_loading is None:
        return res
    #let it type is actually a string
    lora_layer_name : str = str(lora_layer_name_loading)
    del lora_layer_name_loading

    num_loras = len(lora.loaded_loras)
    if text_model_encoder_counter == -1:
        text_model_encoder_counter = len(prompt_loras) * num_loras

    tmp_check_loras = [] #store which lora are already apply
    tmp_check_loras.clear()

    for m_lora in lora.loaded_loras:
        module = m_lora.modules.get(lora_layer_name, None)
        if module is None:
            #fix the loCon issue
            if lora_layer_name.endswith("_11_mlp_fc2"):  # locon doesn't has _11_mlp_fc2 layer
                text_model_encoder_counter += 1
                # c1 c1 c2 c2 .. .. uc uc
                if text_model_encoder_counter == (len(prompt_loras) + num_batches) * num_loras:
                    text_model_encoder_counter = 0
            if lora_layer_name.endswith("_11_1_proj_out"):  # locon doesn't has _11_1_proj_out layer
                diffusion_model_counter += res.shape[0]
                # c1 c2 .. uc
                if diffusion_model_counter >= (len(prompt_loras) + num_batches) * num_loras:
                    diffusion_model_counter = 0
                    add_step_counters()
            continue

        current_lora = m_lora.name
        lora_already_used = False
        for check_lora in tmp_check_loras:
            if current_lora == check_lora:
                #find the same lora, marked
                lora_already_used = True
                break
        #store the applied lora into list
        tmp_check_loras.append(current_lora)
        #if current lora already apply, skip this lora
        if lora_already_used == True:
            continue

        if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
            if hasattr(module, 'inference'):
                patch = module.inference(res)
            elif hasattr(module, 'up'):
                patch = module.up(module.down(res))
            else:
                raise NotImplementedError(
                    "Your settings, extensions or models are not compatible with each other."
                )
        else:
            if hasattr(module, 'inference'):
                patch = module.inference(input)
            elif hasattr(module, 'up'):
                patch = module.up(module.down(input))
            else:
                raise NotImplementedError(
                    "Your settings, extensions or models are not compatible with each other."
                )
   
        alpha : float = 1.0
        if hasattr(module, 'up'):
            alpha = (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)
        else: #handle if module.up is undefined
            alpha = (module.alpha / module.dim if module.alpha else 1.0)

        num_prompts = len(prompt_loras)

        # print(f"lora.name={m_lora.name} lora.mul={m_lora.multiplier} alpha={alpha} pat.shape={patch.shape}")
        if enabled:
            if lora_layer_name.startswith("transformer_"):  # "transformer_text_model_encoder_"
                #
                if 0 <= text_model_encoder_counter // num_loras < len(prompt_loras):
                    # c
                    loras = prompt_loras[text_model_encoder_counter // num_loras]
                    multiplier = loras.get(m_lora.name, 0.0)
                    if multiplier != 0.0:
                        # print(f"c #{text_model_encoder_counter // num_loras} lora.name={m_lora.name} mul={multiplier}  lora_layer_name={lora_layer_name}")
                        res += multiplier * alpha * patch
                else:
                    # uc
                    if (opt_uc_text_model_encoder or is_single_block) and m_lora.multiplier != 0.0:
                        # print(f"uc #{text_model_encoder_counter // num_loras} lora.name={m_lora.name} lora.mul={m_lora.multiplier}  lora_layer_name={lora_layer_name}")
                        res += m_lora.multiplier * alpha * patch

                if lora_layer_name.endswith("_11_mlp_fc2"):  # last lora_layer_name of text_model_encoder
                    text_model_encoder_counter += 1
                    # c1 c1 c2 c2 .. .. uc uc
                    if text_model_encoder_counter == (len(prompt_loras) + num_batches) * num_loras:
                        text_model_encoder_counter = 0

            elif lora_layer_name.startswith("diffusion_model_"):  # "diffusion_model_"

                if res.shape[0] == num_batches * num_prompts + num_batches:
                    # tensor.shape[1] == uncond.shape[1]
                    tensor_off = 0
                    uncond_off = num_batches * num_prompts
                    for b in range(num_batches):
                        # c
                        for p, loras in enumerate(prompt_loras):
                            multiplier = loras.get(m_lora.name, 0.0)
                            if opt_composable_with_step:
                                prompt_block_id = p
                                lora_controller = lora_controllers[prompt_block_id]
                                multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora.name, step_counter, num_steps)
                            if multiplier != 0.0:
                                # print(f"tensor #{b}.{p} lora.name={m_lora.name} mul={multiplier} lora_layer_name={lora_layer_name}")
                                res[tensor_off] += multiplier * alpha * patch[tensor_off]
                            tensor_off += 1

                        # uc
                        if (opt_uc_diffusion_model or is_single_block) and m_lora.multiplier != 0.0:
                            # print(f"uncond lora.name={m_lora.name} lora.mul={m_lora.multiplier} lora_layer_name={lora_layer_name}")
                            multiplier = m_lora.multiplier
                            if is_single_block and opt_composable_with_step:
                                multiplier = composable_lora_step.check_lora_weight(full_controllers, m_lora.name, step_counter, num_steps)
                            res[uncond_off] += multiplier * alpha * patch[uncond_off]
                        
                        uncond_off += 1
                else:
                    # tensor.shape[1] != uncond.shape[1]
                    cur_num_prompts = res.shape[0]
                    base = (diffusion_model_counter // cur_num_prompts) // num_loras * cur_num_prompts
                    prompt_len = len(prompt_loras)
                    if 0 <= base < len(prompt_loras):
                        # c
                        for off in range(cur_num_prompts):
                            if base + off < prompt_len:
                                loras = prompt_loras[base + off]
                                multiplier = loras.get(m_lora.name, 0.0)
                                if opt_composable_with_step:
                                    prompt_block_id = base + off
                                    lora_controller = lora_controllers[prompt_block_id]
                                    multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora.name, step_counter, num_steps)
                                if multiplier != 0.0:
                                    # print(f"c #{base + off} lora.name={m_lora.name} mul={multiplier} lora_layer_name={lora_layer_name}")
                                    res[off] += multiplier * alpha * patch[off]
                    else:
                        # uc
                        if (opt_uc_diffusion_model or is_single_block) and m_lora.multiplier != 0.0:
                            # print(f"uc {lora_layer_name} lora.name={m_lora.name} lora.mul={m_lora.multiplier}")
                            multiplier = m_lora.multiplier
                            if is_single_block and opt_composable_with_step:
                                multiplier = composable_lora_step.check_lora_weight(full_controllers, m_lora.name, step_counter, num_steps)

                            res += multiplier * alpha * patch

                    if lora_layer_name.endswith("_11_1_proj_out"):  # last lora_layer_name of diffusion_model
                        diffusion_model_counter += cur_num_prompts
                        # c1 c2 .. uc
                        if diffusion_model_counter >= (len(prompt_loras) + num_batches) * num_loras:
                            diffusion_model_counter = 0
                            add_step_counters()
            else:
                # default
                if m_lora.multiplier != 0.0:
                    # print(f"default {lora_layer_name} lora.name={m_lora.name} lora.mul={m_lora.multiplier}")
                    res += m_lora.multiplier * alpha * patch
        else:
            # default
            if m_lora.multiplier != 0.0:
                # print(f"DEFAULT {lora_layer_name} lora.name={m_lora.name} lora.mul={m_lora.multiplier}")
                res += m_lora.multiplier * alpha * patch

    return res

def lora_Linear_forward(self, input):
    if (not self.weight.is_cuda) and input.is_cuda: #if variables not on the same device (between cpu and gpu)
        self_weight_cuda = self.weight.cuda() #pass to GPU
        to_del = self.weight
        self.weight = None                    #delete CPU variable
        del to_del
        del self.weight                       #avoid pytorch 2.0 throwing exception
        self.weight = self_weight_cuda        #load GPU data to self.weight
    return lora_forward(self, input, torch.nn.Linear_forward_before_lora(self, input))

def lora_Conv2d_forward(self, input):
    if (not self.weight.is_cuda) and input.is_cuda:
        self_weight_cuda = self.weight.cuda()
        to_del = self.weight
        self.weight = None
        del to_del
        del self.weight #avoid "cannot assign XXX as parameter YYY (torch.nn.Parameter or None expected)"
        self.weight = self_weight_cuda
    return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora(self, input))

def should_reload():
    #pytorch 2.0 should reload
    match = re.search(r"\d+\.\d+",str(torch.__version__)) 
    if not match:
        return True
    ver = float(match.group(0))
    return ver >= 2.0

enabled = False
opt_composable_with_step = False
opt_uc_text_model_encoder = False
opt_uc_diffusion_model = False
opt_plot_lora_weight = False
verbose = True

drawing_lora_names : List[str] = []
drawing_data : List[List[float]] = []
drawing_lora_first_index : List[float] = []
first_log_drawing : bool = False

is_single_block : bool = False
num_batches: int = 0
num_steps: int = 20
prompt_loras: List[Dict[str, float]] = []
text_model_encoder_counter: int = -1
diffusion_model_counter: int = 0
step_counter: int = 0

should_print : bool = True
prompt_blocks: List[str] = []
lora_controllers: List[List[composable_lora_step.LoRA_Controller_Base]] = []
full_controllers: List[composable_lora_step.LoRA_Controller_Base] = []
