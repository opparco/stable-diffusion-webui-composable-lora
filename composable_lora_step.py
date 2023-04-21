from typing import List, Dict
import re
import json
import math
import sys
import traceback
import random

from modules import extra_networks, shared

re_AND = re.compile(r"\bAND\b")

class LoRA_data:
    def __init__(self, name : str, weight : float):
        self.name = name
        self.weight = weight
    def __repr__(self):
        return f"LoRA:{self.name}:{self.weight}"
    def __str__(self):
        return f"LoRA:{self.name}:{self.weight}"

class LoRA_Weight_CMD:
    def getWeight(self, weight : float, progress: float, step : int, all_step : int):
        return weight

class LoRA_Weight_decrement(LoRA_Weight_CMD):
    def getWeight(self, weight : float, progress: float, step : int, all_step : int):
        return weight * (1 - progress)

class LoRA_Weight_increment(LoRA_Weight_CMD):
    def getWeight(self, weight : float, progress: float, step : int, all_step : int):
        return weight * progress

def raise_(ex):
    raise ex
def not_allow(name):
    return lambda: raise_(Exception(f'function {name} is not allow in LoRA Controller'))

LoRA_Weight_eval_scope = {
    "abs": abs,
    "ceil": math.ceil, "floor": math.floor, "trunc": math.trunc,
    "fmod": math.fmod,
    "gcd": math.gcd, "lcm": math.lcm,
    "perm": math.perm, "comb": math.comb, "gamma": math.gamma,
    "sqrt": math.sqrt, "cbrt": lambda x: pow(x, 1.0 / 3.0),
    "exp": math.exp, "pow": math.pow,
    "log": math.log, "ln": math.log, "log2": math.log2, "log10": math.log10,
    "clamp": lambda x: 1.0 if x > 1 else (0.0 if x < 0 else x),
    "asin": lambda x: (math.acos(1.0 - x * 2.0) + 2.0 * math.pi) / (2.0 * math.pi),
    "acos": lambda x: (math.acos(x * 2.0 - 1.0) + 2.0 * math.pi) / (2.0 * math.pi),
    "atan": lambda x: (math.atan(x) + math.pi) / (2.0 * math.pi),
    "sin": lambda x: (math.sin(x * 2.0 * math.pi - (math.pi / 2.0)) + 1.0) / 2.0,
    "cos": lambda x: (math.sin(x * 2.0 * math.pi + (math.pi / 2.0)) + 1.0) / 2.0,
    "tan": lambda x: math.tan(x * 2.0 * math.pi),
    "sinr": math.sin, "cosr": math.cos, "tanr": math.tan,
    "asinr": math.asin, "acosr": math.acos, "atanr": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "abssin": lambda x: abs(math.sin(x * 2 * math.pi)),
    "abscos": lambda x: abs(math.cos(x * 2 * math.pi)),
    "random": random.random,
    "pi": math.pi, "nan": math.nan, "inf": math.inf,
    #not allow functions
    "eval": not_allow("eval"),
    "exec": not_allow("exec"),
    "compile": not_allow("compile"),
    "breakpoint": not_allow("breakpoint"),
    "__import__": not_allow("__import__")
}

class LoRA_Weight_eval(LoRA_Weight_CMD):
    def __init__(self, command : str):
        self.command = command
        self.is_error = False
    def getWeight(self, weight : float, progress: float, step : int, all_step : int):

        result = None
        #setup local variables
        LoRA_Weight_eval_scope["weight"] = weight
        LoRA_Weight_eval_scope["life"] = progress
        LoRA_Weight_eval_scope["step"] = step
        LoRA_Weight_eval_scope["steps"] = all_step
        LoRA_Weight_eval_scope["warmup"] = lambda x: progress / x if progress < x else 1.0
        LoRA_Weight_eval_scope["cooldown"] = lambda x: (1 - progress) / (1 - x) if progress > x else 1.0
        try:
            result = eval(self.command, LoRA_Weight_eval_scope)
            try:
                result = float(result) * weight
            except Exception:
                raise Exception(f"LoRA Controller command result must be a numble, but got {type(result)}")
            if math.isnan(result):
                raise Exception(f"Can not apply a NaN weight to LoRA.")
            if math.isinf(result):
                raise Exception(f"Can not apply a infinity weight to LoRA.")
        except:
            if not self.is_error:
                print(f"CommandError: {self.command}")
                traceback.print_exception(*sys.exc_info())
                self.is_error = True
            return weight

        return result
    def __repr__(self):
        return f"LoRA_Weight_eval:{self.command}"
    def __str__(self):
        return f"LoRA_Weight_eval:{self.command}"

class LoRA_Controller_Base:
    def __init__(self):
        self.base_weight = 1.0
        self.Weight_Controller = LoRA_Weight_CMD()
    def getWeight(self, weight : float, progress: float, step : int, all_step : int):
        return self.Weight_Controller.getWeight(weight, progress, step, all_step)
    def test(self, test_lora : str, step : int, all_step : int):
        return self.base_weight

#normal lora
class LoRA_Controller(LoRA_Controller_Base):
    def __init__(self, name : str, weight : float):
        super().__init__()
        self.name = name
        self.weight = float(weight)
    def test(self, test_lora : str, step : int, all_step : int):
        if test_lora == self.name:
            return self.getWeight(self.weight, float(step) / float(all_step), step, all_step)
        return 0.0
    def __repr__(self):
        return f"LoRA_Controller:{self.name}[weight={self.weight}]"
    def __str__(self):
        return f"LoRA_Controller:{self.name}[weight={self.weight}]"

#lora with start and end
class LoRA_StartEnd_Controller(LoRA_Controller_Base):
    def __init__(self, name : str, weight : float, start : float | int, end : float | int):
        super().__init__()
        self.name = name
        self.weight = float(weight)
        self.start = float(start)
        self.end = float(end)
    def test(self, test_lora : str, step : int, all_step : int):
        if test_lora == self.name:
            start = self.start
            end = self.end
            if start < 1:
                start = self.start * all_step
            if end < 1:
                end = self.end * all_step
            if end < 0:
                end = all_step
            if (step >= start) and (step <= end):
                return self.getWeight(self.weight, float(step - start) / float(end - start), step, all_step)
        return 0.0
    def __repr__(self):
        return f"LoRA_StartEnd_Controller:{self.name}[weight={self.weight},start at={self.start},end at={self.end}]"
    def __str__(self):
        return f"LoRA_StartEnd_Controller:{self.name}[weight={self.weight},start at={self.start},end at={self.end}]"

#switch lora
class LoRA_Switcher_Controller(LoRA_Controller_Base):
    def __init__(self, lora_dist : List[LoRA_data], start : float | int, end : float | int):
        super().__init__()
        self.lora_dist = lora_dist
        the_list : List[str] = []
        self.lora_list = the_list
        self.start = float(start)
        self.end = float(end)
        for lora_item in self.lora_dist:
            self.lora_list.append(lora_item.name)
    def test(self, test_lora : str, step : int, all_step : int):
        lora_count = len(self.lora_dist)
        if test_lora == self.lora_list[step % lora_count]:
            start = self.start
            end = self.end
            if start < 1:
                start = self.start * all_step
            if end < 1:
                end = self.end * all_step
            if end < 0:
                end = all_step
            if (step >= start) and (step <= end):
                return self.getWeight(self.lora_dist[step % lora_count].weight, float(step - start) / float(end - start), step, all_step)
        return 0.0
    def __repr__(self):
        return f"LoRA_Switcher_Controller:{self.lora_dist}[start at={self.start},end at={self.end}]"
    def __str__(self):
        return f"LoRA_Switcher_Controller:{self.lora_dist}[start at={self.start},end at={self.end}]"


def parse_step_rendering_syntax(prompt: str):
    lora_controllers : List[List[LoRA_Controller_Base]] = []
    subprompts = re_AND.split(prompt)
    for i, subprompt in enumerate(subprompts):
        tmp_lora_controllers: List[LoRA_Controller_Base] = []
        step_rendering_list, pure_loratext = get_all_step_rendering_in_prompt(subprompt)
        for item in step_rendering_list:
            tmp_lora_controllers += get_LoRA_Controllers(item)
        lora_list = get_lora_list(pure_loratext)
        for lora_item in lora_list:
            tmp_lora_controllers.append(LoRA_Controller(lora_item.name, lora_item.weight))
        lora_controllers.append(tmp_lora_controllers)
    return lora_controllers

def check_lora_weight(controllers : List[LoRA_Controller_Base], test_lora : str, step : int, all_step : int):
    result_weight = 0.0
    for controller in controllers:
        calc_weight = controller.test(test_lora, step, all_step)
        if calc_weight > result_weight:
            result_weight = calc_weight
    return result_weight

def get_lora_list(prompt: str):
    result : List[LoRA_data] = []
    _, extra_network_data = extra_networks.parse_prompt(prompt)
    for params in extra_network_data['lora']:
        name = params.items[0]
        multiplier = float(params.items[1]) if len(params.items) > 1 else 1.0
        result.append(LoRA_data(name, multiplier))

    if len(result) <= 0:
        result.append(LoRA_data("", 0.0))

    return result

def get_or_list(prompt: str):
    return prompt.split("|")

re_start_end = re.compile(r"\[\s*\[\s*([^\:\]]+)\:\s*\:([^\]]+)\]\s*\:\s*([^\]]+)\]")
re_strat_at = re.compile(r"\[\s*([^\:\]]+)\:\s*([0-9\.]+)\s*\]")
re_bucket_inside = re.compile(r"\[([^\]]+)\]")
re_extra_net = re.compile(r"<([^>]+):([^>]+)>")
re_python_escape = re.compile(r"\$\$PYTHON_OBJ\$\$(\d+)\^")
re_python_escape_x = re.compile(r"\$\$PYTHON_OBJX?\$\$(\d+)\^")
re_sd_step_render = re.compile(r"\[[^\[\]]+\]")
re_super_cmd = re.compile(r"#([^:#\[\]]+)")

class MySearchResult:
    def __init__(self):
        group : List[str] = []
        self.group = group

def extra_net_split(input_str : str, pattern : str):
    result : List[str] = []
    extra_net_list : List[str] = []
    escape_obj_list : List[str] = []
    def preprossing_escape(match_pt : re.Match):
        escape_obj_list.append(str(match_pt.group(0)))
        return f"$$PYTHON_OBJX$${len(escape_obj_list)-1}^"
    def preprossing_extra_net(match_pt : re.Match):
        extra_net_list.append(str(match_pt.group(0)))
        return f"$$PYTHON_OBJ$${len(extra_net_list)-1}^"
    def unstrip_extra_net_pattern(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        try:
            index = int(match_pt.group(1))
            return extra_net_list[index]
        except Exception:
            return input_str
    def unstrip_text_pattern_obj(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        try:
            index = int(match_pt.group(1))
            return escape_obj_list[index]
        except Exception:
            return input_str
    txt : str = input_str
    txt = re.sub(re_python_escape_x, preprossing_escape, txt)
    txt = re.sub(re_extra_net, preprossing_extra_net, txt)
    pre_result = txt.split(pattern)
    for i in range(len(pre_result)):
        try:
            cur_pattern = str(pre_result[i])
            cur_result = re.sub(re_python_escape, unstrip_extra_net_pattern, cur_pattern)
            cur_result = re.sub(re_python_escape_x, unstrip_text_pattern_obj, cur_result)
            result.append(cur_result)
        except Exception as ex:
            break
    if len(result) <= 0:
        return [input_str]
    return result

def extra_net_re_search(pattern : str | re.Pattern[str], input_str : str):
    result = MySearchResult()
    extra_net_list : List[str] = []
    escape_obj_list : List[str] = []
    def preprossing_escape(match_pt : re.Match):
        escape_obj_list.append(str(match_pt.group(0)))
        return f"$$PYTHON_OBJX$${len(escape_obj_list)-1}^"
    def preprossing_extra_net(match_pt : re.Match):
        extra_net_list.append(str(match_pt.group(0)))
        return f"$$PYTHON_OBJ$${len(extra_net_list)-1}^"
    def unstrip_extra_net_pattern(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        try:
            index = int(match_pt.group(1))
            return extra_net_list[index]
        except Exception:
            return input_str
    def unstrip_text_pattern_obj(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        try:
            index = int(match_pt.group(1))
            return escape_obj_list[index]
        except Exception:
            return input_str
    txt : str = input_str
    txt = re.sub(re_python_escape_x, preprossing_escape, txt)
    txt = re.sub(re_extra_net, preprossing_extra_net, txt)
    pre_result = re.search(pattern, txt)
    for i in range(1000):
        try:
            cur_pattern = str(pre_result.group(i))
            cur_result = re.sub(re_python_escape, unstrip_extra_net_pattern, cur_pattern)
            cur_result = re.sub(re_python_escape_x, unstrip_text_pattern_obj, cur_result)
            result.group.append(cur_result)
        except Exception as ex:
            break
    if len(result.group) <= 0:
        return None
    return result

def unescape_string(input_string : str):
    result = ''
    unicode_list = ['u','x']
    
    i = 0 #for(var i=0; i<input_string.length; ++i)
    while i < len(input_string):
        current_char = input_string[i]
        if current_char == '\\':
            i += 1
            if i >= len(input_string):
                break
            string_body = input_string[i]
            if(string_body.lower() in unicode_list):
                result += f"{current_char}{string_body}"
            else:
                char_added = False
                try:
                    unescaped = json.loads(f"\"{current_char}{string_body}\"")
                    if unescaped:
                        result += unescaped
                        char_added = True
                except Exception:
                    pass
                if not char_added:
                    result += string_body
        else:
            result += current_char
        i += 1
    return str(json.loads(json.dumps(result, indent=4).replace("\\\\", "\\")))
    

def get_LoRA_Controllers(prompt: str):
    result = extra_net_re_search(re_start_end, prompt)
    super_cmd = re.search(re_super_cmd, prompt)
    Weight_Controller = LoRA_Weight_CMD()
    if super_cmd:
        super_cmd_text = unescape_string(super_cmd.group(1)).strip()
        if super_cmd_text.startswith("cmd("):
            Weight_Controller = LoRA_Weight_eval(super_cmd_text[4:-1])
        elif super_cmd_text.startswith("decrease"):
            Weight_Controller = LoRA_Weight_decrement()
        elif super_cmd_text.startswith("increment"):
            Weight_Controller = LoRA_Weight_increment()
    def set_Weight_Controller(controller_list : list[LoRA_Controller_Base], the_controller : LoRA_Weight_CMD):
        for i, the_item in enumerate(controller_list):
            controller_list[i].Weight_Controller = the_controller
        return controller_list
    result_list: List[LoRA_Controller_Base] = []
    if result:
        or_list = get_or_list(result.group[1])
        if len(or_list) == 1: #LoRA with start and end
            lora_list = get_lora_list(or_list[0])
            for lora_item in lora_list:
                try:
                    result_list.append(LoRA_StartEnd_Controller(lora_item.name, lora_item.weight, float(result.group[3]), float(result.group[2])))
                except Exception:
                    continue
            return set_Weight_Controller(result_list, Weight_Controller)
        lora_lists : List[List[LoRA_data]] = []
        max_len = -1
        for or_block in or_list: #or 
            lora_list = get_lora_list(or_block)
            lora_list_len = len(lora_list)
            if lora_list_len > max_len:
                max_len = lora_list_len
            lora_lists.append(lora_list)
        if max_len > 0:
            for i in range(max_len):
                tmp_lora_list : List[LoRA_data] = []
                for it_lora_list in lora_lists:
                    tmp_lora = LoRA_data("", 0.0)
                    if i < len(it_lora_list):
                        tmp_lora = it_lora_list[i]
                    tmp_lora_list.append(tmp_lora)
                result_list.append(LoRA_Switcher_Controller(tmp_lora_list, float(result.group[3]), float(result.group[2])))
        return set_Weight_Controller(result_list, Weight_Controller)
    result = extra_net_re_search(re_strat_at, prompt)
    if result:
        or_list = get_or_list(result.group[1])
        if len(or_list) == 1: #LoRA with start and end
            lora_list = get_lora_list(or_list[0])
            for lora_item in lora_list:
                try:
                    result_list.append(LoRA_StartEnd_Controller(lora_item.name, lora_item.weight, float(result.group[2]), -1.0))
                except Exception:
                    continue
            return set_Weight_Controller(result_list, Weight_Controller)
        lora_lists : List[List[LoRA_data]] = []
        max_len = -1
        for or_block in or_list: #or 
            lora_list = get_lora_list(or_block)
            lora_list_len = len(lora_list)
            if lora_list_len > max_len:
                max_len = lora_list_len
            lora_lists.append(lora_list)
        if max_len > 0:
            for i in range(max_len):
                tmp_lora_list : List[LoRA_data] = []
                for it_lora_list in lora_lists:
                    tmp_lora = LoRA_data("", 0.0)
                    if i < len(it_lora_list):
                        tmp_lora = it_lora_list[i]
                    tmp_lora_list.append(tmp_lora)
                result_list.append(LoRA_Switcher_Controller(tmp_lora_list, float(result.group[2]), -1.0))
        return set_Weight_Controller(result_list, Weight_Controller)
    result = extra_net_re_search(re_bucket_inside, prompt)
    if result:
        bucket_inside = result.group[1]
        split_by_colon = extra_net_split(bucket_inside,":")
        if len(split_by_colon) == 1 and (("|" in bucket_inside) or ("#" in bucket_inside)):
            split_by_colon.append('')
            split_by_colon.append('-1')
        if len(split_by_colon) > 2:
            should_pass = False
            or_list = get_or_list(split_by_colon[0])
            if len(or_list) == 1: #LoRA with start and end
                lora_list = get_lora_list(or_list[0])
                for lora_item in lora_list:
                    try:
                        result_list.append(LoRA_StartEnd_Controller(lora_item.name, lora_item.weight, 0.0, float(split_by_colon[2])))
                    except Exception:
                        continue
                should_pass = True
            if not should_pass:
                lora_lists : List[List[LoRA_data]] = []
                max_len = -1
                for or_block in or_list: #or 
                    lora_list = get_lora_list(or_block)
                    lora_list_len = len(lora_list)
                    if lora_list_len > max_len:
                        max_len = lora_list_len
                    lora_lists.append(lora_list)
                if max_len > 0:
                    for i in range(max_len):
                        tmp_lora_list : List[LoRA_data] = []
                        for it_lora_list in lora_lists:
                            tmp_lora = LoRA_data("", 0.0)
                            if i < len(it_lora_list):
                                tmp_lora = it_lora_list[i]
                            tmp_lora_list.append(tmp_lora)
                        result_list.append(LoRA_Switcher_Controller(tmp_lora_list, 0.0, float(split_by_colon[2])))
            should_pass = False
            or_list = get_or_list(split_by_colon[1])
            if len(or_list) == 1: #LoRA with start and end
                lora_list = get_lora_list(or_list[0])
                for lora_item in lora_list:
                    try:
                        result_list.append(LoRA_StartEnd_Controller(lora_item.name, lora_item.weight, float(split_by_colon[2]), -1.0))
                    except Exception:
                        continue
                should_pass = True
            if not should_pass:
                lora_lists : List[List[LoRA_data]] = []
                max_len = -1
                for or_block in or_list: #or 
                    lora_list = get_lora_list(or_block)
                    lora_list_len = len(lora_list)
                    if lora_list_len > max_len:
                        max_len = lora_list_len
                    lora_lists.append(lora_list)
                if max_len > 0:
                    for i in range(max_len):
                        tmp_lora_list : List[LoRA_data] = []
                        for it_lora_list in lora_lists:
                            tmp_lora = LoRA_data("", 0.0)
                            if i < len(it_lora_list):
                                tmp_lora = it_lora_list[i]
                            tmp_lora_list.append(tmp_lora)
                        result_list.append(LoRA_Switcher_Controller(tmp_lora_list, float(split_by_colon[2]), -1.0))
            return set_Weight_Controller(result_list, Weight_Controller)
    return set_Weight_Controller(result_list, Weight_Controller)

def get_all_step_rendering_in_prompt(input_prompt : str):
    read_rendering_item_list : List[str] = []
    escape_obj_list : List[str] = []
    rendering_item_list : List[str] = []
    def preprossing_step_rendering_item(match_pt : re.Match):
        read_rendering_item_list.append(str(match_pt.group(0)))
        return f"$$PYTHON_OBJ$${len(read_rendering_item_list)-1}^"
    def preprossing_step_rendering_text(match_pt : re.Match):
        escape_obj_list.append(str(match_pt.group(0)))
        return f"$$PYTHON_OBJX$${len(escape_obj_list)-1}^"
    def load_step_rendering_item(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        rendering_item_list.append(input_str)
        return input_str
    def unstrip_rendering_text_pattern(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        try:
            index = int(match_pt.group(1))
            return read_rendering_item_list[index]
        except Exception:
            return input_str
    def unstrip_rendering_text_pattern_obj(match_pt : re.Match):
        input_str = str(match_pt.group(0))
        try:
            index = int(match_pt.group(1))
            return escape_obj_list[index]
        except Exception:
            return input_str
    def unstrip_rendering_text(input_str : str):
        old_result : str = "None"
        result : str = input_str
        while old_result != result:
            old_result = result
            result = re.sub(re_python_escape, unstrip_rendering_text_pattern, result)
        old_result = "None"
        while old_result != result:
            old_result = result
            result = re.sub(re_python_escape_x, unstrip_rendering_text_pattern_obj, result)
        return result
    txt : str = input_prompt
    txt = re.sub(re_python_escape_x, preprossing_step_rendering_text, txt)
    old_txt : str = "None"
    while old_txt != txt:
        old_txt = txt
        txt = re.sub(re_sd_step_render, preprossing_step_rendering_item, txt)
    re.sub(re_python_escape, load_step_rendering_item, txt)
    for i, the_item in enumerate(rendering_item_list):
        rendering_item_list[i] = unstrip_rendering_text(the_item)
    return rendering_item_list, txt
