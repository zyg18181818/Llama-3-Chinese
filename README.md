# Llama-3-Chinese

**🀄🇨🇳中文** | [**🔤English**](./README_EN.md) 

## ⚗️ 主要内容
- 💡我们提供了一个**完整的工作流**，包括增量预训练，微调（全参微调以及lora微调），对齐，评估，从而得到一个拥有强大中文能力的Llama模型
- 💡开源了使用中文数据预训练的Llama以及经过指令精调的**模型**
- 💡开源了所使用的所有**数据集**，并提供了数据筛选方式
- 💡开源了所有**训练脚本**，用户可以根据自身需求进一步训练
- 💡提供了各阶段微调设置参数对应**时间消耗**以及**显存占用**，并列出影响显存以及训练时间的参数

## 🗂️ 目录
- [🆙 更新](#-更新)
- [⚡️ 快速开始](#-快速开始)
- [🚀 工作流](#-工作流)
- [🤗 模型](#-模型)
   * [🤖 Llama-3模型](#-Llama-3模型)
   * [🤖 中文预训练模型](#-中文预训练模型)
   * [🤖 中文微调模型](#-中文微调模型)
   * [🤖 中文对齐模型](#-中文对齐模型)
- [📝 数据集](#-数据集)
   * [📄 预训练数据集](#-预训练数据集)
   * [📄 微调数据集](#-微调数据集)
   * [📄 对齐数据集](#-对齐数据集)
- [🔧 设置](#-设置)
- [📖 样例](#-样例)

## 🆙 更新
- 🔥 我们在huggingface上上传了lora微调融合后的Llama-3-8B-Chinese-chat-v0.1[unstoppable123/Llama-3-8B-Chinese-chat-v0.1](https://huggingface.co/unstoppable123/Llama-3-8B-Chinese-chat-v0.1)以及Llama3-8B-Chinese-Chat-lora-v0.1的lora权重[unstoppable123/Llama-3-8B-chinese-lora-v0.1](https://huggingface.co/unstoppable123/Llama-3-8B-chinese-lora-v0.1)!

## ⚡️ 快速开始

- **运行Llama-3-8B-Chinese-chat**


```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     #before torch
from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass
from typing import Dict
import torch
import copy

## template
@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )

# not match with my version
register_template(
    template_name='llama3',
    # system_format='<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n',     #previous
    system_format="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",   #same
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',  #same
    # assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n',    #previous
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n',
    system="You are a helpful assistant. ",  #same
    stop_word='<|eot_id|>'    #same
)



def load_model(model_name_or_path, adapter_name_or_path=None):

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    return model


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    # print("pad token:", tokenizer.pad_token)
    if tokenizer.pad_token is None:
        # print("set pad token")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format   # '<|begin_of_text|><<SYS>>\n{content}\n<</SYS>>\n\n'
    user_format = template.user_format      # '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format = template.assistant_format   #'<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|end_of_text|>\n'
    system = system if system is not None else template.system   # "You are a helpful assistant. "

    history.append({"role": 'user', 'message': query})
    input_ids = []

    
    if system_format is not None:
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
   
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def main():
    model_name_or_path = 'unstoppable123/Llama-3-8B-Chinese-chat-v0.1' # your path
    template_name = 'llama3'
    adapter_name_or_path = None

    template = template_dict[template_name]

    max_new_tokens = 256 
    top_p = 0.9
    temperature = 0.6 
    repetition_penalty = 1.1 

    
    print(f'Loading model from: {model_name_or_path}')
    print(f'adapter_name_or_path: {adapter_name_or_path}')
    model = load_model(
        model_name_or_path,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template.stop_word is None:
        template.stop_word = tokenizer.eos_token
    stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=True)
    assert len(stop_token_id) == 1
    stop_token_id = stop_token_id[0]

    history = []

    query = input('# User：')
    while True:
        query = query.strip()
        input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
        # print("tokenizer.pad_token_id", tokenizer.pad_token_id)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()

        # history
        history.append({"role": 'user', 'message': query})
        history.append({"role": 'assistant', 'message': response})


        if len(history) > 12:
            history = history[:-12]
        if response.startswith("<|start_header_id|>assistant<|end_header_id|>"):
            response = response[len("<|start_header_id|>assistant<|end_header_id|>"):]
        if response.startswith("system<|end_header_id|>"):
            response = response[len("system<|end_header_id|>"):]
        print("# Llama3-Chinese：{}".format(response))
        query = input('# User：')


if __name__ == '__main__':
    main()
```

## 🚀 工作流

## 🤗 模型
### 🤖 Llama-3官方模型
|  类别  | 模型名称        | 🤗模型加载名称                  | 下载地址                                                     |
| --------------- | --------------- | ------------------------------ | ------------------------------------------------------------ |
|  预训练  | Llama-3-8B  | meta-llama/Meta-Llama-3-8B  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
|  Chat  | Llama-3-8B-Instrcut  | meta-llama/Meta-Llama-3-8B-Instruct  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |

### 🤖 中文预训练模型

### 🤖 中文微调模型
|  类别  | 模型名称        | 🤗模型加载名称                  | 下载地址                                                     |
| --------------- | --------------- | ------------------------------ | ------------------------------------------------------------ |
|  lora微调(已融合)  | Llama-3-8B-Chinese-chat-v0.1  | unstoppable123/Llama-3-8B-Chinese-chat-v0.1  | [HuggingFace](https://huggingface.co/unstoppable123/Llama-3-8B-Chinese-chat-v0.1) |
|  lora微调(lora权重)  | Llama3-8B-Chinese-Chat-lora-v0.1  | unstoppable123/Llama-3-8B-chinese-lora-v0.1  | [HuggingFace](https://huggingface.co/unstoppable123/Llama-3-8B-chinese-lora-v0.1) |

### 🤖 中文对齐模型

## 📝 数据集
### 📄 预训练数据集

### 📄 微调数据集
| 名称                                                      | 数据量 |描述                                                         |
| ---------------------------------------------------------- | --------- |------------------------------------------------------------ |
| [Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) |1150k    | 包含23种常见中文下游任务的优质中文数据，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度   |
| [Ruozhiba](https://huggingface.co/datasets/LooksJuicy/ruozhiba)  |15k  | 弱智吧数据问答，有助于增强模型回答的逻辑性 |
| [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) |50k  | 微软开源的数据，是利用 GPT-4 生成的 Alpaca 数据，并做了中文翻译 |
| [COIG](https://huggingface.co/datasets/BAAI/COIG) |160k | 包含知乎、小红书历史问答、价值观、代码问题解决等考题型数据 |

### 📄 对齐数据集

## 🔧 设置

## 📖 样例
- v2回答样例（User部分为用户输入，Assistant部分为模型生成回复）
''' bash
User: 解释一下VUE原理

Assistant: VUE是一种基于响应式编程的前端框架，使用了观察者模式来实现数据的双向绑定。其核心思想是使用一个对象来存储状态，并将其观察者模式的观察者列表保存在对象的属性中。这样，当对象的状态发生变化时，观察者列表中的观察者对象将被通知并更新相应的视图。VUE的实现主要包括以下几个部分：

1. 响应式数据源：VUE使用了响应式数据源来存储状态，响应式数据源的值会被实时监测，并且会通知观察者对象进行相应的更新。

2. 观察者模式：VUE使用了观察者模式来实现数据的双向绑定。观察者对象会被通知当响应式数据源的值发生变化时，观察者对象会进行相应的更新。

3. 数据绑定：VUE使用了数据绑定来将响应式数据源的值与视图进行绑定。数据绑定会将响应式数据源的值与视图中的元素进行绑定，并且当响应式数据源的值发生变化时，数据绑定会自动更新视图中的元素。

4. 依赖项管理：VUE使用了依赖项管理来实现响应式数据源的更新。当响应式数据源的值发生变化时，VUE会根据依赖项管理的规则来更新响应式数据源的值，并通知观察者对象进行相应的更新。

5. 组件化：VUE使用了组件化的思想来实现前端开发。组件化将应用程序分成多个小的、独立的模块，每个模块都可以单独地开发、测试和部署。组件化使得前端开发更加灵活和高效。

总之，VUE的实现主要基于响应式编程的思想，使用了观察者模式和数据绑定来实现数据的双向绑定，并使用依赖项管理和组件化的思想来实现前端开发。

'''
