# Llama-3-Chinese

[**🀄🇨🇳中文**](./README.md) | **🔤English**

## ⚗️ Main Contents
- 💡We provide a **complete workflow**, including incremental pre-training, fine-tuning (full parameter fine-tuning and lora fine-tuning), alignment, evaluation, resulting in a Llama model with strong Chinese capabilities
- 💡We open source **Llama-Chinese** pre-trained with Chinese data and finetuned models with instructions
- 💡We open source all the **datasets** we use and provide a way to filter the data
- 💡We open source all the **training scripts**, users can further train according to their own needs
- 💡We provide each stage of fine-tuning parameters corresponding to **time consumption** and **GPU memory occupation**, and lists the parameters affecting video memory and training time

## 🗂️ Guide
- [🆙 Updates](#-Updates)
- [⚡️ QuickStark](#-QuickStart)
- [🚀 Workflow](#-Workflow)
- [🤗 Models](#-Models)
   * [🤖 Llama-3 official models](#-Llama-3-official-models)
   * [🤖 Chinese Data Pretraining](#-Chinese-Data-Pretraining)
   * [🤖 Chinese Data finetuning](#-Chinese-Data-finetuning)
   * [🤖 Chinese data alignment](#-Chinese-data-alignment)
- [📝 Datasets](#-Datasets)
   * [📄 Chinese pretraining dataset](#-Chinese-pretraining-dataset)
   * [📄 Chinese finetuining dataset](#-Chinese-finetuining-dataset)
   * [📄 Chinese alighnment dataset](#-Chinese-alighnment-dataset)
- [🔧 Settings](#-Settings)
- [📖 Examples](#-Examples)

## 🆙 Updates

- 🔥 2024.04.25: We provide [Llama-3-8B-Chinese-chat-v0.1](https://huggingface.co/unstoppable123/Llama-3-8B-Chinese-chat-v0.1) and the LoRA weights on [Llama3-8B-Chinese-Chat-lora-v0.1](https://huggingface.co/unstoppable123/Llama-3-8B-chinese-lora-v0.1)!
- 🔥 2024.04.28: [Llama3-8B-Chinese-Chat-lora-v0.2](https://huggingface.co/unstoppable123/LLaMA3-7B_chinese_lora_sft_v0.2) is released!

## ⚡️ QuickStart

- **Run Llama-3-8B-Chinese-chat**


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

## 🚀 Workflow

## 🤗 Models
### 🤖 Llama-3 official models
|  Category  | Model Name        | 🤗Download Link                  | Download Link   |
| --------------- | --------------- | ------------------------------ | ------------------------------------------------------------ |
|  Pretrained  | Llama-3-8B  | meta-llama/Meta-Llama-3-8B  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
|  Chat  | Llama-3-8B-Instrcut  | meta-llama/Meta-Llama-3-8B-Instruct  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |

### 🤖 Chinese Data Pretraining

### 🤖 Chinese Data finetuning
|  Category  | Model Name        | 🤗Download Link                  | Download Link   |
| --------------- | --------------- | ------------------------------ | ------------------------------------------------------------ |
|  lora sft(merged)  | Llama-3-8B-Chinese-chat-v0.1  | unstoppable123/Llama-3-8B-Chinese-chat-v0.1  | [HuggingFace](https://huggingface.co/unstoppable123/Llama-3-8B-Chinese-chat-v0.1) |
|  lora sft(lora weights)  | Llama3-8B-Chinese-Chat-lora-v0.1  | unstoppable123/Llama-3-8B-chinese-lora-v0.1  | [HuggingFace](https://huggingface.co/unstoppable123/Llama-3-8B-chinese-lora-v0.1) |
|  lora sft(lora weights) | Llama3-8B-Chinese-Chat-lora-v0.2  | unstoppable123/LLaMA3-7B_chinese_lora_sft_v0.2  | [HuggingFace](https://huggingface.co/unstoppable123/LLaMA3-7B_chinese_lora_sft_v0.2) |


### 🤖 Chinese data alignment

## 📝 Datasets
### 📄 Chinese pretraining dataset

### 📄 Chinese finetuining dataset
| Name                 | Data size | Description                                                         |
| ---------------------------------------------------------- | --------- |------------------------------------------------------------ |
| [Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) |1150k    | Contains 23 common Chinese downstream tasks of high-quality Chinese data, for each task, manually written by a number of command templates, to ensure high quality and richness of data   |
| [Ruozhiba](https://huggingface.co/datasets/LooksJuicy/ruozhiba)  |15k  | The question and answer from Ruozhiba can help enhance the logic of the model's answer |
| [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) |50k  | Microsoft used the open source data generated by GPT-4 and did a Chinese translation |
| [COIG](https://huggingface.co/datasets/BAAI/COIG) |160k | Includes Zhihu, little red book history questions and answers, values, code problem solving and other test type data |

### 📄 Chinese alighnment dataset

## 🔧 Settings

## 📖 Examples
🔵 v0.2 examples（The ***User*** section provides user input, and the ***Assistant*** section generates responses for the model）
```bash
User: 解释一下VUE原理

Assistant: VUE是一种基于响应式编程的前端框架，使用了观察者模式来实现数据的双向绑定。其核心思想是使用一个对象来存储状态，并将其观察者模式的观察者列表保存在对象的属性中。这样，当对象的状态发生变化时，观察者列表中的观察者对象将被通知并更新相应的视图。VUE的实现主要包括以下几个部分：

1. 响应式数据源：VUE使用了响应式数据源来存储状态，响应式数据源的值会被实时监测，并且会通知观察者对象进行相应的更新。

2. 观察者模式：VUE使用了观察者模式来实现数据的双向绑定。观察者对象会被通知当响应式数据源的值发生变化时，观察者对象会进行相应的更新。

3. 数据绑定：VUE使用了数据绑定来将响应式数据源的值与视图进行绑定。数据绑定会将响应式数据源的值与视图中的元素进行绑定，并且当响应式数据源的值发生变化时，数据绑定会自动更新视图中的元素。

4. 依赖项管理：VUE使用了依赖项管理来实现响应式数据源的更新。当响应式数据源的值发生变化时，VUE会根据依赖项管理的规则来更新响应式数据源的值，并通知观察者对象进行相应的更新。

5. 组件化：VUE使用了组件化的思想来实现前端开发。组件化将应用程序分成多个小的、独立的模块，每个模块都可以单独地开发、测试和部署。组件化使得前端开发更加灵活和高效。

总之，VUE的实现主要基于响应式编程的思想，使用了观察者模式和数据绑定来实现数据的双向绑定，并使用依赖项管理和组件化的思想来实现前端开发。

```
