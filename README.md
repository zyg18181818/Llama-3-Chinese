# Llama-3-Chinese

## Updates:

- 🔥 We provide Llama-3-8B-Chinese-chat-v0.1 at [unstoppable123/Llama-3-8B-Chinese-chat-v0.1](https://huggingface.co/unstoppable123/Llama-3-8B-Chinese-chat-v0.1) and the LoRA weights for Llama3-8B-Chinese-Chat-lora-v0.1 at [unstoppable123/Llama-3-8B-chinese-lora-v0.1](https://huggingface.co/unstoppable123/Llama-3-8B-chinese-lora-v0.1)!



## 1.Datasets


## 2.QuickStart

- **Run Llama-3-8B-Chinese-chat**

  You can run the following python script:

  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  
  model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"
  
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
      model_id, torch_dtype="auto", device_map="auto"
  )
  
  messages = [
      {"role": "system", "content": "You are Llama3-8B-Chinese-Chat, which is finetuned on Llama3-8B-Instruct with Chinese-English mixed data by the ORPO alignment algorithm. You, Llama3-8B-Chinese-Chat, is developed by Shenzhi Wang (王慎执 in Chinese). You are a helpful assistant."},
      {"role": "user", "content": "介绍一下你自己"},
  ]
  
  input_ids = tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, return_tensors="pt"
  ).to(model.device)
  
  outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )
  response = outputs[0][input_ids.shape[-1]:]
  print(tokenizer.decode(response, skip_special_tokens=True))
  ```

