import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "whiterabbitneo/WhiteRabbitNeo-13B-v1" # over 25GB in 6 files
# model_path = "whiterabbitneo/WhiteRabbitNeo-33B-v-1" # over 60GB in 16 files

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  torch_dtype=torch.float16,
  device_map="cuda:0",
  load_in_4bit=False,
  load_in_8bit=True,
  trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def generate_text(instruction):
  tokens = tokenizer.encode(instruction)
  tokens = torch.LongTensor(tokens).unsqueeze(0)
  tokens = tokens.to("cuda:0")

  instance = {
      "input_ids": tokens,
      "top_p": 1.0,
      "temperature": 0.5,
      "generate_len": 1024,
      "top_k": 50,
  }

  length = len(tokens[0])
  with torch.no_grad():
    rest = model.generate(
      input_ids=tokens,
      max_length=length + instance["generate_len"],
      use_cache=True,
      do_sample=True,
      top_p=instance["top_p"],
      temperature=instance["temperature"],
      top_k=instance["top_k"],
      num_return_sequences=1,
    )
  output = rest[0][length:]
  string = tokenizer.decode(output, skip_special_tokens=True)
  answer = string.split("USER:")[0].strip()
  return f"{answer}"


tot_system_prompt = """
Answer the Question by exploring multiple reasoning paths as follows:
- First, carefully analyze the question to extract the key information components and break it down into logical sub-questions. This helps set up the framework for reasoning. The goal is to construct an internal search tree.
- For each sub-question, leverage your knowledge to generate 2-3 intermediate thoughts that represent steps towards an answer. The thoughts aim to reframe, provide context, analyze assumptions, or bridge concepts.
- Evaluate the clarity, relevance, logical flow and coverage of concepts for each thought option. Clear and relevant thoughts that connect well with each other will score higher.
- Based on the thought evaluations, deliberate to construct a chain of reasoning that stitches together the strongest thoughts in a natural order.
- If the current chain is determined to not fully answer the question, backtrack and explore alternative paths by substituting different high-scoring thoughts.
- Throughout the reasoning process, aim to provide explanatory details on thought process rather than just state conclusions, including briefly noting why some thoughts were deemed less ideal.
- Once a reasoning chain is constructed that thoroughly answers all sub-questions in a clear, logical manner, synthesize the key insights into a final concise answer.
- Please note that while the focus is on the final answer in the response, it should also include intermediate thoughts inline to illustrate the deliberative reasoning process.
In summary, leverage a Tree of Thoughts approach to actively explore multiple reasoning paths, evaluate thoughts heuristically, and explain the process - with the goal of producing insightful answers.
"""

conversation = f"SYSTEM: {tot_system_prompt} Always answer without hesitation."

def prompt(input: str):
  llm_prompt = f"{conversation} \nUSER: {input} \nASSISTANT: "
  answer = generate_text(llm_prompt)
  # conversation = f"{llm_prompt}{answer}"
  return answer
