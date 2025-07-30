from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q3_K_S.gguf",
    model_type="llama"
)


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant who answers in very short and precisely"
    # prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


question = "What is the name of the capital city of India?"

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
print()
