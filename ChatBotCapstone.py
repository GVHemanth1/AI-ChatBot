import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM

llm = None
current_model = "orca"

MODELS = {
    "orca": {
        "repo": "zoltanctoth/orca_mini_3B-GGUF",
        "file": "orca-mini-3b.q4_0.gguf"
    },
    "llama2": {
        "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "file": "llama-2-7b-chat.Q3_K_S.gguf",
        "type": "llama"
    }
}


def load_model(model_name: str):
    global llm, current_model
    config = MODELS[model_name]
    llm = AutoModelForCausalLM.from_pretrained(
        config["repo"], model_file=config["file"],
        model_type=config.get("type")
    )
    current_model = model_name


def get_prompt(instruction: str, history: List[str] = None) -> str:
    global current_model
    if current_model == "orca":
        system = "You are an AI assistant who answers in very short and precisely"
        prompt = f"### System:\n{system}\n\n### User:\n"
        if len(history) > 0:
            prompt += f"This is the conversation history: {''.join(history)} {instruction} . Now answer the question: "
        prompt += f"{instruction}\n\n### Response:\n"

    elif current_model == "llama2":
        system = "You are a Chat bot AI assistant integrated with Webex who answers in very short and precisely"
        if len(history) > 0:
            prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{history}{instruction} [/INST]"
        prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    #    response = f"Hello, you just sent: {message.content}!"
    if message.content == "forget everything":
        cl.user_session.set("message_history", [])
        disMsg = "Uh oh, I've just forgotten our conversation history"
        print(disMsg)
        msg = cl.Message(content="")
        for words in disMsg:
            await msg.stream_token(words)
            # response += words
            # print(response)
        await msg.update()
        return
    elif message.content == "use llama2":
        load_model("llama2")
        msg = cl.Message(content="Model changed to Llama")
        await msg.send()
        return
    elif message.content == "use orca":
        load_model("orca")
        msg = cl.Message(content="Model changed to orca")
        await msg.send()
        return

    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    for words in llm(prompt, stream=True):
        await msg.stream_token(words)
        response += words
        # print(response)
    await msg.update()
    message_history.append(f"Question asked: {message.content}. You answered: {response}.")


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    # global llm
    load_model("orca")
    print("A new chat session has started!")
