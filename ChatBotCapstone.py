import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant who answers in very short and precisely"
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)} {instruction} . Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    #    response = f"Hello, you just sent: {message.content}!"
    if message.content == "forget everything":
        message_history = ""
        disMsg = "Uh oh, I've just forgotten our conversation history"
        print(disMsg)
        msg = cl.Message(content="")
        for words in disMsg:
            await msg.stream_token(words)
            # response += words
            # print(response)
        await msg.update()

    else:
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
        message_history.append("Question asked :" + message.content + ". " + "You answered : " + response + ". ")


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    print("A new chat session has started!")


"""
history = []


question = "What is the name of the capital city of India?"

answer = ""

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)
question = "And which is of the United States?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
"""
