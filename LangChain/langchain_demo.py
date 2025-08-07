from langchain_community.llms import CTransformers

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF", 
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",
    max_new_token=20
)

print(llm.invoke("What is capital of India"))