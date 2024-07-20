import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-13b-Python-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-13b-Python-hf")

# Create a pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_code(prompt):
    output = pipe(prompt, max_length=50)
    return output[0]['generated_text']

iface = gr.Interface(fn=generate_code, inputs="text", outputs="text")
iface.launch(share=True)
