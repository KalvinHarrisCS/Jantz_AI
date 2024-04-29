from transformers import LlamaForCausalLM, AutoTokenizer

try:
    # Load the model
    model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Load the tokenizer using AutoTokenizer to correctly identify the type
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    while True:
        # Define a prompt or input text
        input_text = input("User: ")

        # Generate text using the model
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids=input_ids, max_length=50)

        # Decode and print the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("AI Assistant: ", generated_text)

except Exception as e:
    print(f"An error occurred: {e}")