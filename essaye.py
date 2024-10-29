from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Nom du modèle
model_name = "gpt2"  # Remplace par "EleutherAI/gpt-neo-125M" si tu veux utiliser GPT-Neo

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=50):
    # Tokeniser le prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Générer du texte
    output = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # Décoder et retourner le texte généré
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
if __name__ == "__main__":
    prompt = "Une histoire fascinante commence par"
    generated_text = generate_text(prompt)
    print("Texte généré : ", generated_text)