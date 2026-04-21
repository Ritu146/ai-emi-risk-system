from transformers import pipeline

# Load once
generator = pipeline("text-generation", model="gpt2")

def generate(query, context):
    try:
        # Convert context list → clean text
        context_text = " ".join(context)

        prompt = f"""
You are a credit risk expert.

Context:
{context_text}

Question:
{query}

Answer in 2-3 clear sentences:
"""

        output = generator(prompt, max_length=100, num_return_sequences=1)

        answer = output[0]['generated_text']

        # Remove prompt from output
        answer = answer.replace(prompt, "").strip()

        return answer

    except Exception as e:
        return "Error generating answer"
