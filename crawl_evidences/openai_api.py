import openai
import numpy as np

key = "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"

openai.api_key = key

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

