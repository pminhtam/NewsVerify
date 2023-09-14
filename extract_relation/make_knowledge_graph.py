from extract_relation.rebel_model import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3,
    }

    # Text to extract triplets from
    text = 'Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.'

    # Tokenizer text
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')

    # Generate
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        **gen_kwargs,
    )

    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Extract triplets
    for idx, sentence in enumerate(decoded_preds):
        print(f'Prediction triplets sentence {idx}')
        print(extract_triplets(sentence))

