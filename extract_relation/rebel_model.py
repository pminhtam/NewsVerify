# Load model directly
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

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
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')

    # Generate
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        **gen_kwargs,
    )

    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Extract triplets
    source = []
    target = []
    edge = []
    for idx, sentence in enumerate(decoded_preds):
        print(f'Prediction triplets sentence {idx}')
        # print(sentence)
        triplets = extract_triplets(sentence)
        for tr in triplets:
            source.append(tr['head'])
            target.append(tr['tail'])
            edge.append(tr['type'])
    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': edge})
    print(kg_df)
    # create a directed-graph from a dataframe
    G= nx.from_pandas_edgelist(kg_df, "source", "target",
                              edge_attr=True, create_using=nx.MultiDiGraph())
    plt.figure(figsize=(12,12))
    edge_labels = dict([((u, v,), d['edge'])
                        for u, v, d in G.edges(data=True)])
    print(edge_labels)
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='red', edge_cmap=plt.cm.Blues, pos = pos)
    nx.draw_networkx_edge_labels(G, pos = pos,  edge_labels = edge_labels)
    plt.show()