import json
import numpy as np
from tqdm import tqdm

np.random.seed(100)

def load_fever_data(data_path = '../fever_2018/train.jsonl', num_claims = 100):
    claims = []

    with open(data_path, 'r') as fp:
        for sample in tqdm(fp):
            sample = json.loads(sample)
            claim = {"claim":sample['claim'], "verifiable":sample['verifiable'],
                     "label":sample['label']}
            claims.append(claim)
    if num_claims < len(claims):
        selected_claims = np.random.choice(claims, num_claims)
        return selected_claims
    else:
        return claims