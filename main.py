# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import torch
from concurrent.futures import ThreadPoolExecutor

from crawl_evidences.crawl_evidences import *

nli_labels = ['contradiction', 'neutral', 'entailment']


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = CrossEncoder('./output/cross-encoder-distilroberta-base')
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(device)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    # scores = model.predict([["Donald Trump is the 1st president", "Obama is the first president"],
    #                         ["Donald Trump is the 1st president", "Openness, understanding key to harmonious neighborhoods"]])
    # url = 'http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/'
    # res = parse_passages(url)

    # claim = "Roman Atwood is a content creator"
    # claim = 'Donald Trump is the richest president in US.'
    # claims = ["World-renowned singer Celine Dion died or revealed new personal health developments in late July 2023."]

    claims = load_fever_data(num_claims= 10)
    # print(claims)
    # exit()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for claim in claims:
        value = process_claim(claim, encoder_model, nli_model, tokenizer)
        if value == "tp":
            tp += 1
        elif value == "fp":
            fp += 1
        elif value == "tn":
            tn += 1
        elif value == "fn":
            fn += 1
    print("****************************************************************")
    print("****************************************************************")
    print("****************************************************************")
    print("tp  : ",tp)
    print("fp  : ",fp)
    print("tn  : ",tn)
    print("fn  : ",fn)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
