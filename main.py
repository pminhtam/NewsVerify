# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor

from crawl_evidences.crawl_evidences import *

nli_labels = ['contradiction', 'neutral', 'entailment']
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--encoder', type=str, default="./output/cross-encoder-distilroberta-base", help="path to encoder model")
    parser.add_argument('--nli', type=str, default="facebook/bart-large-mnli", help="path to nli model")
    parser.add_argument('--tokenizer', type=str, default="facebook/bart-large-mnli", help="path to tokenizer")
    parser.add_argument('--data_path', type=str, default="./fever_2018/train.jsonl", help="path to  data_path")
    parser.add_argument('--num_claims', type=int, default=100, help='number of claims')
    parser.add_argument('--verbose', type=int, default=0, help='0 : not print any details, 1 : print summary of each claim'
                                                               ' and 2 : print all details of each Evidence')
    return parser.parse_args()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = CrossEncoder(args.encoder)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # scores = model.predict([["Donald Trump is the 1st president", "Obama is the first president"],
    #                         ["Donald Trump is the 1st president", "Openness, understanding key to harmonious neighborhoods"]])
    # url = 'http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/'
    # res = parse_passages(url)

    # claim = "Roman Atwood is a content creator"
    # claim = 'Donald Trump is the richest president in US.'
    # claims = ["World-renowned singer Celine Dion died or revealed new personal health developments in late July 2023."]

    claims = load_fever_data(data_path=args.data_path, num_claims= args.num_claims)
    # print(claims)
    # exit()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for claim in claims:
        value = process_claim(claim, encoder_model, nli_model, tokenizer,device=device, verbose = args.verbose)
        if value == "tp":
            tp += 1
        elif value == "fp":
            fp += 1
        elif value == "tn":
            tn += 1
        elif value == "fn":
            fn += 1
    print("****************************************************************")
    print("********************************************************************************************************************************")
    print("************************************************************************************************************************************************************************************************")
    print("tp  : ",tp)
    print("fp  : ",fp)
    print("tn  : ",tn)
    print("fn  : ",fn)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
