import time
import torch
from concurrent.futures import ThreadPoolExecutor
from .openai_api import *
from .parser import *
from .models_compare import *
from .load_data import *

nli_labels = ['contradiction', 'neutral', 'entailment']


def process_claim(claim_evidence, nli_model, tokenizer,device="cpu", verbose=0):
    claim_dict = claim_evidence[0]
    evidence_list = claim_evidence[1]
    if type(claim_dict) is dict:
        claim = claim_dict['claim']
        label = claim_dict['label_text']
    else:
        claim = claim_dict
        label = ""
    # print("Claim : ", claim)
    # print("Evidences: ")
    nli_labels_arr = []
    num_contradiction = 0
    num_neutral = 0
    num_entailment = 0
    for e in evidence_list:
        # print(e)
        premise = e
        hypothesis = claim
        # run through model pre-trained on MNLI
        x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                             truncation_strategy='only_first')
        logits = nli_model(x.to(device))[0]
        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true
        entail_contradiction_logits = logits
        probs = entail_contradiction_logits.softmax(dim=1)
        # prob_label_is_true = probs[:, 1]
        nli_labels_argmax = nli_labels[torch.argmax(probs)]
        if verbose >= 2:
            print("=================================================================================")
            print("Claim :  \t \t ", claim)
            print("++++++++++++++++")
            print("Evidence :  \t \t  ", e)
            print("++++++++++++++++")
            print("entail_contradiction_logits : \t \t  ", entail_contradiction_logits.detach().cpu().numpy())
            print("probs : \t \t", probs.detach().cpu().numpy())
            print("nli_labels :  \t \t  ", nli_labels_argmax)

        if nli_labels_argmax == "contradiction":
            num_contradiction += 1
        elif nli_labels_argmax == "neutral":
            num_neutral += 1
        elif nli_labels_argmax == "entailment":
            num_entailment += 1
        nli_labels_arr.append(nli_labels_argmax)
        # print("prob_label_is_true   : ", prob_label_is_true)
        del probs, entail_contradiction_logits, logits, x
    if verbose >=1:
        print("=================================================================================")
        print("Claim : ", claim)
        print("Label : ", label)
        print("NLI labels ", nli_labels_arr)
        print("NLI num_contradiction ", num_contradiction)
        print("NLI num_neutral ", num_neutral)
        print("NLI num_entailment ", num_entailment)
        print("****************************************************************")
        print("*********************************************************************************************")
        print("********************************************************************************************************************************")
    if label == "SUPPORTS":
        if num_entailment > num_contradiction:
            return "tp"
        else:
            return "fp"
    elif label == "REFUTES":
        if num_entailment >= num_contradiction:
            return "fn"
        else:
            return "tn"
    return 0
import argparse
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

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = CrossEncoder(args.encoder)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    claims = load_fever_data_with_wiki_evidence(data_path=args.data_path)
    # print(claims)
    # exit()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for claim_evidence in claims:
        # print(claim_evidence)
        value = process_claim(claim_evidence, nli_model, tokenizer,device=device, verbose = args.verbose)
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
