import time
import torch
import json
import os
import yaml
from concurrent.futures import ThreadPoolExecutor
from .openai_api import *
from .parser import *
from .models_compare import *
# from .load_data import *

nli_labels = ['contradiction', 'neutral', 'entailment']

if os.path.isfile('config.yaml'):
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        openai.api_key = cfg["openapi_key"]

def generate_prompt(claim, evidences):
    prompt = "Justify the claim given the evidences. Write only the answer in [Fake, True] in the first line. Then, point out the most 5 relevant evidences leading to the conclusion with the format: [Original evidence number]: '''[Original evidence]'''. Explanation: [explanation]."
    prompt += "\nClaim: '''{}'''".format(claim)
    prompt += "\nEvidences:\n"
    for i in range(len(evidences)):
        prompt += "{}.'''{}'''\n".format(i + 1, evidences[i])

    prompt += "Example: \nTrue\n1: '''Qatar's Sheikh Jassim bin Hamad Al Thani has won the race to buy Manchester United'''. Explanation: This indicates that Sheikh Jassim bin Hamad Al Thani has emerged as the winner in the race.".format(
        claim)

    return prompt

global iii
iii = 0
def process_claim(claim_evidence, device="cpu", verbose=0):
    global iii
    claim_dict = claim_evidence['instance']
    evidence_list = claim_evidence['evidence']
    if type(claim_dict) is dict:
        claim = claim_dict['claim']
        label = claim_dict['label_text']
    else:
        claim = claim_dict
        label = ""

    if label == "REFUTES":
    # if label == "SUPPORTS":
        return
    iii += 1
    if iii < 241  :
        return
    # print("Claim : ", claim)
    # print("Evidences: ")
    nli_labels_arr = []
    num_contradiction = 0
    num_neutral = 0
    num_entailment = 0

    prompt = generate_prompt(claim, evidence_list)

    answer = get_completion(prompt)

    result = answer.split("\n")[0].lower()

    print(result)

    if verbose >=1:
        print("=================================================================================")
        print("Claim : ", claim)
        print("Label : ", label)
        print("answer : ", answer)
        print("****************************************************************")
        print("*********************************************************************************************")
        print("********************************************************************************************************************************")
    if label == "SUPPORTS":
        if result == "true":
            return "tp"
        elif result == "fake":
            return "fp"
        else:
            return 'same'
    elif label == "REFUTES":
        if result == "true":
            return "fn"
        elif result== "fake":
            return "tn"
        else:
            return 'same'
    return 0
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    # parser.add_argument('--encoder', type=str, default="./output/cross-encoder-distilroberta-base", help="path to encoder model")
    parser.add_argument('--data_path', type=str, default="./fever_2018/train.jsonl", help="path to  data_path")
    parser.add_argument('--num_claims', type=int, default=100, help='number of claims')
    parser.add_argument('--verbose', type=int, default=0, help='0 : not print any details, 1 : print summary of each claim'
                                                               ' and 2 : print all details of each Evidence')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder_model = CrossEncoder(args.encoder)


    # claims = load_fever_data_with_wiki_evidence(data_path=args.data_path)
    with open("utils/fever_data_1k.json", 'r') as f:
        claims = json.load(f)
    # print(claims)
    # exit()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    same = 0
    for claim_evidence in claims:
        # print(claim_evidence)
        value = process_claim(claim_evidence, device=device, verbose = args.verbose)
        if value == "tp":
            tp += 1
        elif value == "fp":
            fp += 1
        elif value == "tn":
            tn += 1
        elif value == "fn":
            fn += 1
        elif value == "same":
            same += 1
    print("****************************************************************")
    print("****************************************************************")
    print("****************************************************************")
    print("tp  : ",tp)
    print("fp  : ",fp)
    print("tn  : ",tn)
    print("fn  : ",fn)
    print("same  : ",same)
