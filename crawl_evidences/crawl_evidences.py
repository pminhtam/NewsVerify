import time
import torch
from concurrent.futures import ThreadPoolExecutor
from .openai_api import *
from .parser import *
from .models_compare import *
from .load_data import *

nli_labels = ['contradiction', 'neutral', 'entailment']


def generate_prompt(claim, evidences):
    prompt = "Justify the claim given the evidences. Point out the most 5 relevant evidences leading to the conclusion with the format: [Original evidence number]: [explanation]. The claim and evidences are in triple backticks"
    prompt += "\nClaim: '''{}'''".format(claim)
    prompt += "\nEvidences:\n"
    for i in range(len(evidences)):
        prompt += "{}.'''{}'''\n".format(i + 1, evidences[i])

    return prompt




def get_evidences(claim,model, k=20):
    print("Crawl the evidence links from Google Search")
    start = time.perf_counter()
    links = crawl(claim)
    end = time.perf_counter()
    print("It took {:.2f} second(s) to finish.".format(end - start))

    print("Crawl the evidences")
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        results = executor.map(parse_passages, links)
    end = time.perf_counter()

    print("It took {:.2f} second(s) to finish.".format(end - start))

    candidates = []
    for result in results:
        if len(result) > 0:
            candidates.extend(result)

    scores = model.predict([[claim, x] for x in candidates])

    best_candidates = [candidates[i] for i in np.argsort(scores)[::-1][:k]]
    return best_candidates

def process_claim(claim_dict, encoder_model, nli_model, tokenizer,device="cpu", verbose=0):
    if type(claim_dict) is dict:
        claim = claim_dict['claim']
        verifiable = claim_dict['verifiable']
        label = claim_dict['label']
    else:
        claim = claim_dict
        verifiable = ""
        label = ""
    # print("Claim : ", claim)
    best_evidences = get_evidences(claim, encoder_model)
    # print("Evidences: ")
    nli_labels_arr = []
    num_contradiction = 0
    num_neutral = 0
    num_entailment = 0
    for e in best_evidences:
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
            print("============================================================")
            print("Claim : ", claim)
            print("++++++++++++++++")
            print("Evidence  :  :   :  ", e)
            print("++++++++++++++++")
            print("entail_contradiction_logits \t  : ", entail_contradiction_logits)
            print("probs   \t \t \t : ", probs)
            print("nli_labels   \t \t: ", nli_labels_argmax)

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
        print("++++++++++++++++")
        print("Claim : ", claim)
        print("verifiable : ",verifiable)
        print("label : ", label)
        print("NLI labels ", nli_labels_arr)
        print("NLI num_contradiction ", num_contradiction)
        print("NLI num_neutral ", num_neutral)
        print("NLI num_entailment ", num_entailment)
        print("****************************************************************")
        print("****************************************************************")
        print("****************************************************************")
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
