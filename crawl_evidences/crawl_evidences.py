from openai_api import *
from parser import *
from models_compare import *
from concurrent.futures import ThreadPoolExecutor
import time
import torch

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder('../output/cross-encoder-distilroberta-base')
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(device)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    nli_labels = [ 'contradiction', 'neutral' , 'entailment']
    # scores = model.predict([["Donald Trump is the 1st president", "Obama is the first president"],
    #                         ["Donald Trump is the 1st president", "Openness, understanding key to harmonious neighborhoods"]])
    # url = 'http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/'
    # res = parse_passages(url)

    claim = "Roman Atwood is a content creator"
    best_evidences = get_evidences(claim,model)
    print("Evidences: ")
    for e in best_evidences:
        print(e)
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
        print("entail_contradiction_logits   : ", entail_contradiction_logits)
        print("probs   : ", probs)
        print("nli_labels    : ", nli_labels[torch.argmax(probs)])
        # print("prob_label_is_true   : ", prob_label_is_true)


