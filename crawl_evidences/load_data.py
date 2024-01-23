import json
import numpy as np
from tqdm import tqdm
import sys

np.random.seed(100)

def load_fever_data(data_path = './fever_2018/train.jsonl', num_claims = 100):
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

import pandas as pd

def preprocess_wiki(data_path = '../fever_2018/train.jsonl', wiki_path="../fever_2018/wiki-pages/" ):
    claims = pd.read_json(path_or_buf=data_path, lines=True)
    # claims = [json.loads(data_path) for jline in jsonl_content.splitlines()]
    wiki = pd.read_json(path_or_buf=wiki_path + "/wiki-001.jsonl", lines=True)
    # claims = json.load(open(data_path))
    print(claims["evidence"].get(0))
    print(wiki)
    evidences_arr = []
    for evidence in claims["evidence"]:
        # print(evidence[0])
        [evidences_arr.append(evi) for evi in evidence[0]]
    # print(evidences_arr)
    evidences_df = pd.DataFrame(evidences_arr,
                       columns=['id', 'evidence', 'url', 'sentence'])
    df3 = pd.merge(evidences_df, wiki, on=['id'], how='inner')
    print(df3)

sys.path.append('../../naacl2018-fever/src/')
from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.data_set import DataSet as FEVERDataSet
from retrieval.sentence import FEVERSentenceFormatter
from rte.riedel.data import FEVERLabelSchema
def load_fever_data_with_wiki_evidence(data_path = '../fever_2018/train.jsonl', db_path='../../naacl2018-fever/data/fever/fever.db'):
    jlr = JSONLineReader()

    docdb = FeverDocDB(db_path)

    formatter = FEVERSentenceFormatter(set(docdb.get_doc_ids()), FEVERLabelSchema())
    ds = FEVERDataSet(data_path, reader=jlr, formatter=formatter)

    ds.read()

    # for instance in tqdm(ds.data):
    for instance in ds.data:
        if instance is None:
            continue
        # print(instance)
        try:
            evidences = []
            # print("====================claim ============================ ", instance)
            # for page in set([ev[0] for ev in instance['evidence']]):
            for page in set([ev[0][0] for ev in instance['evidence']]):
                # claim = instance['claim'].strip()
                # print(page)
                # print("................................................................")
                # paragraph = docdb.get_doc_text(page[0]).split(" . ")[page[1]-1]
                # paragraph = docdb.get_doc_text(page[0])
                paragraph = docdb.get_doc_text(page)
                # tokenized_paragraph = _wiki_tokenizer.tokenize(paragraph)
                # print(paragraph)
                evidences.append(paragraph)
                # evidences = set([ev[1] for ev in instance['evidence'] if ev[0] == page])
                # print(evidences)
            yield instance,evidences
        except Exception as e:
            # print(instance)
            # print(e)
            continue
if __name__ == '__main__':
    # preprocess_wiki()
    # data = []
    data_supports = []
    data_refutes = []
    num_supports = 0
    num_refutes = 0
    num_normals = 0
    data = load_fever_data(data_path = '../fever_2018/train.jsonl', num_claims = 100000000)
    for d in data:
        if d['label'] == 'SUPPORTS':
            # data_supports.append({'claim': instance['claim'],'evidence_locate': instance['evidence'],'evidence': evidences})
            # data_supports.append({'instance': instance,'evidence': evidences})
            num_supports += 1
        elif d['label'] == 'REFUTES':
            # data_refutes.append({'instance': instance,'evidence': evidences})
            num_refutes += 1
        else:
            print(d['label'])
            num_normals += 1

    print(num_supports)
    print(num_refutes)
    print(num_normals)
    exit()

    for (instance,evidences) in load_fever_data_with_wiki_evidence():
        # print("================================================================")
        # print(instance['label_text'])
        if instance['label_text'] == 'SUPPORTS':
            # data_supports.append({'claim': instance['claim'],'evidence_locate': instance['evidence'],'evidence': evidences})
            # data_supports.append({'instance': instance,'evidence': evidences})
            num_supports += 1
        elif instance['label_text'] == 'REFUTES':
            # data_refutes.append({'instance': instance,'evidence': evidences})
            num_refutes += 1
        else:
            print(instance['label_text'])
            num_normals += 1
        # data.append({'claim': instance['claim'],'evidence_locate': instance['evidence'],'evidence': evidences})
        # print("instance  : ",instance)
        # print("evidences  : ", evidences)
    # json.dump(data,open("fever_data.json",'w'))
    # json.dump(data_supports,open("fever_data_supports.json",'w'))
    # json.dump(data_refutes,open("fever_data_refutes.json",'w'))
    print(num_supports)
    print(num_refutes)
    print(num_normals)
