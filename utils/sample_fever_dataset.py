import json
import random

with open("../crawl_evidences/fever_data_supports.json",'r') as f:
    data_supports = json.load(f)

with open("../crawl_evidences/fever_data_refutes.json",'r') as f:
    data_refutes = json.load(f)

data_supports_500 = random.choices(data_supports, k=500)
data_refutes_500 = random.choices(data_refutes, k=500)
# print(type(data_supports))
print(len(data_supports_500))
print(len(data_refutes_500))

data_1k = data_refutes_500 + data_supports_500
print(len(data_1k))
json.dump(data_1k, open("fever_data_1k.json", 'w'))
