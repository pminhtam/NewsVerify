import pandas as pd
from neomodel import StructuredNode, StringProperty, RelationshipTo, RelationshipFrom, config
from sider_obj import Drug_ID, Side_Effect
config.DATABASE_URL = 'bolt://neo4j:12345678@100.64.241.89:7687'

"""
class Book(StructuredNode):
    title = StringProperty(unique_index=True)
    author = RelationshipTo('Author', 'AUTHOR')

class Author(StructuredNode):
    name = StringProperty(unique_index=True)
    books = RelationshipFrom('Book', 'AUTHOR')


bb = Book.nodes.get_or_none(title='Harry potter and the..')
print(bb)
# harry_potter = Book(title='Harry potter and the..').save()
# rowling =  Author(name='J. K. Rowling').save()
# harry_potter.author.connect(rowling)
"""

drug_names = pd.read_csv("../data/drug_names.tsv",sep="	",header=None)
# print(drug_names)
"""neo4j
MATCH (n:Drug_ID) DETACH DELETE n
"""
# """
for idx, drug in drug_names.iterrows():
    # print("xxxxxxxxxxxxx  ",drug)
    # print("zzzzzzzzzzzzz  ",drug[0])
    # print("zzzzzzzzzzzzz  ",drug[1])
    compound_id = drug[0]
    name = drug[1]
    print(compound_id, ":   name : ",name)
    drug = Drug_ID.nodes.get_or_none(compound_id=compound_id)
    if drug is None:
        Drug_ID(compound_id=compound_id, name=name).save()
# """
effects = pd.read_csv("../data/meddra_all_se.tsv", sep="	", header=None)
# print(effects)

for idx, eff in effects.iterrows():
    print(idx)
    compound_id = eff[0]
    # print(compound_id)
    drug = Drug_ID.nodes.get_or_none(compound_id=compound_id)
    # print(drug)
    if drug is None:
        print(compound_id)
    umls_id = eff[4]
    name = eff[5]
    side_eff = Side_Effect.nodes.get_or_none(umls_id=umls_id)
    if side_eff is None:
        side_eff = Side_Effect(umls_id=umls_id, name=name).save()
    drug.effect.connect(side_eff)
