from neomodel import StructuredNode, StringProperty, RelationshipTo, RelationshipFrom, config


class Drug_ID(StructuredNode):
    compound_id = StringProperty(unique_index=True)
    name = StringProperty()
    effect = RelationshipTo('Side_Effect', 'effect')

class Side_Effect(StructuredNode):
    umls_id = StringProperty(unique_index=True)
    name = StringProperty()
    drug = RelationshipFrom('Drug_ID', 'effect')
