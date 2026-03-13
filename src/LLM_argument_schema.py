from typing import List, Set
# Third party Imports
from pydantic import BaseModel, Field, model_validator


class ArgumentUnit(BaseModel):
    """Represents a single atomic unit extracted verbatim from input discussion. Each unit represents a clear,
     single, clear argumentative intent."""
    reason: str = Field(..., description="The concise reasoning of the argument unit explaining its logical role or intent"
                                         " (example: \"evidence supporting X.\" or \"rebuts Y.\"."
                        )
    id: int = Field(...,
                    description="A sequential identifier for the argument unit, assigned in order of appearance in "
                                "the input text."
                    )
    text: str = Field(..., description="The textual content of the argument unit taken verbatim from the input text and self-containing a single, clear argumentative intent.")


class ArgumentUnits(BaseModel):
    """Represents a collection of units of a discussion."""
    argument_units: List[ArgumentUnit] = Field(default_factory=list,
                                               description="The list of identified units taken verbatim from the discussion."
                                               )

    @model_validator(mode='after')
    def validate_units(self) -> 'ArgumentUnits':
        if len(self.argument_units) < 2:
            raise ValueError("At least two argument units are required.")

        for unit in self.argument_units:
            # if unit.reason is None or unit.reason == "":
            #     raise ValueError("Reason for unit must be provided.")
            if unit.text is None or (len(unit.text) <= 1):
                raise ValueError("Text must be provided.")

        return self


class ArgumentRelation(BaseModel):
    """Represents a relationship between two argument units."""
    source_id: int = Field(..., description="The unit_id of the source argument unit which supports or attacks another unit.")
    target_id: int = Field(..., description="The unit_id of the target argument unit which is supported or attacked by another unit.")
    type: str = Field(..., description="The type of relationship ('support' or 'attack') where support indicates that "
                                       "the source accepts/justifies/defends the target and attack indicates that the "
                                       "source rejects/questions/contradicts the target.")


class ArgumentRelations(BaseModel):
    """Represents a collection of relationships between two argument units."""
    relations: List[ArgumentRelation] = Field(default_factory=list,
                                              description="The list of identified relationships between two argument units."
                                              )

    @model_validator(mode='after')
    def validate_relations(self) -> 'ArgumentRelations':
        if len(self.relations) < 1:
            raise ValueError('The argument units must have at least one relation.')

        for relation in self.relations:
            if relation.type in ["support", "attack"]:
                continue
            else:
                raise ValueError("The relation type must be either 'support' or \'attack\'.")

        return self


class ArgumentGraph(BaseModel):
    """
    The root model for the complete structured argument graph.
    includes a required 'reasoning' field instead of 'unit_type' and 'summary'.
    """

    argument_units: List[ArgumentUnit] = Field(..., description="A list of all argument units in the text.")
    relations: List[ArgumentRelation] = Field(..., description="A list of all relations connecting the units.")

    @model_validator(mode='after')
    def check_graph_constraints(self) -> 'ArgumentGraph':
        """
        Validation constraints:
        1. Graph Complexity: At least 2 units and 1 relation.
        2. Graph Connectivity: Every unit must participate in at least one relation.
        """

        # --- 1. Check Graph Complexity ---
        if len(self.argument_units) < 2:
            raise ValueError("ArgumentGraph must contain at least two Argument Units.")

        if len(self.relations) < 1:
            raise ValueError("ArgumentGraph must contain at least one Argument Relation.")

        for unit in self.argument_units:
            if unit.reason is None or unit.reason == "":
                raise ValueError("Reason for unit must be provided.")
            if unit.text is None or (len(unit.text) <= 1):
                raise ValueError("Text must be provided.")

        # --- 2. Check Graph Connectivity ---
        unit_ids: Set[int] = {unit.id for unit in self.argument_units}
        participating_ids: Set[int] = set()
        for relation in self.relations:
            participating_ids.add(relation.source_id)
            participating_ids.add(relation.target_id)

            if relation.type in ["support", "attack"]:
                continue
            else:
                raise ValueError("The relation type must be either 'support' or \'attack\'.")

            # # check source_id > target_id [directionality]
            # if relation.source_id > relation.target_id:
            #     continue
            # else:
            #     ValueError(f"Relation source should have a higher ID than its target.")
            #
        # dangling_source = participating_ids - unit_ids
        # if dangling_source:
        #     raise ValueError(
        #         f"Relation source/target IDs are not defined in units list. Undefined IDs: {dangling_source}."
        #         )

        # unconnected_ids = unit_ids - participating_ids
        # if unconnected_ids:
        #     raise ValueError(
        #         f"All Argument Units must participate in at least one relation. Unconnected "
        #         f"units found: {unconnected_ids}. Please ensure these units are connected."
        #         )

        return self
