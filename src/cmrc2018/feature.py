from typing import List, Dict


class SquadExample:
    """
    A single training/test example for simple sequence classification.

    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id: str,
                 question_text: str,
                 doc_tokens: List[str],
                 orig_answer_text: str=None,
                 start_position: int=None,
                 end_position: int=None):
        self.query_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = f"qas_id: {self.query_id}, question_text: {self.question_text}, doc_tokens: [{' '.join(self.doc_tokens[:20])} ... {' '.join(self.doc_tokens[-20:])}]"
        if self.start_position:
            s += f", start_position: {self.start_position}"
        if self.end_position:
            s += f", end_position: {self.end_position}"
        return s


# class Feature:
#     """A single set of features of data."""
#     def __init__(self,
#                  unique_id: int,
#                  example_index: int,
#                  doc_span_index: int,
#                  tokens: List[str],
#                  token_to_orig_map: Dict[int, int],
#                  token_is_max_context: Dict[int, bool],
#                  input_ids: List[int],
#                  input_mask: List[int],
#                  segment_ids: List[int],
#                 #  input_span_mask,
#                  start_position: List[int]=None,
#                  end_position: List[int]=None):
#         self.unique_id = unique_id
#         self.example_index = example_index
#         self.doc_span_index = doc_span_index
#         self.tokens = tokens
#         self.token_to_orig_map = token_to_orig_map
#         self.token_is_max_context = token_is_max_context
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         # self.input_span_mask = input_span_mask
#         self.start_position = start_position
#         self.end_position = end_position
