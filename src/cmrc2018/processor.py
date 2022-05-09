#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : mrc.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/5/29 16:39
@version: 1.0
@desc  : 
"""
import collections
from typing import List

from tqdm import tqdm

from . import tokenization
from utils import load_jsonl
from .feature import SquadExample


def customize_tokenizer(text: str, do_lower_case=False) -> List[str]:
    # tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenization._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(
            c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


def read_squad_examples(input_file: str, has_labels: bool, num_examples=None) -> List[SquadExample]:
    """Read a SQuAD json file into a list of SquadExample."""

    print(f'Start reading examples from {input_file}')
    query_data = load_jsonl(input_file, num_examples)

    examples = []
    last_context = None

    for sample in tqdm(query_data):
        context = sample["context"]
        if context != last_context:
            # Get doc_tokens.
            # Continuous English characters will be one token.
            raw_doc_tokens = customize_tokenizer(context, do_lower_case=False)
            doc_tokens = []
            char_to_word_offset = []
            k = 0
            temp_word = ""
            for c in context:
                if tokenization._is_whitespace(c):
                    char_to_word_offset.append(k - 1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            assert k == len(raw_doc_tokens)

        query_id = sample["id"]
        question_text = sample["question"]
        start_position = None
        end_position = None
        orig_answer_text = None

        if has_labels:
            # Correct start and end position of the answer.
            answer = sample["answers"][0]
            orig_answer_text = answer["text"]

            if orig_answer_text not in context:
                print("Could not find answer")
            else:
                answer_offset = context.index(orig_answer_text)
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]

                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = "".join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = "".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    continue

        example = SquadExample(
            qas_id=query_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)

    print('read_squad_examples complete')
    return examples


def convert_query_to_token(tokenizer, query, max_length):
    context_output = tokenizer.encode(query, add_special_tokens=False)
    # 截取长度
    query_tokens = context_output.tokens
    if len(query_tokens) > max_length:
        query_tokens = query_tokens[0:max_length]
    return query_tokens


def _improve_answer_span(
    doc_tokens: list, input_start: int, input_end: int, tokenizer,
    orig_answer_text: str) -> tuple:
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans: list, cur_span_index: int, position: int) -> bool:
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(
    examples: List[SquadExample], 
    tokenizer, 
    has_labels: bool,
    doc_stride: int=128, 
    max_seq_length: int=512, 
    max_query_length: int=None) -> List[dict]:
    """Loads a data file into a list of `InputBatch`s."""
    all_features = []
    unique_id = 1000000000

    for example_index, example in tqdm(enumerate(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)
        query_tokens = query_tokens[:max_query_length]

        # Build index mapping, tokenize each doc_token individually
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) == 1:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_tokens[0])
            else:
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if has_labels:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # CMRC2018 has very long contexts!
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # Build a feature for each span.
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            token_to_orig_map = {}
            token_is_max_context = {}

            '''
            tokens:             [CLS] the query tokens [SEP] the doc tokens [SEP] [0 ... 0]
            segment_ids:        [0, 0, ...                0] [1, 1, ...        1] [0 ... 0]
            input_mask:         [1, 1, ...                                     1] [0 ... 0]
            input_span_mask:    [1]   [0, 0, ...          0] [1, 1, ...        1] [0 ... 0]
            '''

            tokens = ["[CLS]"] + query_tokens + ['[SEP]']
            segment_ids = [0] * len(tokens)                  # token_type_ids
            # input_span_mask = [1] + [0] * (len(tokens) - 1)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                # input_span_mask.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            # input_span_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            input_mask = [1] * len(input_ids)  # attention_mask

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                # input_span_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert len(input_span_mask) == max_seq_length

            feature = {
                'unique_id': unique_id,
                'example_index': example_index,
                'doc_span_index': doc_span_index,
                'tokens': tokens,
                'token_to_orig_map': token_to_orig_map,
                'token_is_max_context': token_is_max_context,
                # Below is used passed to model
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'segment_ids': segment_ids,
            }
            if has_labels:
                start_position = None
                end_position = None
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

                feature.update({
                    'start_position': start_position,
                    'end_position': end_position,
                })

            all_features.append(feature)
            unique_id += 1
            
            if example_index < 1:
                print(f"*** Example {example_index} ***")
                print(example)
    return all_features

