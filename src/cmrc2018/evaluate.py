from collections import namedtuple, OrderedDict, defaultdict
import math
from typing import List, Dict

from .data import SquadExample
from . import tokenization
from utils import dump_json


Logits = namedtuple('Logits', ['unique_id', 'start_logits', 'end_logits'])


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if False:
            print(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if False:
            print("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in enumerate(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if False:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if False:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def write_predictions(
    all_examples: List[SquadExample], 
    all_features: List[dict],
    all_logits: List[Logits], 
    file_preds: str, 
    file_nbest: str,
    do_lower_case: bool=True, 
    max_answer_length: int=128,
    n_best_size: int=20,
    ) -> Dict[str, str]:
    '''
    Write final predictions to the json file and log-odds of null if needed.

    Return dict: query_id -> str answer.
    '''
    print("Writing predictions to: %s" % (file_preds))
    print("Writing nbest to: %s" % (file_nbest))

    example_index_to_features = defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature['example_index']].append(feature)

    unique_id_to_logits = {}
    for logits in all_logits:
        unique_id_to_logits[logits.unique_id] = logits

    _PrelimPrediction = namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_preds = OrderedDict()
    all_nbest_preds = OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_preds = []

        # Get preliminary predictions
        for (feature_index, feature) in enumerate(features):  # multi-trunk
            # For each featuresm, loop different combinations of 
            # the top-`n_best_size` start and end indices
            logits = unique_id_to_logits[feature['unique_id']]
            start_indexes = _get_best_indexes(logits.start_logits, n_best_size)
            end_indexes = _get_best_indexes(logits.end_logits, n_best_size)

            def is_valid_pred(start_index: int, end_index: int) -> bool:
                if start_index >= len(feature['tokens']):
                    return False
                if end_index >= len(feature['tokens']):
                    return False
                if start_index not in feature['token_to_orig_map']:
                    return False
                if end_index not in feature['token_to_orig_map']:
                    return False
                if not feature['token_is_max_context'].get(start_index, False):
                    return False
                if end_index < start_index:
                    return False
                if end_index - start_index + 1 > max_answer_length:
                    return False
                return True

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions
                    # (e.g., predict that the start of the span is in the 
                    # question). We throw out all invalid predictions.
                    if is_valid_pred(start_index, end_index):
                        prelim_preds.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=logits.start_logits[start_index],
                                end_logit=logits.end_logits[end_index]))

        # Sort by descending logits (sum of start and end logits)
        prelim_preds = sorted(
            prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", 
            ["text", "start_logit", "end_logit", "start_index", "end_index"])

        seen_predictions = {}
        n_best_preds = []
        # get n-best predictions
        for pred in prelim_preds:
            if len(n_best_preds) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature['tokens'][pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature['token_to_orig_map'][pred.start_index]
                orig_doc_end = feature['token_to_orig_map'][pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                final_text = final_text.replace(' ', '')
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            n_best_preds.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_index=pred.start_index,
                    end_index=pred.end_index))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not n_best_preds:
            n_best_preds.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0,
                start_index=-1, end_index=-1)
            )

        assert len(n_best_preds) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in n_best_preds:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        nbest_result = []
        for (i, entry) in enumerate(n_best_preds):
            # output = OrderedDict()
            # output["text"] = entry.text
            # output["probability"] = probs[i]
            # output["start_logit"] = entry.start_logit
            # output["end_logit"] = entry.end_logit
            # output["start_index"] = entry.start_index
            # output["end_index"] = entry.end_index
            # nbest_result.append(output)
            result = {
                'text': entry.text,
                'prob': probs[i],
                'start_logit': entry.start_logit,
                'end_logit': entry.end_logit,
                'start_index': entry.start_index,
                'end_index': entry.end_index
            }
            nbest_result.append(result)

        assert len(nbest_result) >= 1

        all_preds[example.query_id] = best_non_null_entry.text
        all_nbest_preds[example.query_id] = nbest_result

    dump_json(all_preds, file_preds, indent=2)
    dump_json(all_nbest_preds, file_nbest, indent=2)
    # with open(output_prediction_file, "w") as f:
    #     f.write(json.dumps(all_preds, indent=2, ensure_ascii=False) + "\n")

    # with open(output_nbest_file, "w") as f:
    #     f.write(json.dumps(all_nbest_preds, indent=2) + "\n")

    return all_preds


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def get_metrics(query_id_to_answers: dict, query_id_to_pred: dict) -> dict:
    '''
    Compute metrics for the provided predictions.
    '''
    correct = 0
    for query_id, pred in query_id_to_pred.items():
        answers = query_id_to_answers[query_id]
        for ans in answers:
            if remove_punctuation(ans) == remove_punctuation(pred):
                correct += 1
                break
    return {
        'acc': correct / len(query_id_to_pred),
    }

        


