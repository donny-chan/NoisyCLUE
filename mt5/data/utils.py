import torch

def get_features(examples, label_list, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    verbalizer = {'0': 'non_equivalent', '1': 'equivalent'}

    features = []
    texts = [x['text'] for x in examples]
    features = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation='longest_first',
        padding='max_length',
        return_tensors='pt')
    features['labels'] = torch.tensor([verbalizer[x['label']] for x in examples])
    return features
