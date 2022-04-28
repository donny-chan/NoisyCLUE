import torch

def get_features(examples, label_list, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    texts = [x['text'] for x in examples]
    features = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation='longest_first',
        padding='max_length',
        return_tensors='pt')
    features['labels'] = torch.tensor([label_map[x['label']] for x in examples])
    return features
