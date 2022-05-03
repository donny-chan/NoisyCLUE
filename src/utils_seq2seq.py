from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer

from data.afqmc import AfqmcSeq2SeqDataset


def _get_dataset(file: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    return AfqmcSeq2SeqDataset(file, phase, **kwargs)


def get_dataset(data_dir: Path, phase: str, **kwargs) -> AfqmcSeq2SeqDataset:
    return _get_dataset(data_dir / f'{phase}.json', phase, **kwargs)


def predict(
    trainer: Seq2SeqTrainer, 
    dataset: AfqmcSeq2SeqDataset, 
    output_dir: Path,
    args: Namespace,
    ) -> tuple:
    '''
    Return (predictions: list, result: dict)
    '''
    def get_test_acc(preds: torch.Tensor, labels: torch.Tensor) -> float:
        '''
        preds: (#examples, seq_len)
        labels: (#examples, seq_len)
        '''
        assert preds.size() == labels.size()
        eq = torch.eq(preds, labels)
        count = len(labels)
        correct = 0
        for i in range(count):
            if eq[i].all():
                correct += 1
        return correct / count

    def collate_fn(examples: list):
        '''Each element in `examples` is a dict from str to list.'''
        batch = {}
        for key in examples[0].keys():
            batch[key] = torch.tensor([x[key] for x in examples])
        return batch

    def prediction_step(batch: dict) -> tuple:
        return trainer.prediction_step(trainer.model, inputs=batch, 
            prediction_loss_only=False)

    trainer.model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    total_loss = 0
    acc = 0
    num_steps = 0
    all_preds = []
    for batch in dataloader:
        loss, logits, labels = prediction_step(batch)
        logits = logits[0]
        preds = torch.argmax(logits, dim=2)  # (N, seq_len, vocab_size) -> (N, seq_len)
        all_preds += list(preds.cpu().numpy())

        total_loss += loss.item()
        acc += get_test_acc(preds, labels)
        num_steps += 1

    # Get result
    acc /= num_steps
    loss = total_loss / num_steps
    result = {
        'acc': acc,
        'loss': loss,
    }
    return all_preds, result
