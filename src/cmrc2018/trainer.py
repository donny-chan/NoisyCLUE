import argparse
from pathlib import Path
import time
from typing import List

from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from trainer import get_adamw, get_linear_scheduler
import utils
from .data import CMRC2018Dataset
from .evaluate import write_predictions, Logits, get_metrics


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

class Trainer:
    def __init__(self, model: torch.nn.Module, tokenizer, args: argparse.Namespace):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.output_dir = Path(args.output_dir)
        utils.dump_args(args, self.output_dir / 'train_args.json')
        # self.bert_config = BertConfig.from_pretrained(self.model_path, output_hidden_states=False)
        # self.model = BertForQuestionAnswering.from_pretrained(args.model_path)
        # self.tokenizer = BertTokenizer.from_pretrained(args.model_path)

    def configure_optimizers(
        self, 
        lr: float,
        num_opt_steps: int,
        weight_decay: float=0.01,
        warmup_ratio: float=0.1, 
        ):
        self.optimizer = get_adamw(self.model, lr, weight_decay)
        self.scheduler = get_linear_scheduler(
            self.optimizer, warmup_ratio, num_opt_steps)

    def train_step(self, batch: dict) -> torch.nn.Module:
        '''
        Forward step, for training.

        return loss
        '''
        outputs = self.model(**self.batch_to_inputs(batch))
        return outputs.loss
    
    def batch_to_inputs(self, batch: dict) -> dict:
        inputs = {
            'input_ids': batch['input_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda(),
            'token_type_ids': batch['segment_ids'].cuda(),
        }
        if 'start_position' in batch and 'end_position' in batch:
            inputs['start_positions'] = batch['start_position'].cuda()
            inputs['end_positions'] = batch['end_position'].cuda()
        return inputs

    def eval_step(self, batch: tuple) -> tuple:
        '''
        return: `(start_logits, end_logits)`
        '''
        outputs = self.model(**self.batch_to_inputs(batch))
        return outputs.loss, outputs.start_logits, outputs.end_logits

    def evaluate(
        self, dataset: CMRC2018Dataset, output_dir: Path, 
        is_labeled: bool, desc: str):
        '''
        Evaluate model on a dataset

        return 
        ```python
        {
            'result': {
                'loss': float,
                'acc': float,
                'time_elapsed': float,
            },
            'preds': [str, ...],  # if return_preds is True
        }
        ```
        
        '''
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size)
        self.model.eval()
        
        num_features = len(dataset)
        num_steps = len(dataloader)
        total_loss = 0

        # Result for getting predictions
        all_start_logits = []
        all_end_logits = []
        all_unique_ids = []

        start_time = time.time()
        log(f'*** Start {desc} ***')
        log(f'Batch size: {self.args.batch_size}')
        log(f'# features: {num_features}')
        log(f'# steps: {num_steps}')

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                loss, start_logits, end_logits = self.eval_step(batch)
                
                if is_labeled: total_loss += loss.item()

                # Gather data for result
                all_start_logits += start_logits.detach().cpu().tolist()
                all_end_logits += end_logits.detach().cpu().tolist()
                feature_indices = batch['idx']
                all_unique_ids += [
                    dataset.features[i]['unique_id'] for i in feature_indices]

        time_elapsed = time.time() - start_time

        log(f'*** End {desc} ***')

        all_logits = [
            Logits(
                unique_id=all_unique_ids[i], 
                start_logits=all_start_logits[i], 
                end_logits=all_end_logits[i]) for i in range(len(all_unique_ids))
        ]
        
        # Get predictions from logits
        file_preds = output_dir / 'preds.json'
        file_nbest = output_dir / 'nbest.json'
        output_dir.mkdir(exist_ok=True, parents=True)
        query_id_to_pred = write_predictions(
            dataset.examples,
            dataset.features,
            all_logits,
            file_preds=file_preds,
            file_nbest=file_nbest,
        )
        metrics = get_metrics(dataset.get_query_id_to_answers(), query_id_to_pred)

        loss = None if total_loss is None else total_loss / num_steps
        outputs = {
            'result': {
                'loss': loss,
                'acc': metrics['acc'],
                'time_elapsed': time_elapsed,
            },
            'preds': query_id_to_pred,
        }
        return outputs

    def train(self, train_dataset: CMRC2018Dataset, val_dataset: CMRC2018Dataset):
        '''Train model'''
        args = self.args
        utils.dump_args(args, self.output_dir / 'train_args.json')

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)

        # Optimizer
        num_opt_steps = len(train_dataloader) // args.grad_acc_steps * args.num_epochs
        print('Configure optimizer...')
        self.configure_optimizers(
            num_opt_steps=num_opt_steps,
            warmup_ratio=0.1,
            lr=args.lr)

        num_steps = len(train_dataloader) * args.num_epochs
        total_steps = 0
        all_train_loss = []

        start_time = time.time()

        log('*** Start training ***')
        log(f'Batch size: {args.batch_size}')
        log(f'# train examples: {len(train_dataset)}')
        log(f'# val examples: {len(val_dataset)}')
        log(f'# epochs: {args.num_epochs}')
        log(f'# steps: {num_steps}')

        for ep in range(args.num_epochs):
            self.model.train()
            self.model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                # Forward
                loss = self.train_step(batch)
                total_steps += 1
                all_train_loss.append(loss.item())

                if total_steps % args.grad_acc_steps == 0:
                    # Backward
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Results and log
                if step % args.log_interval == 0:
                    state = {
                        'ep': ep + step / len(train_dataloader),
                        'lr': self.scheduler.get_last_lr()[0],
                        'loss': loss.item(),
                        'time_elapsed': time.time() - start_time,
                    }
                    log(state)
            
            # validation
            with torch.no_grad():
                checkpoint_dir = self.output_dir / f'checkpoint-{ep}'
                eval_output = self.evaluate(
                    val_dataset, 
                    output_dir=checkpoint_dir, 
                    is_labeled=True,
                    desc="Validation")
                result = eval_output['result']
                preds = eval_output['preds']
                log('result:', result)

                # Save checkpoint and results
                self.save_checkpoint(checkpoint_dir)
                utils.dump_json(result, checkpoint_dir / f'eval_result.json')
                utils.dump_json(preds, checkpoint_dir / f'preds.json')

        log(f'*** End training ***')
        result = {
            'avg_loss': sum(all_train_loss) / total_steps,
            'time_elapsed': time.time() - start_time,
        }
        log('result:', result, '\n')

    def save_checkpoint(self, output_dir: Path):
        '''
        Save everything needing to resume training
        
        NOTE: Resuming is not supported yet
        '''
        # utils.dump_json(self.model.config, output_dir / 'config.json')
        torch.save(self.optimizer, output_dir / 'optimizer.pt')
        torch.save(self.model.state_dict(), output_dir / 'pytorch_model.bin')
        torch.save(torch.cuda.get_rng_state(), output_dir / 'rng_state.pth')
        torch.save(self.scheduler.state_dict(), output_dir / 'scheduler.pt')
        # utils.dump_json(self.args, output_dir / 'training_args')

    def load_model(self, file: Path):
        '''Load model from file'''
        state_dict = torch.load(file, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def load_best_model(self, output_dir: Path) -> torch.nn.Module:
        '''
        Load best model (by validation loss) from output_dir,
        Return the model (just a syntax-sugar)
        '''
        log('Loading best checkpoint')
        best_checkpoint_dir = None
        min_eval_loss = 1e9
        for checkpoint_dir in sorted(output_dir.glob('checkpoint-*')):
            if not checkpoint_dir.is_dir(): continue
            eval_result = utils.load_json(checkpoint_dir / 'eval_result.json')
            eval_loss = eval_result['loss']
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                best_checkpoint_dir = checkpoint_dir

        best_checkpoint_file = best_checkpoint_dir / 'pytorch_model.bin'
        log(f'Best checkpoint: {best_checkpoint_file}')
        self.load_model(best_checkpoint_file)
        return self.model
