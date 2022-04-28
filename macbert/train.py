from pathlib import Path
from datetime import datetime

from transformers import BertForSequenceClassification, BertTokenizer
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import AdamW, get_scheduler
from torch.utils.data import TensorDataset, DataLoader

from data import afqmc

model_path = 'hfl/chinese-macbert-base'

data_dir = Path('../data/afqmc/split')

def features_to_dataset(features):
    return TensorDataset(
        features['input_ids'],
        features['token_type_ids'],
        features['attention_mask'],
        features['labels'])

def get_afqmc_dataset(tokenizer, max_seq_len):
    print('Getting training examples')
    def _get(phase):
        kwargs = {'max_seq_len': max_seq_len,
                  'tokenizer': tokenizer,
                  'num_examples': 256}
        return afqmc.AfqmcDataset(data_dir / f'{phase}.json', phase, **kwargs)
    return _get('train'), _get('dev')


def evaluate(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    score = 0.0
    for batch in dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = model(input_ids, token_type_ids, attention_mask)


# Data
print('Preparing data')
tokenizer = BertTokenizer.from_pretrained(model_path)
train_dataset, eval_dataset = get_afqmc_dataset(tokenizer, 512)

# Model
print('Getting model')
model = BertForSequenceClassification.from_pretrained(model_path)


batch_size = 4
num_epochs = 2
warmup_prop = 0.1
lr = 2e-5

print(len(train_dataset))

model = model.cuda()

train_args = TrainingArguments(
    'results/temp', 
    overwrite_output_dir=True, # TODO: remove on release
    do_train=True,
    do_predict=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=2,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    logging_dir='results/' + datetime.now().strftime("%y%m%d%H%M%S"),
    logging_first_step=True,
    logging_steps=5,
    seed=0,
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,
)

for ep in range(num_epochs):
    trainer.train(num_train_epochs=4)


# Test on clean data
test_dataset = afqmc.AfqmcDataset(
    data_dir / 'dev.json',
    'dev',
    
)
trainer.predict()
