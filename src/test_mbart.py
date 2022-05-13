import torch
assert torch.cuda.is_available()


from pathlib import Path

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from utils import load_jsonl, dump_jsonl, get_param_count, Logger

model_path = "facebook/mbart-large-50-many-to-one-mmt"

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
article_zh = '新华时评：把优秀返乡农民工打造成乡村振兴生力军-新华网'


output_path = Path('results/nmt/mbart-large')
log_file = output_path / 'test.log'

logger = Logger(log_file)

logger.log('Getting model...')
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
logger.log('Model loaded.')
logger.log(f'# params: {get_param_count(model)}')


logger.log('translate Hindi to English')
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi)
output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
logger.log(output)
# => "The head of the UN says there is no military solution in Syria."

logger.log('translate Arabic to English')
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar)
outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
logger.log(output)
