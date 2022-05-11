from typing import List


def get_entity_bios(seq, id2label: List[str]):
	"""Gets entities from sequence.
	note: BIOS
	Args:
		seq (list): sequence of labels.
	Returns:
		list: list of (chunk_type, chunk_start, chunk_end).
	Example:
		# >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
		# >>> get_entity_bios(seq)
		[['PER', 0,1], ['LOC', 3, 3]]
	"""
	chunks = []
	chunk = [-1, -1, -1]

	for indx, tag in enumerate(seq):
		if not isinstance(tag, str):
			# Convert to str
			if 0 <= tag < len(id2label):
				tag = id2label[tag]
			else:
				tag = 'O'
		if tag.startswith("S-"):
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[2] = indx
			chunk[0] = tag.split('-')[1]
			chunks.append(chunk)
			chunk = (-1, -1, -1)
		if tag.startswith("B-"):
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[0] = tag.split('-')[1]
		elif tag.startswith('I-') and chunk[1] != -1:
			_type = tag.split('-')[1]
			if _type == chunk[0]:
				chunk[2] = indx
			if indx == len(seq) - 1:
				chunks.append(chunk)
		else:
			if chunk[2] != -1:
				chunks.append(chunk)
			chunk = [-1, -1, -1]
	return chunks



def get_metrics(labels: List[List[int]], preds: List[List[int]], id2label: dict) -> dict:
	'''
	Calculates metrics for NER.
	'''
	assert len(preds) == len(labels), 'The size of two lists should be same.'
	prec, recall, f1 = 0, 0, 0
	num_preds = 0
	num_tags = 0
	for i in range(len(preds)):
		# Turn into List[str]
		pred_entities = get_entity_bios(preds[i], id2label)
		label_entities = get_entity_bios(labels[i], id2label)

		correct = 0
		if label_entities == []:
			print(pred_entities, label_entities)
			print(labels[i])
			print(i)
			print([id2label[x] for x in labels[i]])
			raise ValueError('label_entities should not be empty.')
		for pred_entity in pred_entities:
			if pred_entity in label_entities:
				correct += 1
		num_preds += len(pred_entities)
		num_tags += len(label_entities)
	prec = 0 if num_preds == 0 else correct / num_preds
	recall = 0 if num_tags == 0 else correct / num_tags
	# This is micro-F1
	if prec + recall == 0: 
		f1 = 0
	else:
		f1 = 2 * prec * recall / (prec + recall)
	return {'prec': prec, 'recall': recall, 'f1': f1}
