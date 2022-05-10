def get_entity_bios(seq,id2label):
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
			tag = id2label[tag]
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



def get_metrics(preds, labels, id2label: dict) -> dict:
	'''
	
	'''
	assert len(preds) == len(labels), 'The size of two lists should be same.'
	prec, recall, f1 = 0, 0, 0
	tp, tn, fp, fn = 0, 0, 0, 0
	for i in range(len(preds)):
		# Turn into List[str]
		pred_entities = get_entity_bios(preds[i], id2label)
		label_entities = get_entity_bios(labels[i], id2label)
		print(pred_entities, label_entities)
		exit()
