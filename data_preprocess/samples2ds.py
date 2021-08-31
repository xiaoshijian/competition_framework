import copy
import torch
from transformers import BertTokenizerFast

class CailNerDataset():
    def __init__(self,
                 samples,
                 labels_ids_mapping,
                 max_length=256):
        self.samples = copy.deepcopy(samples)
        self.labels_ids_mapping = copy.copy(labels_ids_mapping)
        self.tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.max_length = max_length

    def __getitem__(self, item):
        sample = self.samples[item]
        ordnum = sample['ordnum']

        tokenized_inputs = self.tokenizer.encode_plus(sample['char_list'],
                                                      is_split_into_words=True,
                                                      #  return_tensors='pt'
                                                      )
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        # word_ids = tokenized_inputs.word_ids(batch_index=0)
        original_char_index_with_se = [-1] + copy.copy(sample['original_char_index']) + [-1]

        label_ids = [
            self.labels_ids_mapping[label] if label in self.labels_ids_mapping else self.labels_ids_mapping['O'] for
            label in sample['labels_list']]
        # label_ids = [-100] + label_ids + [-100]
        label_ids = [self.labels_ids_mapping['O']] + label_ids + [self.labels_ids_mapping['O']]

        # padding
        length_before = len(input_ids)
        if length_before < self.max_length:
            input_ids = input_ids + [0] * (self.max_length - length_before)
            attention_mask = attention_mask + [0] * (self.max_length - length_before)
            # label_ids = label_ids + [-100] * (self.max_length - length_before)
            label_ids = label_ids + [self.labels_ids_mapping['O']] * (self.max_length - length_before)
            original_char_index_with_se = original_char_index_with_se + [-1] * (self.max_length - length_before)

        return {
            'ordnum': torch.tensor(ordnum, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'orig_char_pos_list': torch.tensor(original_char_index_with_se, dtype=torch.long),
        }

        # char_list_with_se = ['[CLS]'] + copy.copy(char_list) + ['[SEP]']  # 添加了start_end
        # labels_list_with_se = ['O'] + copy.copy(labels_list) + ['O']  # 用O来标识吧
        # original_char_index_with_se = [-1] + copy.copy(original_char_index) + [-1]
        # inputs_ids
        # attention

    def __len__(self):
        return len(self.samples)
