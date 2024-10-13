from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer, BertModel
from torchtext.vocab import build_vocab_from_iterator, Vectors, vocab
import torch

UNK, PAD, SEP = '[UNK]', '[PAD]', '[SEP]'

class BasicTokenizer:
    def __init__(self, max_len=256, data_path=None):
        self.tokenizer = get_tokenizer('basic_english')
        self.max_len = max_len
        self.vocab = None
        self.data_path = data_path
        self.vectors = None

    def yield_tokens(self, data_iter):
        for content in data_iter:
            yield self.tokenizer(content)

    def build_vocab(self, text_list, seed):
        self.vocab = build_vocab_from_iterator(self.yield_tokens(text_list), specials=[UNK, PAD])
        self.vocab.set_default_index(self.vocab[UNK])
        torch.save(self.vocab, self.data_path + 'vocab_{}'.format(seed))

    def load_vocab(self, text_list, seed):
        try:
            self.vocab = torch.load(self.data_path + 'vocab_{}'.format(seed))
        except Exception as e:
            print(e)
            self.build_vocab(text_list, seed)

    def encode(self, text):
        tokens = self.tokenizer(text)
        seq_len = len(tokens)
        if seq_len <= self.max_len:
            tokens += (self.max_len - seq_len) * [PAD]
        else:
            tokens = tokens[:self.max_len]
            seq_len = self.max_len
        ids = self.vocab(tokens)
        masks = [1] * seq_len + [0] * (self.max_len - seq_len)
        return ids, seq_len, masks


class CustomBertTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, bert_path=None, data_path=None):
        super(CustomBertTokenizer, self).__init__(max_len)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def build_vocab(self, text_list=None, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def load_vocab(self, text_list=None, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def encode(self, text):
        result = self.tokenizer(text)
        result = self.tokenizer.pad(result, padding='max_length', max_length=self.max_len)
        ids = result['input_ids']
        mask = result['attention_mask']
        seq_len = sum(mask)
        # SEP_IDX = self.tokenizer.convert_tokens_to_ids([SEP])
        SEP_IDX = self.tokenizer.vocab[SEP]
        if seq_len > self.max_len:
            ids = ids[:self.max_len - 1] + [SEP_IDX]
            mask = mask[:self.max_len]
            seq_len = self.max_len
        return ids, seq_len, mask


class VectorTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, vector_path=None, data_path=None, name=None):
        super(VectorTokenizer, self).__init__(max_len, data_path)
        self.vector_path = vector_path
        self.vectors = None
        self.name = name

    def build_vocab(self, text_list=None, seed=None):
        vec = Vectors(self.vector_path)
        self.vocab = vocab(vec.stoi, min_freq=0)
        self.vocab.append_token(UNK)
        self.vocab.append_token(PAD)
        self.vocab.set_default_index(self.vocab[UNK])
        unk_vec = torch.mean(vec.vectors, dim=0).unsqueeze(0)
        pad_vec = torch.zeros(vec.vectors.shape[1]).unsqueeze(0)
        self.vectors = torch.cat([vec.vectors, unk_vec, pad_vec])
        if self.name:
            torch.save(self.vocab, self.data_path + '{}_vocab'.format(self.name))
            torch.save(self.vectors, self.data_path + self.name)
        else:
            torch.save(self.vocab, self.data_path + 'vector_vocab')
            torch.save(self.vectors, self.data_path + 'vectors')

    def load_vocab(self, text_list=None, seed=None):
        try:
            if self.name:
                self.vocab = torch.load(self.data_path + '{}_vocab'.format(self.name))
                self.vectors = torch.load(self.data_path + self.name)
            else:
                self.vocab = torch.load(self.data_path + 'vector_vocab')
                self.vectors = torch.load(self.data_path + 'vectors')
        except Exception as e:
            print(e)
            self.build_vocab()
