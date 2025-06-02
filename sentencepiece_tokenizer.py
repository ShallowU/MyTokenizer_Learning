import sentencepiece as spm

with open("toy.txt", "w", encoding="utf-8") as f:
  f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

# train a sentencepiece model on it
# the settings here are (best effort) those used for training Llama 2
import os

options = dict(
  # input spec
  input="toy.txt",
  input_format="text",
  # output spec
  model_prefix="models/sentencepiece_tok400", # output filename prefix
  # algorithm spec
  # BPE algorithm
  model_type="bpe",  # 当然还有unigram这种算法，这里我们选择BPE
  vocab_size=400,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization，就是不对文本进行任何处理保持原样
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences，最大训练的句子数
  max_sentence_length=4192, # max number of bytes per sentence，每个句子的最大字节数
  seed_sentencepiece_size=1000000, # number of sentences to use for seed sentencepiece model
  shuffle_input_sentence=True, # 打乱输入句子顺序
  # rare word treatment
  character_coverage=0.99995,   # 99.995% of characters in the training data will be covered by the model，比如在1M个句子中，99.995%的字符会被包含在模型中，仅出现一次或者少数的字符会被忽略
  # byte fallback
  byte_fallback=True, # 若为true，则在训练过程中，若某个字符未被包含在模型中，则会将其转换为对应的字节序列（回退到256个基本字节），这样可以确保所有字符都能被处理，若为false，则会忽略未包含在模型中的字符，这可能导致某些字符无法被正确处理，比如一长串中文只会变为一个unk标记，id为0的标记
  # merge rules
  split_digits=True,            # 这些规则类似bpe的regex split规则
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,# 在每个句子前添加一个虚拟前缀这里用的是空格，通常用于处理句子的开头
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)
sp = spm.SentencePieceProcessor()
sp.load('models/sentencepiece_tok400.model')
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
print(vocab[:50])  # print first 50 tokens

ids = sp.encode("hello world!!!?  😉你好,吃饭了吗", out_type=int)
print("encode text:", "hello world!!!?  😉你好,吃饭了吗")
print("encoded ids:", ids)
# Let's also decode the ids back to text
print("decoded text:", sp.decode(ids))
print("详细每个token的id和对应的piece:")
print([(idx, sp.id_to_piece(idx)) for idx in ids])