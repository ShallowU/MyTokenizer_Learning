import jieba
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC

# --- 0. 准备一个中文样本文本 ---
sample_text_chinese = "我爱自然语言处理，特别是大语言模型。今天天气真不错！"
print(f"原始文本: {sample_text_chinese}\n")

# --- 1. Jieba 分词 (传统中文分词) ---
print("--- Jieba 分词示例 ---")

# 默认精确模式
seg_list_default = jieba.lcut(sample_text_chinese)
print(f"Jieba (精确模式): {' / '.join(seg_list_default)}")

# 全模式
seg_list_full = jieba.lcut(sample_text_chinese, cut_all=True)
print(f"Jieba (全模式): {' / '.join(seg_list_full)}")

# 搜索引擎模式
seg_list_search = jieba.lcut_for_search(sample_text_chinese)
print(f"Jieba (搜索引擎模式): {' / '.join(seg_list_search)}")

# 添加自定义词典（Jieba 允许用户干预分词结果）
jieba.add_word("大语言模型") # 将“大语言模型”视为一个词
seg_list_custom = jieba.lcut(sample_text_chinese)
print(f"Jieba (添加自定义词后): {' / '.join(seg_list_custom)}")
print("Jieba点评: 基于词典和统计方法，分出的是我们传统意义上的“词”。全模式会列出所有可能的词，搜索引擎模式会切分出更短的词。可以添加自定义词典。")
print("-" * 30 + "\n")


# --- 2. Hugging Face Tokenizers (面向LLM的子词分词) ---
print("--- Hugging Face Tokenizers 示例 ---")

# --- 2.1 使用预训练的Tokenizer (以bert-base-chinese 为例) ---
print("--- 2.1 使用预训练的 bert-base-chinese Tokenizer ---")
# bert-base-chinese 使用的是 WordPiece 算法
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-chinese")

print(f"使用的Tokenizer: {tokenizer_bert.name_or_path}")
print(f"词表大小 (Vocabulary Size): {tokenizer_bert.vocab_size}")

# Tokenize
tokens_bert = tokenizer_bert.tokenize(sample_text_chinese)
print(f"BERT Tokenizer 分词结果: {tokens_bert}")

# Encode (将tokens转换为ID)
encoded_bert = tokenizer_bert.encode(sample_text_chinese, add_special_tokens=True)
print(f"BERT Tokenizer 编码结果 (Input IDs): {encoded_bert}")

# Decode (将ID转换回文本)
decoded_bert = tokenizer_bert.decode(encoded_bert)
print(f"BERT Tokenizer 解码结果: {decoded_bert}")

# 查看一些特殊tokens
print(f"BERT UNK token: {tokenizer_bert.unk_token} (ID: {tokenizer_bert.unk_token_id})")
print(f"BERT CLS token: {tokenizer_bert.cls_token} (ID: {tokenizer_bert.cls_token_id})")
print(f"BERT SEP token: {tokenizer_bert.sep_token} (ID: {tokenizer_bert.sep_token_id})")
print(f"BERT PAD token: {tokenizer_bert.pad_token} (ID: {tokenizer_bert.pad_token_id})")
print(f"BERT MASK token: {tokenizer_bert.mask_token} (ID: {tokenizer_bert.mask_token_id})")

print("BERT Tokenizer点评: 注意到了吗？")
print("1. '大语言模型' 可能被拆分成更小的单元，如 '大', '语', '言', '模', '型' 或 '大', '语言', '##模型' (WordPiece中##表示词内子词)。")
print("2. 出现了特殊标记如 [CLS] 和 [SEP] (如果 add_special_tokens=True)。")
print("3. 即便是单个汉字，也可能因为在词表中而直接映射。")
print("-" * 30 + "\n")


# --- 2.2 从零开始训练一个简单的BPE Tokenizer (使用 tokenizers 库) ---
print("--- 2.2 从零开始训练一个简单的 BPE Tokenizer ---")

# 准备一个小语料库 (实际中需要非常大的语料库)
corpus = [
    "我爱自然语言处理",
    "大语言模型是未来的趋势",
    "今天天气真不错，适合学习自然语言处理。",
    "中文分词很有趣，也很有挑战。",
    "这是一个小小的演示语料。"
]

# 1. 初始化Tokenizer模型 (使用BPE)
my_bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 设置Normalizer (例如NFKC Unicode标准化)
my_bpe_tokenizer.normalizer = NFKC()

# 3. 设置Pre-tokenizer (对于中文，ByteLevel可以很好地处理Unicode字符，将其分解为字节序列)
# add_prefix_space=False: 对于中文，我们通常不期望在每个词前加空格，ByteLevel会处理UTF-8字节
my_bpe_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 4. 设置Trainer (BPE训练器)
# vocab_size: 期望的词表大小 (这里设置得很小，仅为演示)
# special_tokens: 定义一些特殊用途的token
trainer = BpeTrainer(vocab_size=1000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

# 5. 训练Tokenizer
# train() 方法期望一个文件列表，我们这里用一个包含字符串的列表，并将其写入临时文件（或直接传递迭代器）
# For simplicity, we'll use the list directly (newer versions of tokenizers might handle iterables better,
# but traditionally it's file paths. Let's simulate by joining and splitting for this example if needed or pass list directly)
# The train_from_iterator method is preferred for in-memory iterables
my_bpe_tokenizer.train_from_iterator(corpus, trainer=trainer)

print(f"训练得到的BPE词表大小: {my_bpe_tokenizer.get_vocab_size()}")

# 6. 使用训练好的Tokenizer进行分词和编码
text_to_tokenize_custom = "我爱大语言模型和自然语言处理。"
custom_encoding = my_bpe_tokenizer.encode(text_to_tokenize_custom)

print(f"自定义BPE对 '{text_to_tokenize_custom}' 的分词结果: {custom_encoding.tokens}")
print(f"自定义BPE对 '{text_to_tokenize_custom}' 的编码结果 (IDs): {custom_encoding.ids}")

# 解码
decoded_custom = my_bpe_tokenizer.decode(custom_encoding.ids)
print(f"自定义BPE解码结果: {decoded_custom}")

print("\n自定义BPE Tokenizer点评:")
print("1. 我们用一个小语料训练了一个BPE tokenizer。词表大小是我们设定的（或接近）。")
print("2. 常见的多字节UTF-8字符（如汉字）被ByteLevel预处理器分解成字节，然后BPE算法在字节级别上工作，合并高频字节对。")
print("3. 分词结果是子词单元，可能是单个字节（显示为字符），也可能是合并后的高频序列。")
print("4. 这是一个非常简化的例子，实际LLM的tokenizer训练需要大规模、高质量的语料和更精细的参数调整。")
print("-" * 30 + "\n")

print("实验总结:")
print("Jieba代表了传统的基于词典和统计的中文分词，目标是切分出语义完整的词。")
print("Hugging Face Tokenizers (如BERT的WordPiece或我们自己训练的BPE) 则采用子词策略，")
print("旨在平衡词表大小、OOV问题和模型效率，是现代LLM的常见选择。")
print("它们的输出单元（token）不一定是完整的词，可能是更小的有意义片段或单个字符/字节。")