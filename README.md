> [!NOTE]
> **Learning way:** 
>
> basic_tokenizer.py---->regext_tokenizer.py---->gpt4_tokenizer.py---->sentencepiece_tokenizer.py---->jieba_demo.py

## 0.介绍

这是我学习LLM tokenizer的经历和文档，旨在帮助和我一样初始新手向的小白。这里面的内容来自[andrej karpathy](https://www.youtube.com/@AndrejKarpathy)课程视频，感谢大神的课程！

若根据文档该文档进行实验，则环境配置简单如下:

```
git clone 该仓库
cd 该仓库文件夹
```

```
pip install tiktoken sentencepiece jieba tokenizers
```

## 1.实现基本的 BasicTokenizer

- train()
训练函数，就是通过训练文本，迭代num_merges = vocab_size - 256这么多次，不停的替换，保存下vocab 格式：前256的idx->bytes([idx])，后面每次迭代替换的为vocab[idx] = vocab[pair[0]] + vocab[pair[1]]。而merge是(int, int) -> int
- decode()
将ids  (list of integers), return Python string
对ids中的每个id进行vocab中的替换，转换为字节流，字节流进行utf-8解码时候选项errors="replace"
- encode()
将文本进行utf-8编码再list转换为整数，如果数量大于1即不是单个字节，就对该序列遍历进行转换为((int,int),次数)，对merge中的从256开始从小到大进行不断替换合并，直到pair没在merge中找到。
## 2.实现RegexTokenizer，添加正则表达式处理chunks
- self.pattern 
```
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```
这个复杂的正则表达式根据语言学规则将文本分割成有意义的块:区分了大小写，并且相对于gpt2的正则表达式，更加合理。

> 缩写 ('s, 'll等)
> 连续的字母序列(词)
> 短数字序列
> 标点符号
> 空白字符

- train()
在上述的基础上，re.findall(self.compiled_pattern, text)分成很多chunks，每个chunks单独utf-8编码再list整数，对每个chunks中tuple出现频率统计，每次仍然对索引最小的的tuple进行并行chunks的merge。
- decode()
不变
- encode()
先对text按正则表达式re.findall(self.compiled_pattern, text)分块，对每个chunk进行单独的utf-8编码和list整数，在进行小的_encode_chunk(self, text_bytes)，与basic的encode 一样，最终将所有的chunk编码拼接在一起
## 3.复现openai的gpt4 Tokenizer的问题

- 1.合并规则恢复问题： 分词器与GPT-4的分词器要想完全匹配，需要获取相同的合并规则（merges）。但是从现有的tiktoken库中直接获取这些规则并不简单， GPT-4分词器在内部使用一种称为_mergeable_ranks的格式来存储词汇表。
  gpt4 merge:the `merges` are already the byte sequences in their merged state.
  my merge:字典，整数对应的

```python
def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges
```

GPT-4分词器的合并规则存储在一个名为_mergeable_ranks的字典中。这个字典的键是字节序列（bytes对象），值是整数，表示该字节序列在BPE合并操作中的优先级/顺序。为了让我们的分词器能够正确地使用这些合并规则，我们需要将_mergeable_ranks转换为merges字典。这个恢复过程有点复杂，因为对于一个给定的字节序列，可能有多种拆分方式，recover_merges函数需要找出正确的拆分方式。

_mergeable_ranks是一个优化后的格式，存储了字节序列到排名的映射.而我的分词器需要的是(左令牌ID, 右令牌ID) -> 新令牌ID的映射

recover_merges函数的作用就是从优化的格式恢复出原始的合并规则格式，这样我们的分词器就能完全匹配GPT-4分词器的行为。

mergeable_ranks 是一个字典，其中：
键（token）：是字节序列（bytes 对象），代表一个 token，如是单个字节（如 b'a'），多个字节（如 b'in'，表示已经合并的 token）
值（rank）：是整数，表示该 token 在 BPE 合并操作中的优先级/顺序
越小的数字表示越早被合并，这实际上就是 token 的 ID。
迭代所有 token 和对应的 rank，跳过单字节 token（基础 token，无需恢复合并关系），对于多字节 token使用 bpe 辅助函数确定哪两个子 token 在该 rank 之前合并形成了当前 token，获取这两个子 token 的索引（rank 值），将这对索引及其合并结果记录到 merges 字典中。



- 2.GPT-4分词器还有一个额外的复杂性 - 它对基本字节进行了重新排序（permutation）。这意味着基本字节的编码不是简单的0-255，而是经过了一个映射。在标准BPE实现中，通常会假设基本字节的编码是0-255，但GPT-4分词器对这些基本字节进行了重新排序。这个重排映射存储在_mergeable_ranks的前256个元素中。decode函数需要使用这个映射来正确地将字节转换为ID和反向转换，例如decode中：

```python
bytes(self.inverse_byte_shuffle[b] for b in part)
    result.append(part)
```
## 4.添加 special tokens

decode中：分别处理普通标记和特殊标记

```
    def decode(self, ids):
        # 分别处理普通标记和特殊标记
        result = []
        
        for idx in ids:
            if idx in self.vocab:
                # 普通标记：获取字节序列并应用逆字节重排
                part = self.vocab[idx]
                part = bytes(self.inverse_byte_shuffle[b] for b in part)
                result.append(part)
            elif idx in self.inverse_special_tokens:
                # 特殊标记：直接添加，不应用任何重排
                special_text = self.inverse_special_tokens[idx]
                result.append(special_text.encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        
        # 合并所有结果并解码为文本
        text_bytes = b"".join(result)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
```

encoding时候：

```
if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
```

看不同的模式,对不同的模式采用不用的正则表达式过滤特殊标记。

```
special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
```

对源文本按special tokens分块，单独处理再按上述regrex encoding处理。



## 5.理解另一种方式的tokenizer：SentencePiece

与 tiktoken 不同，它可以高效地训练和推理 BPE 分词器。Llama 和 Mistral 系列都使用它。为多语言设计，对非英语语言（如中文、日语）表现更好。

SentencePiece 是一个无监督的文本分词器，主要特点包括：

1. **语言无关性**：不依赖于特定语言的预处理，可以处理任何语言
2. **端到端系统**：直接从原始文本训练，无需语言特定的预处理
3. **支持多种算法**：包含 BPE (Byte-Pair Encoding) 和 Unigram 语言模型
4. **字节回退机制**：当遇到未知字符时，可以回退到字节级别表示

**工作方式**：

- 将文本视为一个单一序列，不预先进行分词
- 可以通过设置 `byte_fallback=True`确保任何字符都能被处理
- 能够处理空格、特殊符号和表情符号
- 添加 `▁`（U+2581，下划线）作为空格的表示，以便能够恢复原始空格



与tiktoken最大的区别在于：sentencepiece 直接在 Unicode 码点上运行 BPE！它还有一个 character_coverage 选项，用于处理出现次数极少的非常罕见的码点。它要么将它们映射到 UNK 标记上，要么在 byte_fallback 开启的情况下，先用 utf-8 编码，然后再对原始字节进行编码。



## 6.tokenization遇到的问题

 1. 为什么大语言模型拼写不准确？

**问题**：字符被切分成不同大小的词元（token）。提示词"How many letters 'l' are in '.DefaultCellStyle'"会得到各种错误结果。

**解释**：语言模型不是逐字符处理文本，而是将文本分解成词元。例如，".DefaultCellStyle"可能被分成[".Default", "Cell", "Style"]这样的词元。这导致模型无法准确计算单个字符的出现次数，因为字符'l'分散在不同词元中，模型难以追踪它们的确切数量。

2. 为什么大语言模型难以执行简单的文本操作（如字符串反转）？

**问题**：同样，字符被切分成不同大小的词元。这使ChatGPT产生错误，但反转".D e f a u l t C e l l S t y l e"（每个字符之间有空格）却能正常工作

**解释**：当每个字符之间加入空格时，分词器通常会将每个字符作为独立的词元处理，模型就能更准确地处理每个字符。而正常文本中，单词会被分成各种大小的词元，模型很难追踪每个字符的位置，导致反转操作出错。

3. 为什么大语言模型在处理非英语语言时表现较差？

**问题**：训练数据中非英语语言表示较少，导致非英语字符的分词较少（如果有的话）。这会使注意力缓冲区膨胀，导致效果不佳。

**解释**：英语在大多数模型的训练数据中占主导地位，因此分词器针对英语进行了优化。对于中文、阿拉伯文等非英语语言，一个字符或一个词可能需要多个词元来表示，这增加了序列长度，占用了更多的注意力计算资源，最终导致模型对这些语言的处理能力下降。

4. 为什么大语言模型无法正确执行简单算术运算？

**问题**：数字分词是根本原因。更准确地说，[整数分词是极其混乱的](vscode-file://vscode-app/d:/MicrosoftIDE/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)。

**解释**：数字在分词系统中处理方式非常不统一。比如，数字"1234"可能被分解为["12", "34"]或["1", "23", "4"]等不同组合，这使模型难以理解数字的真实值和顺序。当进行算术运算时，模型必须将这些分散的词元重新组合成完整的数字，再进行计算，容易出错。

 5. 为什么输入特殊字符如`<|endoftext|>`会停止生成？

**问题**：这是一个特殊标记。它被非常特殊地解释，因此不要将其视为普通词元，而是一个命令。

**解释**：像`<|endoftext|>`这样的特殊标记在模型训练过程中有特定含义，用于标记文本结束。当模型遇到这类标记时，会将其理解为"生成结束"的信号，而不是普通文本内容。这类似于计算机编程中的控制字符，是给模型的指令而非内容。

 6. 为什么尾随空格会导致大语言模型出现打嗝？

**问题**：GPT等模型的词元通常遵循"(空格)(内容)"模式，因此添加空格会使GPT尝试生成与这个预先存在的空格相匹配的内容，而这种情况很少见，因此会"污染"上下文。

**解释**：许多词元是以空格开头的，如" the"、" and"等。当一行文本以空格结束时，模型会认为下一个词应该不带空格开头，但这与其训练模式不符。结果，模型可能生成不自然或不连贯的内容，因为它试图适应这种不常见的模式。

 7. 为什么在单词中遇到大写字母时大语言模型会崩溃？

**问题**：模型以前从未见过这种情况；它感到困惑，很快转向预测单词结束标记。

**解释**：在训练数据中，单词中间出现大写字母的情况相对罕见（除了像驼峰命名法这样的特例）。当模型遇到如"heLLo"这样的词时，由于缺乏类似模式的训练经验，可能会错误地认为大写字母标志着某种特殊用途或单词边界，导致预测出现异常。

8. 为什么通过YAML而非JSON与大语言模型交互可能更好？

**问题**：YAML包含更少的特殊字符，因此不太可能绊倒分词器。

**解释**：JSON使用许多特殊字符（如{}[],:""）作为语法的一部分。这些字符在分词时可能被单独处理，增加了模型正确解析和生成JSON的难度。相比之下，YAML语法更接近自然语言，使用缩进和较少的特殊字符，更适合分词器处理，减少了出错可能性。

 9. 为什么LLM实际上并不意味着端到端的语言建模？

**问题**：我们需要分词来形成统一且可扩展的文本表示基础。

**解释**：虽然大语言模型旨在处理自然语言，但它们实际上并不直接处理原始文本。文本必须先转换为词元序列，再由模型处理，然后输出也必须从词元转回文本。这种中间转换步骤使得语言模型处理不是真正的"端到端"，而是依赖于预定义的分词规则。

10. "SolidGoldMagikarp"现象是怎么回事？

**问题**：这完全打破了ChatGPT等大型语言模型，本质上破解了它的安全限制。似乎"reddit.com/u/SolidGoldMagikarp"发帖如此频繁，以至于分词器从训练中记住了它，但大语言模型却不认识它。

**解释**：这是一个有趣的现象，展示了分词器和模型理解之间的脱节。该Reddit用户名被作为单一词元存储在分词器词汇表中（可能因其在训练数据中频繁出现），但模型本身对这个词元的语义理解有限。当用这个罕见但有效的词元构建提示时，可能导致模型行为异常，因为它打破了模型对输入的正常理解路径，有时能绕过安全过滤机制。

## 7.中文汉字tokenize

中文汉字的处理方式与英文等拼音文字有所不同，中文没有像英文那样的天然空格作为词语的明显分隔符。

当目标语言为中文时，推荐使用WordPiece + jieba 的方案。

中文中，有明确的字/词概念，却没有子词的概念（如英文中有”app”, “##le”, 中文却没有”苹” “##果”），而转bytes 后对子词更友好，此外，中文通常需要3个bytes（GBK）或者4个bytes（Chinese-Japanese character set），对于一个中文的字，很有可能需要大于1个token 来表示，反而会增加tokenize 后序列的长度，对模型的训练与使用不利；此外，中文中空格也没有切分词/句子 的语义，保留空格反而会由于各种空格的错误使用带来问题，最终的推荐方案就是jieba + Word Piece/SentencePieceUnigram。

jieba:https://github.com/fxsjy/jieba

学习jieba：

## 参考资料

[andrej karpathy's video](https://www.youtube.com/watch?v=zduSFxRajkE&t=5338s)

[minbpe](https://github.com/karpathy/minbpe/blob/master/exercise.md)

[notes for the above video](https://github.com/MK2112/nn-zero-to-hero-notes/blob/main/N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb)



