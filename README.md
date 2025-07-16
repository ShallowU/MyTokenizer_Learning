# LLM Tokenizer 学习指南

> [!NOTE]
> **学习路径推荐**  
> `basic_tokenizer.py` → `regext_tokenizer.py` → `gpt4_tokenizer.py` → `sentencepiece_tokenizer.py` → `jieba_demo.py`

## 📖 项目介绍

这是一个从零开始学习大语言模型（LLM）Tokenizer 的教程项目，特别适合初学者。内容基于 [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) 的优秀课程。

### 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/ShallowU/MyTokenizer_Learning.git
cd MyTokenizer_Learning

# 安装依赖
pip install tiktoken sentencepiece jieba tokenizers
```

## 🎯 学习内容

### 1. 基础 Tokenizer (BasicTokenizer)

**核心概念：** 字节对编码（BPE）的基本实现

**关键方法：**
- **`train()`**: 训练分词器
  - 迭代 `num_merges = vocab_size - 256` 次
  - 不断合并最频繁的字节对
  - 构建词汇表：`vocab[idx] = vocab[pair[0]] + vocab[pair[1]]`

- **`decode()`**: 解码 token 序列
  ```python
  # 将 token ID 列表转换为字符串
  ids → vocab 查找 → 字节流 → UTF-8 解码
  ```

- **`encode()`**: 编码文本
  ```python
  # 文本 → UTF-8 字节 → 应用合并规则 → token ID 列表
  ```

### 2. 正则表达式 Tokenizer (RegexTokenizer)

**改进点：** 添加智能文本分块，提高分词质量

**核心特性：**
```python
# GPT-4 使用的分割模式
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

**分割规则：**
- 📝 缩写词（'s, 'll 等）
- 🔤 连续字母序列（单词）
- 🔢 短数字序列
- ⭐ 标点符号
- ⬜ 空白字符

**工作流程：**
```
原始文本 → 正则分块 → 每块独立编码 → 合并结果
```

### 3. GPT-4 兼容 Tokenizer

**挑战与解决方案：**

#### 🔧 挑战1：合并规则恢复
**问题：** GPT-4 使用 `_mergeable_ranks` 格式存储词汇表  
**解决：** 实现 `recover_merges()` 函数转换格式

```python
def recover_merges(mergeable_ranks):
    """从 GPT-4 格式恢复合并规则"""
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # 跳过基础字节
        # 恢复合并对
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges
```

#### 🔧 挑战2：字节重排序
**问题：** GPT-4 对基础字节进行了重新排序  
**解决：** 使用逆映射恢复原始字节顺序

### 4. 特殊 Token 处理

**功能扩展：** 支持特殊控制 token（如 `<|endoftext|>`）

**实现要点：**
- 分别处理普通 token 和特殊 token
- 支持不同的特殊 token 处理模式
- 避免特殊 token 被意外分割

### 5. SentencePiece Tokenizer

**特点对比：**

| 特性 | TikToken | SentencePiece |
|------|----------|---------------|
| 语言支持 | 主要英语 | 多语言友好 |
| 处理级别 | 字节级 | Unicode 码点级 |
| 空格处理 | 正则分割 | `▁` 符号表示 |
| 未知字符 | 字节回退 | UNK 或字节回退 |

**优势：**
- 🌍 更好的多语言支持
- 🎯 端到端训练
- 🔄 字节回退机制
- ⚡ 高效的训练和推理

### 6. 中文处理方案

**推荐方案：** Jieba + WordPiece

**原因分析：**
- 中文没有天然的词语分隔符
- 直接字节编码会增加序列长度
- 需要先进行词语切分再应用子词算法

**实现思路：**
```
中文文本 → Jieba 分词 → WordPiece 子词化 → Token 序列
```

## ⚠️ Tokenization 常见问题

### 1. 拼写计数错误
**现象：** 模型无法准确计算字符数量  
**原因：** 字符被分散在不同 token 中

### 2. 字符串操作困难
**现象：** 反转字符串等操作出错  
**解决：** 字符间加空格强制单字符分词

### 3. 非英语语言效果差
**原因：** 训练数据英语占主导，其他语言 token 效率低

### 4. 算术运算错误
**原因：** 数字分词不一致，如 "1234" → ["12", "34"]

### 5. 特殊字符影响
**现象：** `<|endoftext|>` 等特殊 token 触发意外行为

### 6. 空格敏感性
**现象：** 尾随空格影响生成质量  
**原因：** 破坏了 "(空格)(内容)" 的常见模式

### 7. 大小写异常
**现象：** 单词内大写字母导致预测异常  
**原因：** 训练中此类模式较少

### 8. 格式选择建议
**推荐：** YAML > JSON  
**原因：** YAML 特殊字符更少，分词更友好

## 📚 参考资料

- [Andrej Karpathy's Tokenizer 课程](https://www.youtube.com/watch?v=zduSFxRajkE&t=5338s)
- [MinBPE 项目](https://github.com/karpathy/minbpe)
- [课程笔记](https://github.com/MK2112/nn-zero-to-hero-notes/blob/main/N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb)
- [Jieba 中文分词](https://github.com/fxsjy/jieba)

---

💡 **提示：** 建议按照学习路径顺序逐步实践，每个阶段都有对应的代码实现可供参考。