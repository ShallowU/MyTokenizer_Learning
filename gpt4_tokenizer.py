import tiktoken
import unicodedata
import os
from regext_tokenizer import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)
def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts
def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
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

class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # get the official tokenizer and its merges
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        # the merges are those of gpt4, but we have to recover them
        self.merges = recover_merges(mergeable_ranks)
        # reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # finally register the special tokens
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # before we start processing bytes, we have to permute them
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids
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
    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # save/load would require some thought.
    # we'd have to change save/load of base to add support for byte_shuffle...
    # alternatively, we could move byte_shuffle to base class, but that would
    # mean that we're making ugly our beautiful Tokenizer just to support
    # the GPT-4 tokenizer and its weird historical quirks around byte_shuffle.
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        # just for visualization purposes let's output the GPT-4 tokens
        # in the exact same format as the base class would.
        # simple run as:
        # python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        # build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # now merge the shuffled bytes and write to file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")


enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode("<|endoftext|>hello world!!!? (안녕하세요!) lol123 😉你好",allowed_special="all")
text = enc.decode(ids) # get the same text back
print("official Encoded IDs:", ids)
print("official Decoded text:", text)
# let's also test the GPT4Tokenizer
gpt4_tokenizer = GPT4Tokenizer()
ids = gpt4_tokenizer.encode("<|endoftext|>hello world!!!? (안녕하세요!) lol123 😉你好",allowed_special="all")
text = gpt4_tokenizer.decode(ids)
print("my GPT4 Encoded IDs:", ids)
print("my GPT4 Decoded text:", text)
# let's also save the vocab
gpt4_tokenizer.save_vocab("models/gpt4_c100k.vocab")
# and let's also print the special tokens
for token, idx in GPT4_SPECIAL_TOKENS.items():
    print(f"Special token: {token} -> ID: {idx}")
# and let's also print the byte shuffle
# print("Byte shuffle mapping:")
# for b, idx in sorted(gpt4_tokenizer.byte_shuffle.items()):
#     print(f"Byte {b:02x} -> ID: {idx}")
# # and let's also print the inverse byte shuffle
# print("Inverse byte shuffle mapping:")
# for idx, b in sorted(gpt4_tokenizer.inverse_byte_shuffle.items()):
#     print(f"ID {idx} -> Byte: {b:02x}")
