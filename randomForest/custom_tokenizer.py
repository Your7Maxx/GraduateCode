# custom_tokenizer.py
from slpp import ShellTokenizer
from transformers import PreTrainedTokenizerFast

class CustomShellTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text):
        # 使用您的ShellTokenizer来分词
        t = ShellTokenizer(verbose=True)
        tokens, _ = t.tokenize(text)
        return tokens
