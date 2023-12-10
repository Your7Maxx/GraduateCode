import numpy as np
import bashlex
import logging
import re
import base64
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


class ShellTokenizer():
    def __init__(self, debug=False, verbose=False):
        self.ERR = 0
        self.verbose = verbose
        self.data = None
        self.global_counter = Counter()
        self.tokenized_corpus = []
        self.cache = {}

        # setup logging
        level = logging.INFO if verbose else logging.WARNING
        level = logging.DEBUG if debug else level
        logconfig = {
            "level": level,
            "format": "%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s"
        }
        logging.basicConfig(**logconfig)


    def preprocess_data(self, data):
        # 在这里对数据进行处理，例如清理、转换等操作
        processed_data = [self.preprocess_command(command) for command in data]
        return processed_data

    def preprocess_command(self, command):
        # 替换IP、URL和日期信息为"*"
        command = self.replace_ip_url_date(command)
        command = self.decrypt_encrypted_data(command)

        return command

    def replace_ip_url_date(self, command):
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4} \d{1,2}:\d{1,2}(:\d{1,2})?'

        command = re.sub(ip_pattern, '1.1.1.1', command)
        command = re.sub(url_pattern, 'example.com', command)
        command = re.sub(date_pattern, '2000-01-01', command)

        return command

    def decrypt_encrypted_data(self, command):
        pattern = r'echo\s*"(.*?)"\s*\|\s*base64\s*-d'
        match = re.search(pattern, command)
        if match:
            encrypted_data = match.group(1)
            decrypted_data = self.decrypt_base64(encrypted_data)

            decrypted_command = re.sub(pattern, decrypted_data, command)
        else:
            decrypted_command = command

        return decrypted_command

    def decrypt_base64(self, data):
        try:
            decoded_data = base64.b64decode(data.strip()).decode()
            return decoded_data
        except base64.binascii.Error:

            return data


    def _bashlex_wrapper(self, command):
        try:
            nodes = bashlex.parse(command)
            return nodes

        except (bashlex.errors.ParsingError, NotImplementedError, TypeError):
            self.ERR += 1
            rude_parse = []
            for element in re.split(r" |,|{|}", command):
                rude_parse.append(element)
            return rude_parse

    def _update_objects(self, counters, tokens):
        try:
            self.global_counter += counters
            self.tokenized_node.extend(tokens)
        except Exception as ex:
            print(ex)
            import pdb;pdb.set_trace()


    def _iterate_bashlex(self, bashlex_object):
        local_counter = Counter()
        local_tokens = []

        if isinstance(bashlex_object, list):
            for element in bashlex_object:
                self._update_objects(*self._iterate_bashlex(element))

        elif isinstance(bashlex_object, bashlex.ast.node):
            object_methods = [x for x in dir(bashlex_object) if '__' not in x]

            if 'command' in object_methods:
                self._update_objects(*self._iterate_bashlex(bashlex_object.command))
            elif 'list' in object_methods:
                for element in bashlex_object.list:
                    self._update_objects(*self._iterate_bashlex(element))
            elif 'parts' in object_methods:

                if bashlex_object.parts:
                    for part in bashlex_object.parts:
                        self._update_objects(*self._iterate_bashlex(part))
                else:
                    local_counter, local_tokens = self._parse_word(bashlex_object)
            elif 'word' in object_methods:
                local_counter, local_tokens = self._parse_word(bashlex_object)

            elif 'value' in object_methods:
                self.global_counter[bashlex_object.value] += 1
                self.tokenized_node.append(bashlex_object.value)
            elif "pipe" in object_methods:
                self.global_counter[bashlex_object.pipe] += 1
                self.tokenized_node.append(bashlex_object.pipe)
            elif "op" in object_methods:
                self.global_counter[bashlex_object.op] += 1
                self.tokenized_node.append(bashlex_object.op)
            elif "type" in object_methods:
                self.global_counter[bashlex_object.type] += 1
                self.tokenized_node.append(bashlex_object.type)
            else:
                logging.info(f"[DEBUG] Weird case - not parsed correctly: {bashlex_object}")

        return local_counter, local_tokens

    def _parse_word(self, bashlex_object):
        local_counter = Counter()
        local_tokens = []
        size = 1

        word = re.sub(r"[<>#{}]", "", bashlex_object.word).strip()
        if len(bashlex_object.word) > 20:
            p = self._bashlex_wrapper(word)
            if isinstance(p[0], bashlex.ast.node):
                try:
                    size = len(p[0].parts)
                except AttributeError:
                    size = len(p[0].list)

        if size > 1:
            for node in p:
                self._update_objects(*self._iterate_bashlex(node))

        elif '=' in bashlex_object.word and \
        '==' not in bashlex_object.word:
            l = bashlex_object.word.split('=')

            if 'chmod' in bashlex_object.word.lower():
                local_counter[l[0]] += 1
                local_counter['='.join(l[1:])] += 1
                local_tokens.extend([l[0], '='.join(l[1:])])

            elif len(l) == 2:
                local_counter[l[0]] += 1
                local_counter[l[1]] += 1
                local_tokens.extend([l[0], l[1]])

            else:
                local_counter[bashlex_object.word] += 1
                local_tokens.append(bashlex_object.word)

        else:
            local_counter[bashlex_object.word] += 1
            local_tokens.append(bashlex_object.word)

        return local_counter, local_tokens

    def tokenize_corpus(self, corpus):
        l = len(corpus)
        for i, command in enumerate(corpus):
            if self.verbose:
                print(f"[*] Parsing in process: {i+1}/{l}", end="\r")

            self.tokenized_corpus.append(self.tokenize_command(command, i=i))
        return self.tokenized_corpus, self.global_counter

    def tokenize_command(self, command, i="*"):
        if command in self.cache:
            # 如果命令已经在缓存中，直接返回缓存的分词结果
            return self.cache[command]
        else:
            tokenized_command = []
            nodes = self._bashlex_wrapper(command)

            if isinstance(nodes[0], str):
                logging.debug(f"[{i}] 'bashlex' failed, regex tokenization:\n\t{command}\n\t{nodes}\n")

                tokenized_command.extend(nodes)
                self.global_counter += Counter(nodes)

            elif isinstance(nodes[0], bashlex.ast.node):
                for node in nodes:
                    self.tokenized_node = []
                    self._update_objects(*self._iterate_bashlex(node))
                    tokenized_command.extend(self.tokenized_node)
            else:
                logging.info(f"[-] Unexpected return type from 'bashlex', skipping command:\n\t{command}")

            self.cache[command] = tokenized_command
            return tokenized_command

    def tokenize(self, data):
        processed_data = self.preprocess_data(data)
        return self.tokenize_corpus(processed_data)


class ShellEncoder():

    def __init__(self, shell_tokenizer, raw_data):
        self.shell_tokenizer = shell_tokenizer
        self.raw_data = raw_data
        self.X_tfidf = None
        self.X_hashing = None
        self.X_maxx = None

    def tf_hash(self):
        # 使用TfidfVectorizer获取TF-IDF特征向量
        tv = TfidfVectorizer(
            lowercase=False,
            tokenizer=self.shell_tokenizer.tokenize_command,
            token_pattern=None,
            max_features=500
        )
        self.X_tfidf = tv.fit_transform(self.raw_data)
        print(self.X_tfidf.shape)

        # 使用HashingVectorizer获取Hashing特征向量
        hv = HashingVectorizer(
            lowercase=False,
            tokenizer=self.shell_tokenizer.tokenize_command,
            token_pattern=None,
            n_features=500
        )
        self.X_hashing = hv.fit_transform(self.raw_data)
        print(self.X_hashing.shape)
        # 将TF-IDF特征向量和Hashing特征向量相加
        self.X_maxx = self.X_tfidf + self.X_hashing

        return self.X_maxx