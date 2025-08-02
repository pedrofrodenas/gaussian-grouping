import torch
import open_clip
import regex as re
import gzip
import html
import ftfy
import requests
from functools import lru_cache

# =================================================================================
# Helper Functions (extracted from the original source)
# =================================================================================

@lru_cache()
def bytes_to_unicode():
    """
    Returns a dictionary mapping utf-8 bytes to unicode strings.
    This is used to handle the BPE algorithm on a byte-level.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Return a set of symbol pairs in a word.
    A word is represented as a tuple of symbols (variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    """
    Applies basic text cleaning using ftfy and html unescaping.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    """
    Removes extra whitespace from text.
    """
    text = " ".join(text.split())
    text = text.strip()
    return text

# =================================================================================
# Replicated Tokenizer Class
# =================================================================================

class ReplicatedTokenizer:
    """
    A self-contained replication of the open_clip.SimpleTokenizer.
    It fetches the BPE vocabulary from the web and implements the exact
    same tokenization logic.
    """
    def __init__(self, context_length: int = 77):
        # --- Configuration ---
        self.context_length = context_length
        bpe_url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
        
        # --- Load BPE Merges ---
        try:
            response = requests.get(bpe_url)
            response.raise_for_status() # Raise an exception for bad status codes
            merges_gz = response.content
            merges_bytes = gzip.decompress(merges_gz)
            merges_text = merges_bytes.decode("utf-8")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading BPE vocab: {e}")
            raise
            
        merges = merges_text.split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]

        # --- Build Vocabulary and Encoders/Decoders ---
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
            
        special_tokens = ['<start_of_text>', '<end_of_text>']
        vocab.extend(special_tokens)

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # --- BPE Ranks and Regex Pattern ---
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        
        # --- Special Token IDs ---
        self.sot_token_id = self.encoder['<start_of_text>']
        self.eot_token_id = self.encoder['<end_of_text>']

    def bpe(self, token: str):
        """
        The core Byte-Pair Encoding algorithm.
        """
        if token in self.cache:
            return self.cache[token]
        
        # Add end-of-word marker and split into characters
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            # Find the next best pair to merge
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            # Merge the pair
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
                
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str):
        """
        Encodes a single string into a list of BPE token IDs.
        """
        bpe_tokens = []
        # 1. Clean the text (basic, whitespace, lowercase)
        text = whitespace_clean(basic_clean(text)).lower()
        
        # 2. Pre-tokenize using regex
        for token in re.findall(self.pat, text):
            # 3. Map to unicode bytes
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 4. Apply BPE and get token IDs
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            
        return bpe_tokens

    def __call__(self, texts, context_length: int = None):
        """
        The main entry point for tokenization. Handles batching, padding,
        and truncation.
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        
        all_tokens = []
        for text in texts:
            # 1. Encode text and add special SOT/EOT tokens
            encoded = [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            all_tokens.append(encoded)

        # 2. Create the final tensor with padding and truncation
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                # Truncate and ensure the last token is EOT
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            
            # Add to the result tensor
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

# =================================================================================
# Verification
# =================================================================================

if __name__ == '__main__':
    # --- Sample Input ---
    sample_texts = [
        "a photo of a cat",
        "a drawing of a dog on a skateboard",
        "This is a much longer sentence to test truncation and ensure everything works as expected."
    ]
    
    print("--- Verification ---")
    

    # --- 2. Use the replicated tokenizer ---
    print("\n2. Tokenizing with ReplicatedTokenizer...")
    replicated_tokenizer = ReplicatedTokenizer()
    replicated_tokens = replicated_tokenizer(sample_texts)
    print("Replicated tokenizer output shape:", replicated_tokens.shape)
    print("Replicated tokens (first example):\n", replicated_tokens[0])
    print("Replicated tokens (second example):\n", replicated_tokens[1])
    print("Replicated tokens (third example):\n", replicated_tokens[2])
