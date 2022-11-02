from tqdm.notebook import trange
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

class TextTranslator:
    
    def __init__(self, batch_size=8, max_len=64):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B", cache_dir="/mnt/sentence_transformers_models").to(self.device)
        self.m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", cache_dir="/mnt/sentence_transformers_models")
        self.batch_size = batch_size
        self.max_len = max_len
    
    def translate_text(self, source_text_list, source_lang, target_lang, show_progress_bar=False):
        translated_text_list = []
        
        iterator_range = trange if show_progress_bar else range
        
        self.m2m_tokenizer.src_lang = source_lang
        
        for i in iterator_range(0, len(source_text_list), self.batch_size):
            
            source_text_batch = source_text_list[i:(i+self.batch_size)]
            
            # translate source_lang to target_lang
            encoded_text = self.m2m_tokenizer(source_text_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
            
            # Put inputs onto GPU (if available)
            encoded_text = {key: tensor.to(self.device) for key, tensor in encoded_text.items()}
            generated_tokens = self.m2m_model.generate(**encoded_text, forced_bos_token_id=self.m2m_tokenizer.get_lang_id(target_lang))
            
            # Put inputs and outputs onto CPU
            encoded_text = {key: tensor.to("cpu") for key, tensor in encoded_text.items()}
            generated_tokens = generated_tokens.to("cpu")
            
            translated_text_list.extend(self.m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
            
        return translated_text_list