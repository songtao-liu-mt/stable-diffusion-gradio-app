from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
class TranslatorWrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        print('loading translator...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
    def __call__(self, text):
        tokenized_text = self.tokenizer.prepare_seq2seq_batch([text], return_tensors='pt').to(self.device)
        translation = self.model.generate(**tokenized_text)
        return self.tokenizer.batch_decode(translation, skip_special_tokens=False)[0]