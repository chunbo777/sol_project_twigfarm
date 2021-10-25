import torch
import torch.nn as nn
from transformers import BertModel,AutoModelWithLMHead
from transformers import BartModel #clf 모델로 바트 추가
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from bert_pretrained.tokenizer import bert_tokenizer,  bart_tokenizer
from options import args


if args.language == 'ko':
    model_type = 'monologg/kobert'
    # model_type="beomi/kcbert-base"
else:
    model_type = 'bert-base-cased'
BERT = BertModel.from_pretrained(model_type).to(args.device)
# BERT = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base")
# if args.language == 'ko':
#     model_type = "monologg/koelectra-base-v3-discriminator"
# BERT = ElectraModel.from_pretrained(model_type).to(args.device)
BART = BartModel.from_pretrained(get_pytorch_kobart_model())


def get_bert_word_embedding():
    num_embeddings = (bert_tokenizer.vocab_size
                      + len(bert_tokenizer.get_added_vocab())) #vocab_size + bos, eos
    embed_dim = BERT.embeddings.word_embeddings.embedding_dim

    # need to add embedding for bos and eos token
    embedding = nn.Embedding(
        num_embeddings,
        embed_dim,
        padding_idx=bert_tokenizer.pad_token_id
    )
    embedding.weight.data[:bert_tokenizer.vocab_size].copy_(
        BERT.embeddings.word_embeddings.weight.data
    )
    return embedding


@torch.no_grad()
def extract_features(text):
    if args.clf_model == "bert":
        inputs = bert_tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True
        ).to(args.device)
        features = BERT(**inputs)[0]
        return features.squeeze(0).mean(0)
    if args.clf_model == "bart":
        inputs = bart_tokenizer(
                [text]
                # ,add_special_tokens=True
                ,return_tensors='pt'
                # , padding=True
        ).to(args.device)
        inputs = inputs["input_ids"]
        features = BART(inputs)[0]
        # sentence embedding = mean of word embedding
        return features.squeeze(0).mean(0)
