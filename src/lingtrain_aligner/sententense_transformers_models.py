import os
import pickle

from lingtrain_aligner.helper import lazy_property
from sentence_transformers import SentenceTransformer
import torch

from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# torch.backends.quantized.engine = 'qnnpack'

SENTENCE_TRANSFORMERS_MODEL_PATH = "./models/sentence_transformers-v2.bin"
SENTENCE_TRANSFORMERS_XLM_100_MODEL_PATH = "./models/sentence_transformers_xlm_100.bin"
SENTENCE_TRANSFORMERS_LABSE_MODEL_PATH = "./models/labse.bin"
SENTENCE_TRANSFORMERS_RUBERT_TINY_MODEL_PATH = "./models/rubert-tiny.bin"


class SentenceTransformersModel:
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_MODEL_PATH):
            print(
                f"Loading saved distiluse-base-multilingual-cased-v2 model. Device: {device}"
            )
            # self.model = torch.quantization.quantize_dynamic(pickle.load(open(SENTENCE_TRANSFORMERS_MODEL_PATH, 'rb')), {torch.nn.Linear}, dtype=torch.qint8)
            _model = pickle.load(open(SENTENCE_TRANSFORMERS_MODEL_PATH, "rb"))
            _model._target_device = device  # patch
        else:
            print("Loading distiluse-base-multilingual-cased-v2 model from Internet.")
            _model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        return _model

    def embed(self, lines, batch_size, normalize_embeddings, show_progress_bar):
        vecs = self.model.encode(
            lines,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return vecs


class SentenceTransformersModelXlm100:
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_XLM_100_MODEL_PATH):
            print(
                f"Loading saved xlm-r-100langs-bert-base-nli-mean-tokens model. Device: {device}"
            )
            # self.model = torch.quantization.quantize_dynamic(pickle.load(open(SENTENCE_TRANSFORMERS_MODEL_PATH, 'rb')), {torch.nn.Linear}, dtype=torch.qint8)
            _model = pickle.load(open(SENTENCE_TRANSFORMERS_XLM_100_MODEL_PATH, "rb"))
            _model._target_device = device  # patch
        else:
            print(
                "Loading xlm-r-100langs-bert-base-nli-mean-tokens model from Internet."
            )
            _model = SentenceTransformer("xlm-r-100langs-bert-base-nli-mean-tokens")
        return _model

    def embed(self, lines, batch_size, normalize_embeddings, show_progress_bar):
        vecs = self.model.encode(
            lines,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return vecs


class SentenceTransformersModelLaBSE:
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_LABSE_MODEL_PATH):
            print(f"Loading saved LaBSE model. Device: {device}")
            _model = pickle.load(open(SENTENCE_TRANSFORMERS_LABSE_MODEL_PATH, "rb"))
            _model._target_device = device  # patch
        else:
            print("Loading LaBSE model from Internet.")
            _model = SentenceTransformer("LaBSE")
        return _model

    def embed(self, lines, batch_size, normalize_embeddings, show_progress_bar):
        vecs = self.model.encode(
            lines,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return vecs


class RuBertTinyModel:
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_RUBERT_TINY_MODEL_PATH):
            print("Loading saved rubert tiny model.")
            _model = pickle.load(
                open(SENTENCE_TRANSFORMERS_RUBERT_TINY_MODEL_PATH, "rb")
            )
        else:
            print("Loading rubert tiny model from Internet.")
            _model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        return _model

    @lazy_property
    def tokenizer(self):
        print("Loading rubert tiny tokenizer from Internet.")
        _tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        return _tokenizer

    def embed(self, lines, batch_size, normalize_embeddings, show_progress_bar):
        vecs = []
        for text in lines:
            t = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                model_output = self.model(**t)
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)
            vecs.append(embeddings[0].cpu().numpy())
        return vecs


sentence_transformers_model = SentenceTransformersModel()
sentence_transformers_model_xlm_100 = SentenceTransformersModelXlm100()
sentence_transformers_model_labse = SentenceTransformersModelLaBSE()
rubert_tiny = RuBertTinyModel()


# print(os.getcwd())

# _model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# with open('distiluse-base-multilingual-cased-v5.bin', 'wb') as handle:
#     pickle.dump(_model, handle)


# _model = pickle.load(open("F:\git\lingtrain-aligner-editor\be\models\distiluse-base-multilingual-cased-v2", 'rb'))
# torch.save(_model.state_dict(), 'distiluse-base-multilingual-cased-v4.bin')
