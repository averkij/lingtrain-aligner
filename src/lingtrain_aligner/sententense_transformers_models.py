import os
import pickle

from lingtrain_aligner.helper import lazy_property
from sentence_transformers import SentenceTransformer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
import torch

from tqdm.auto import tqdm
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
SONAR_MODEL_PATH = "./models/sonar.bin"


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
            _model = SentenceTransformer(
                "distiluse-base-multilingual-cased-v2", cache_folder="./models_cache"
            )
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
            _model = SentenceTransformer(
                "xlm-r-100langs-bert-base-nli-mean-tokens",
                cache_folder="./models_cache",
            )
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
            _model = SentenceTransformer("LaBSE", cache_folder="./models_cache")
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
            _model = AutoModel.from_pretrained(
                "cointegrated/rubert-tiny2", cache_dir="./models_cache"
            )
        return _model

    @lazy_property
    def tokenizer(self):
        print("Loading rubert tiny tokenizer from Internet.")
        _tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny2", cache_dir="./models_cache"
        )
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


class SonarModel:
    @lazy_property
    def model(self):
        if os.path.isfile(SONAR_MODEL_PATH):
            print("Loading saved SONAR model")
            _model = pickle.load(
                open(SONAR_MODEL_PATH, "rb")
            )
        else:
            print("Loading SONAR model from Internet.")
            _model = M2M100Encoder.from_pretrained(
                "cointegrated/SONAR_200_text_encoder", cache_dir="./models_cache"
            )
        return _model

    @lazy_property
    def tokenizer(self):
        print("Loading SONAR tokenizer from Internet.")
        _tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/SONAR_200_text_encoder", cache_dir="./models_cache"
        )
        return _tokenizer

    def embed(self, lines, batch_size, normalize_embeddings, show_progress_bar, lang="ell_Grek"):
        # Ideally, we should indicate the real language of the text when encoding it.
        # By default, we indicate greek, because with this language, it is the easiest for the model to understand that the language tag is wrong and ignore it.
        self.tokenizer.src_lang = lang
        vecs = []
        wrapped_lines = tqdm(lines) if show_progress_bar else lines
        for text in wrapped_lines:
            t = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                per_token_embeddings = self.model(**t).last_hidden_state
                mask = t.attention_mask
                embeddings = (per_token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings)
            vecs.append(embeddings[0].cpu().numpy())
        return vecs


sentence_transformers_model = SentenceTransformersModel()
sentence_transformers_model_xlm_100 = SentenceTransformersModelXlm100()
sentence_transformers_model_labse = SentenceTransformersModelLaBSE()
rubert_tiny = RuBertTinyModel()
sonar = SonarModel()


# print(os.getcwd())

# _model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# with open('distiluse-base-multilingual-cased-v5.bin', 'wb') as handle:
#     pickle.dump(_model, handle)


# _model = pickle.load(open("F:\git\lingtrain-aligner-editor\be\models\distiluse-base-multilingual-cased-v2", 'rb'))
# torch.save(_model.state_dict(), 'distiluse-base-multilingual-cased-v4.bin')
