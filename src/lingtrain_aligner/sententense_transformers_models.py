import os
import pickle

from lingtrain_aligner.helper import lazy_property
from sentence_transformers import SentenceTransformer
from torch import device

# torch.backends.quantized.engine = 'qnnpack'

SENTENCE_TRANSFORMERS_MODEL_PATH = './models/sentence_transformers-v2.bin'
SENTENCE_TRANSFORMERS_XLM_100_MODEL_PATH = './models/sentence_transformers_xlm_100.bin'
SENTENCE_TRANSFORMERS_LABSE_MODEL_PATH = './models/labse.bin'


class SentenceTransformersModel():
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_MODEL_PATH):
            print("Loading saved distiluse-base-multilingual-cased-v2 model.")
            # self.model = torch.quantization.quantize_dynamic(pickle.load(open(SENTENCE_TRANSFORMERS_MODEL_PATH, 'rb')), {torch.nn.Linear}, dtype=torch.qint8)
            _model = pickle.load(open(SENTENCE_TRANSFORMERS_MODEL_PATH, 'rb'))
            _model._target_device = device("cpu")  # patch
        else:
            print("Loading distiluse-base-multilingual-cased-v2 model from Internet.")
            _model = SentenceTransformer(
                'distiluse-base-multilingual-cased-v2')
        return _model

    def embed(self, lines):
        vecs = self.model.encode(lines)
        return vecs


class SentenceTransformersModelXlm100():
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_XLM_100_MODEL_PATH):
            print("Loading saved xlm-r-100langs-bert-base-nli-mean-tokens model.")
            # self.model = torch.quantization.quantize_dynamic(pickle.load(open(SENTENCE_TRANSFORMERS_MODEL_PATH, 'rb')), {torch.nn.Linear}, dtype=torch.qint8)
            _model = pickle.load(
                open(SENTENCE_TRANSFORMERS_XLM_100_MODEL_PATH, 'rb'))
            _model._target_device = device("cpu")  # patch
        else:
            print("Loading xlm-r-100langs-bert-base-nli-mean-tokens model from Internet.")
            _model = SentenceTransformer(
                'xlm-r-100langs-bert-base-nli-mean-tokens')
        return _model

    def embed(self, lines):
        vecs = self.model.encode(lines)
        return vecs


class SentenceTransformersModelLaBSE():
    @lazy_property
    def model(self):
        if os.path.isfile(SENTENCE_TRANSFORMERS_LABSE_MODEL_PATH):
            print("Loading saved LaBSE model.")
            _model = pickle.load(
                open(SENTENCE_TRANSFORMERS_LABSE_MODEL_PATH, 'rb'))
            _model._target_device = device("cpu")  # patch
        else:
            print("Loading LaBSE model from Internet.")
            _model = SentenceTransformer('LaBSE')
        return _model

    def embed(self, lines):
        vecs = self.model.encode(lines)
        return vecs


sentence_transformers_model = SentenceTransformersModel()
sentence_transformers_model_xlm_100 = SentenceTransformersModelXlm100()
sentence_transformers_model_labse = SentenceTransformersModelLaBSE()
