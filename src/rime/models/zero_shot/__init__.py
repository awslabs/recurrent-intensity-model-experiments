from rime.util import MissingModel
try:
    from .tf_idf import TF_IDF
    from .bayes_lm import BayesLM
    from .item_knn import ItemKNN
except ImportError as e:
    BayesLM = MissingModel("BayesLM", e)
    ItemKNN = MissingModel("ItemKNN", e)
