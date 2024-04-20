# the bi-encoder model architecture (which consists of a query/a document encoder)
# from .inbatch import InBatch
# [todo] try to merge this into the first one.
from .single_inbatch import InBatchInteraction, InBatchInteractionWithSpan
from .multi_inbatch import InBatchLateInteraction

# the encoder models
from ._contriever import Contriever
