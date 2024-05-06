from pykeen.models import ERModel, TransE, DistMult
from pykeen.nn import Embedding
from pykeen.nn.modules import Interaction, NormBasedInteraction
from torch import FloatTensor
from pykeen.pipeline import pipeline
from class_resolver import Hint, HintOrType, OptionalKwargs
from torch.nn import functional
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_, xavier_normal_norm_
from pykeen.typing import Constrainer, Initializer
from pykeen.regularizers import Regularizer, LpRegularizer
from typing import Union, Any, ClassVar, Mapping
from pykeen.utils import negative_norm_of_sum, tensor_product
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE

def kgcmodel_interaction(
    h: FloatTensor,
    r: FloatTensor,
    t: FloatTensor,
    p: Union[int, str] = 2,
    power_norm: bool = False,
) -> FloatTensor:
    return (tensor_product(h, r, t).sum(dim=-1) * negative_norm_of_sum(h, r, -t, p=p, power_norm=power_norm))

class KGCModelInteraction(NormBasedInteraction[FloatTensor, FloatTensor, FloatTensor]):
    
    func = kgcmodel_interaction


class KGCModel(ERModel):

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        regularizer: HintOrType[Regularizer] = LpRegularizer,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:

        if regularizer is LpRegularizer and regularizer_kwargs is None:
            regularizer_kwargs = DistMult.regularizer_default_kwargs

        super().__init__(
            interaction=KGCModelInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations=Embedding,
            entity_representations_kwargs=dict(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
            ),
            relation_representations=Embedding,
            relation_representations_kwargs=dict(
                embedding_dim=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            **kwargs,
        )

if __name__ == '__main__':

    result_KGCModel = pipeline(
        model=KGCModel,
        dataset='nations',
        training_kwargs={'num_epochs':100},
        random_seed=1603073093
    )
    
    print(f"MRR: {result_KGCModel.metric_results.to_flat_dict()['both.realistic.inverse_harmonic_mean_rank']}")
    for k in [1,3,5,10]:
        print(f"Hits@{k} : {result_KGCModel.metric_results.to_flat_dict()['both.realistic.hits_at_'+str(k)]}")