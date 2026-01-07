from dataclasses import dataclass, field
from methods.base.config import CLSConfig

@dataclass
class SNAPConfig(CLSConfig):
    use_numerical_embedding: bool = field(
        default=True,
        metadata={"help": "Whether to use numerical embeddings in the prompt encoder."},
    )
    use_multi_head_self_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use multi-head self-attention in the prompt encoder."}
    )
    num_features: int = field(
        default=23,
        metadata={"help": "The number of features in the dataset used for numerical embeddings."},
    )
    head_dim: int = field(
        default=128,
        metadata={"help": "The dimension of each attention head in the multi-head self-attention."},
    )
    attention_bias: bool = field(
        default=False,
        metadata={"help": "Whether to include bias terms in the projection layer of the multi-head self-attention."},
    )
    attention_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout rate for attention layers in the multi-head self-attention."},
    )
