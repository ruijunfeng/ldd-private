from dataclasses import dataclass, field

@dataclass
class OrbitConfig():
    model_name: str = field(
        default="Qwen/Qwen3-Embedding-8B",
        metadata={"help": "The name of the pre-trained model to use for embeddings."},
    )
    instruction: str = field(
        default="Given an applicant's credit profile, retrieve other applicant credit profiles that are similar or related to this one.",
        metadata={"help": "The instruction for retrieving similar profiles."},
    )
    hidden_dim: int = field(
        default=4096,
        metadata={"help": "The hidden dimension size of the model embeddings."},
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "The dropout rate for the classifier."},
    )
    top_k: int = field(
        default=40,
        metadata={"help": "The number of top similar profiles to consider for context embeddings."},
    )
    lr: float = field(
        default=1e-3,
        metadata={"help": "The learning rate for the optimizer."},
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "The batch size for training."},
    )
    patience: int = field(
        default=20,
        metadata={"help": "The number of epochs with no improvement after which training will be stopped."},
    )
    max_epochs: int = field(
        default=500,
        metadata={"help": "The number of training epochs."},
    )
