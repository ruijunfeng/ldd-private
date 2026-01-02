from dataclasses import dataclass, field

question_template = """You are an expert credit risk assessment model.
Your task is to determine whether this user should be classified as good credit or bad credit.
A good credit means the user is likely to repay reliably.
A bad credit means the user is high-risk and likely to default.

Given the following user features:
{profile}

Is this user's credit good or bad?
"""

@dataclass
class SeqCLSConfig():
    model_name: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "The name of the model to be used for zero-shot evaluation."},
    )
    question_template: str = field(
        default=question_template,
        metadata={"help": "The template for the question prompt."},
    )
    num_labels: str = field(
        default=2,
        metadata={"help": "The number of labels for classification."},
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "The learning rate for the optimizer."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size for training."},
    )
    accumulate_grad_batches: int = field(
        default=16,
        metadata={"help": "The number of batches to accumulate gradients over."},
    )
    max_epochs: int = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
