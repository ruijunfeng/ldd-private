# SNAP: Adapting LLMs for Credit Risk Assessment via Self-Attentive Numerical Alignment and Projection
## Scripts
nohup python methods.machine_learning.model &
nohup python methods.tabllm.model &
nohup python trainer.py --experiment_name calm &
nohup python trainer.py --experiment_name snap &
nohup python trainer.py --experiment_name snap --use_numerical_embedding &
nohup python trainer.py --experiment_name snap --use_multi_head_self_attn &
nohup python trainer.py --experiment_name snap --use_numerical_projector &
## Evaluation Setups
Area Under the Curve, Kolmogorov–Smirnov
## Research Questions
### RQ1: ablation study
Evaluate the effectiveness of each components:
w/o SNAP (pure lora)
w/o Numerical Embedding (use 23 plain embeddings to replace it)
w/o Multi-Head Self-Attention (use numerical embedding and numerical projector)
w/o Numerical Projector (use numerical embedding and multi-head self-attention)
SNAP
### RQ2: performance analysis
Traditional machien learning models, zero-shot prompting (TabLLM), lora (CALM), and SNAP
### RQ3: feature robustness
Delete feature columns under different proportion and see the performance changes (25%, 50%, 75%)
## Title Alternatives
Breaking Numerical Blindness: Intra-Numerical Prompt Tuning for Credit Risk Assessment
Overcoming Numerical Blindness of LLMs in Credit Risk Assessment
Beyond Textual Semantics: Learning Numerical Feature Interactions with LLMs for Credit Risk Assessment
The Language of Risk: Teaching LLMs to Understand Numerical Interactions in Credit Data
Closing the Gap: Enabling Large Language Models to Reason with Numerical Features in Credit Risk Assessment
