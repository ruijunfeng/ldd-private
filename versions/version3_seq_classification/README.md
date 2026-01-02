# Sequence Classfication
## Method
Format the problem into a sequence classfication task. The lm_head is replaced with score, where the output_dim of score is num_labels. 
Notice: In hugging face, for binary classfication, output logits is of shape 2 rather than 1. So softmax is needed rather than sigmoid. Hugging face do this to seprate with regression task.
The difference is how to decode the logits in module.py test_step().