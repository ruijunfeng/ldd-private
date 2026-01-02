# ORBIT: Observing Risk via Behavioral Interaction of Traits
## Methods
Single classifier: Use profile embeddings with a single linear layer.
Multiple classifiers: Use separate classifiers to classify profile embeddings, context embeddings, and top-k scores. And average over their logits as the final results.
Context classifier: Compute context embeddings using a weighted sum of the top-k embeddings, concatenate them with the profile embeddings, and feed the result into a classifier.
Cross-attention classifier: Use embeddings from similar profiles as key–value pairs to perform cross-attention. Conceptually, this acts as a context classifier: the query is replaced with a version combined with the instruction, and the attention scores are replaced with top-k cosine-similarity scores.
Training data as database. For each profile as query, retrieve the most similar profiles.
## Conclusion
Using one layer and mapping it to a scalar works best, with lr=1e-4, batch_size=64, epochs=1000, patience=20, topk=40.
Do not integrate embeddings from the applicant's top-k neighbors; neither concatenating them nor adding them yields good results.
