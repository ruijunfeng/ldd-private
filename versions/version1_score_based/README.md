# LDD: Local Default Density
## Method
Convert the training data into a vector database. When a new profile is used as a query, retrieve the most similar profiles.
Profiles with label 1 receive a score of 1, and profiles with label 0 receive a score of –1.
Use cosine similarity × score; if the sum of scores is greater than 0, the prediction is more likely to be 1 (the default).
## Conclusion
Setting top_k = 40 yields the best results.