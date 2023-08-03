# De-duplication
This is the implementation of Black-box De-duplication of Silent Compiler Bugs via Deep Semantic Analysis
# Data Preparation
We have released our failure-triggering test programs and transformed failure-free ones data at ....
# Code Representation and Classification
Run classification.py to get the trained classification model

Note that we have released the parameters of the classification model trained on all four datasets. You can find them at ....
# Feature Extraction
Run get_distance.py to get the distance matrix

In directory 'distances', you can find the distance matrixs on four datasets and calculcated by five techniques : 'BLADE', 'Tamer', 'Trans', 'D3', 'D3-prog'
# Prioritization
run prioritization.py to get the final result

The jpg will be saved and you can check it.
