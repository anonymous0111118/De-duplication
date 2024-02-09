# De-duplication
This is the implementation of BLADE (Black-box De-duplication of Silent Compiler Bugs via Deep Semantic Analysis).
# Data Preparation
We have released our failure-triggering test programs and transformed failure-free ones data at [data.zip](https://drive.google.com/file/d/1bkSBTMpuV5_5wdpzJshcHKvOjSsKsLjw/view?usp=drive_link) 
# Code Representation and Classification
Run classification.py to get the trained classification model.
Before that you can log in to [Hugging Face](https://huggingface.co/) to choose a pre-trained code representation model that you prefer. We use UniXCoder.

Note that we have released the parameters of the classification model trained on all four datasets. If you want to skip the training phase, you can use the parameters we trained directly. You can find them at [model](https://drive.google.com/drive/folders/1KAiOzVI-XmD_POtJa6xANr702DFNYos3?usp=sharing).
# Feature Extraction
Run get_distance.py to get the distance matrix.

We have re-implement the other four techniques based on our four datasets. If you want to skip this step, in directory 'distances', you can find the distance matrixs on four datasets and calculcated by five techniques : 'BLADE', 'Tamer', 'Trans', 'D3', 'D3-prog' with their numpy files.


# Prioritization
Run prioritization.py to get the final result.

The figures will be saved and you can check it.

# Measure effectiveness in capturing causal relationships.
It's hard to directly measure the accuracy of captured causal relationships, since the root causes of bugs are described in natural languages within bug reports while the failure-relevant features extracted from test programs are represented as semantic vectors (typical outputs of code representation models). There is no automatic methods measuring their matching degrees. 
But we manually analyzed some cases to demonstrate the accuracy.
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/052cde89-1e2e-4fa0-bca8-df41438d1ee5" alt="Case 1" width="300" height="300">
