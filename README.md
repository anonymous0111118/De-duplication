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

**We have tracked the changes in attention for each token in the fitting process. Tokens displayed in red indicate an increase in attention during fitting, while tokens displayed in black indicate a decrease in attention during fitting.**
* Case No.1
  
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/a8edf745-f992-4197-9843-5e8756e3745a" alt="Case 1" width="300" height="300">

After our **manual analysis**, this example triggered a compiler bug because of the way the structure is defined (two integer member variables) and the different initial values of a and b global variables. Through cross-copying in the main function, due to 06 The assignment statement line b=b is meaningless, but it triggers the compiler's optimization, and only this line is retained in the consecutive assignment statements, and is finally shown through the print function of b.d.

The heatmap diagram of **attention** shows that the structure is defined, and the initial value of the second member variable d of global variables a and b is different (since a does not give any initial value, the model can only enhance the attention when defining a), The additional assignment statement in line 06 and the final output statement for b.d together constitute the triggering semantics of this bug.

In this example, attention is consistent with the results of manual analysis, proving that the failure-relevant semantics extracted by BLADE is effective.

* Case No.2
