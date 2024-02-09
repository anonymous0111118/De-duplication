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

**We have tracked the changes in attention for each token during the fitting process. Tokens highlighted in
<font color="red">red indicate an increase in attention during fitting</font> 
, while tokens displayed in black indicate a decrease in attention during fitting.**

* Case No.1
  
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/a8edf745-f992-4197-9843-5e8756e3745a" alt="Case 1" width="300" height="300">

After our **manual analysis**, this example revealed a compiler bug caused by 
1. the structure's definition (two integer member variables),
2. the distinct initial values of global variables *a* and *b*
3. and the statement of cross-assignment on lines 05 and 06.

The root cause of this bug is that under the -O2 optimization level, after cross-assignment of structure type objects *a* and *b* with two member variables, line 5 was incorrectly optimized out.

The heatmap diagram of **attention** illustrates:
1. The model notices that a certain structure type had two member variables when it was defined. (line 01)
2. Immediately afterwards, the model finds that when *a* and *b* are initialized, the values of the second member variables were not equal. ( the *0* is black but *1* is red in *{0,1}* in line 02). 
3. The model pays close attention to the cross assignment operation between *a* and *b*. (line 05 & line 06)

In summary, through the analysis of attention, the failure-relevant semantics understood by the model are the cross assignment of variables *a* *b* of two structural types with different values of the second member variable, which demonstrates the alignment between attention and the results of manual analysis, thereby verifying the effectiveness of the failure-relevant semantics extracted by BLADE.




* Case No.2

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/e0e7d37e-3655-4da5-8162-01ca5277a5a0" alt="Case 3" width="300" height="300">

This example encountered the same bug as case 1, and after conducting a thorough **manual analysis**, it was discovered that the root cause of the bug was identical in both cases.

Furthermore, by examining the heatmap of **attention**, it becomes evident that the model assigns a similar level of attention to this example as it did to case 1. This suggests that the model recognizes the shared patterns and relevance between the two cases, reinforcing the consistency of its attention mechanism.

* Case No.3

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/6ecdf7e2-c55c-49f8-ad10-eb983de8cfb0" alt="Case 4" width="300" height="300">

Based on our **manual analysis** of this example, we have observed the following:

In the C language, the *char* type is considered *signed* by default. When comparing a *char* type with an *unsigned* char type, the *unsigned char* should be implicitly converted to *char*. Therefore, in the *if-statement*, the value of *b* should be *-1*. However, when *b* is defined as *const*, under the optimization level of -O2, it is compared with *c* as an integer with a wrong value of 255. This inconsistency leads to an incorrect output.

Regarding the heatmap changes in **attention**, it is evident that the model increasingly focuses on the definitions of *c* and *b*. In the *if-statement*, the model pays growing attention to the variables *c* and *b*. There, the model's comprehension of the distinct types involved suggests its recognition of the existence of type conversion operations, thereby capturing failure-relevant semantics.


* Case No.4
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/20bc675b-6d09-4e22-aa20-085305a014fb" alt="Case 4" width="300" height="300">

After conducting a thorough **manual analysis**, it was determined that this example encountered the same bug as case 3, with the root cause being identical.

Despite the absence of *if-statements* in this program, its logical structure aligns with that of case 3. Remarkably, the **attention** pattern exhibited by the model for this example closely resembles that of case 3. The model primarily focuses on the definitions and comparisons involving *char* and *const char*, without placing significant emphasis on the *if-expression* present in case 3. This demonstrates the model's accurate identification of the genuine failure-relevant semantics within this set of examples.


* Case No.5


<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/2cd5e487-a07c-4a09-b11c-4122258c7616" alt="Case 2" width="500" height="500">

Let's analyze a more complex example. After conducting a thorough **manual analysis**, we have determined that the integer value at the address pointed to by the global pointer *d* (lines 05-08) is assigned as *1*. After that, when *g* is declared in the 10th line, on -O2 optimization level, the address is wrongly assigned to *g*, resulting in the initial integer value of *g* being *1*. This initialization has a subsequent impact on the final output.

Based on my experience, I will provide you with some tips to better comprehend the heatmap of **attention** change:

1. Analyze the variable types that attract the model's attention. This examination will shed light on the specific data types the model focuses on.
2. Investigate the constant values on the right side of assignment statements that capture the model's attention. This exploration will help unveil the particular values that the model deems significant in relation to the code's execution.
3. Explore the complete statements to which the model assigns special attention. This analysis will offer valuable insights into the code sections that are crucial for the model's understanding and capturing of failure-relevant semantics.


We can find:
1. The model's attention is primarily directed towards specific variable types that are essential for triggering bugs. Surprisingly, modifying variable types that the model does not pay attention to can still trigger bugs. This suggests that the model's focus on certain variable types is crucial, while others may have indirect effects on bug performance.

2. The constants that capture the model's attention tend to exhibit bug-free behavior after modifications. Conversely, the constants that the model does not pay attention to are typically unnecessary for bug performance.

3. The model assigns significant attention to the declaration of variable *g*, as it serves as the **key trigger** for this bug. Interestingly, this bug does not require an explicit assignment statement but rather quietly occurs through the address assignment of *g* during declaration.
     
In the heatmap of **attention** changes, it is evident that the model allocates significant attention to assignment statements within the main function. Specifically, for integer assignments that impact the output results, such as in lines 6 and 14, the model shows distinct attention to the specific values on the right side of the assignments. Conversely, the assignment statement in line 12, which has no impact on the results, receives less attention from the model.

Moreover, the analysis reveals that the discrepancy in the initial value of *char g;* at different optimization levels in line 10 is the main reason for triggering this bug. Remarkably, the model accurately captures this distinctive characteristic. As a result, the model's explanation of failure-relevant semantics in this example is highly accurate.

**discussion**

The explanation method employed by BLADE is known for its high accuracy, primarily attributed to its emphasis on elucidating the last layer of the neural network. Attention, serving as a parameter of the upstream transformer, undergoes mapping to the input. This mapping process facilitates the capturing of not only the accuracy loss associated with downstream classification parameters but also the pronounced second part loss stemming from the parameters mapped onto the input. Consequently, the resulting explanation may not possess the desired level of intuitiveness. 

**Therefore, even though BLADE's AI explanation method differs from the human-understandable attention heatmap, it is more precise.**
