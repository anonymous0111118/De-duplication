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

The **root cause** of the bug triggered by this program is
1. the structure's definition (two integer member variables),
2. the distinct initial values of global variables *a* and *b*
3. and the statement of cross-assignment on lines 05 and 06.

Under the -O2 optimization level, after cross-assignment of structure type objects *a* and *b* with two member variables, line 5 was incorrectly optimized out.

The heatmap diagram of **attention** illustrates:
1. The model notices that a certain structure type had two member variables when it was defined. (line 01)
2. Immediately afterwards, the model finds that when *a* and *b* are initialized, the values of the second member variables were not equal. ( the *0* is black but *1* is red in *{0,1}* in line 02). 
3. The model pays close attention to the cross assignment operation between *a* and *b*. (line 05 & line 06)

In summary, through the analysis of attention, the failure-relevant semantics understood by the model are the cross assignment of variables *a* *b* of two structural types with different values of the second member variable, which demonstrates the alignment between attention and the root cause, thereby verifying the effectiveness of the failure-relevant semantics extracted by BLADE.




* Case No.2

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/e0e7d37e-3655-4da5-8162-01ca5277a5a0" alt="Case 3" width="300" height="300">

This example encountered the same bug as case 1, and the **root cause** of the bug was identical in both cases.

Furthermore, by examining the heatmap of **attention**, it becomes evident that the model assigns a similar level of attention to this example as it did to case 1. This suggests that the model recognizes the shared patterns and relevance between the two cases, reinforcing the consistency of its attention mechanism.

* Case No.3
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/20bc675b-6d09-4e22-aa20-085305a014fb" alt="Case 4" width="300" height="300">

First of all, for ease to understand, there is a prerequisite knowledge：In the C language, the *char* type is considered *signed* by default. When comparing a *char* type with an *unsigned* char type, the *unsigned char* should be implicitly converted to *char*.

The **root cause** of the bug triggered by this program is:
1. There's a global variable *a*.
2. An unsigned char type variable *b* is declared and given a value *254* (which is greater than *127* and when it is compared to a char variable it should be converted to *254 - 266 = -2*).
3. A char type variable *c*.
4. At least one of *b* and *c* is *const*.
5. *b* is compared to *c* and in -O2 optimization level, the value of *b* isn't correctly coverted to *-2* but still *254*.
6. The camparasion result is assigned to *a*.

Regarding the heatmap changes in **attention**, it is evident that:
1. The model discovers the declaration of *a*, but does not pay attention to its type.( *int* is black but *a* is red in line 01)
2. The model focuses on the whole declaration statement of *b*, including its type and the initial value assigned to it *254*, which is likely to indicate that the model has noticed the semantics that b is greater than *127*. (line 02)
3. The model finds the declaration of *c* but ignores its initial value. Combining the previous line, it can be seen that the model knows this semantics, that is, it only needs to consider whether the assignment of the unsigned char type is greater than *127*. (black *0* in line 03) (red *254* in line 02)
4. The model focuses on *const*. (line 03)
5. The model pays more attention to the execution of the *d* function with a comparison between *b* and *c*. (line 07)
6. Through the red *= e*, model knows the comparision result of *b* and *c* is assigned to the gloval variable *a* in function *d*. (line 04)


In summary, through the analysis of attention, the failure-relevant semantics understood by the model are when comparing an unsigned char type variable exceeding 127 with another char type variable, adding at least one of them as const, results differently in different optimization levels, which shows the alignment between attention and the root cause, thereby demonstrating the effectiveness of the failure-relevant semantics extracted by BLADE.


* Case No.4
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/6ecdf7e2-c55c-49f8-ad10-eb983de8cfb0" alt="Case 4" width="300" height="300">

It was determined that this example encountered the same bug as case 3, with the **root cause** being identical.

Despite the *if-statements* in this program, its logical structure aligns with that of case 3. The model doesn't pay much attention to the *if-statement*. It primarily focuses on the definitions and comparisons involving *char* and *const unsigned char* and the value of *unsigned char*. This demonstrates the model's accurate identification of the genuine failure-relevant semantics within this set of examples.


* Case No.5


<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/2cd5e487-a07c-4a09-b11c-4122258c7616" alt="Case 2" width="500" height="500">

Let's analyze a more complex example. The **root cause** of the bug triggered by this program is:
1. There're *char* type pointer *c* and *e*, and a pinter *d*.
2. In the first code block, the value in the address *d* points changes to *1*.
3. *Char* type variable *g* is declared without a value.
4. There're some other complex elements trigger the optimization such as *e*'s possibly assignment or *g*'s address will be used. Finally in the -O2 optimization level, *g*'s address is assigned wrongly by *d* and *g*'s value is naturally *1*.

We can read the heat map of **attention** changed and find:
1. The model discovers the declaration of *a*, *b*, pointer *d*, and the point *c*'s type *char*. (line 01 ~ 03)
2. In the first block, the model pays only attention to the value changed in the address d points but ignores other things. (black *f* in line 06 & 07)
3. The model pays great attentions to the whole declation statement of *char g*. (all red in line 10. This may illustrate the model finds there the value of *g* given differently in different optimization levels. And it is *g*'s value which finally influences the output.)
4. The model finds some other assignment statements which are one part of reasons to trigger the optimization (like *g*'s address will be used in line 13, and global char pointer *e*'s value can be possibly changed.) or directly influence the final output (line 14).

In summary, through the analysis of attention, the failure-relevant semantics understood by the model are one *char* type variable without a initial value occupies the global pointer's address which points another modified value. These are highly coincident with the root cause, proving that the model's ability to explain failure-relevant semantics is excellent.





