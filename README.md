# De-duplication
This is the implementation of BLADE (Black-box De-duplication of Silent Compiler Bugs via Deep Semantic Analysis).
# Data Preparation
We have released our failure-triggering test programs and transformed failure-free ones data at [data.zip](https://drive.google.com/file/d/1bkSBTMpuV5_5wdpzJshcHKvOjSsKsLjw/view?usp=drive_link) 
# Code Representation and Classification
Run 
```$ python classification.py``` to get the trained classification model.
Before that you can log in to [Hugging Face](https://huggingface.co/) to choose a pre-trained code representation model that you prefer. We use UniXCoder.

Note that we have released the parameters of the classification model trained on all four datasets. If you want to skip the training phase, you can use the parameters we trained directly. You can find them at [model](https://drive.google.com/drive/folders/1KAiOzVI-XmD_POtJa6xANr702DFNYos3?usp=sharing).
# Feature Extraction
Run
```$ python get_distance.py``` to get the distance matrix.

We have re-implement the other four techniques based on our four datasets. If you want to skip this step, in directory 'distances', you can find the distance matrixs on four datasets and calculcated by five techniques : 'BLADE', 'Tamer', 'Trans', 'D3', 'D3-prog' with their numpy files.


# Prioritization
Run 
```$ python prioritization.py``` to get the final result.

The figures will be saved and you can check it.

# Measure effectiveness in capturing causal relationships.
It's hard to directly measure the accuracy of captured causal relationships, since the root causes of bugs are described in natural languages within bug reports while the failure-relevant features extracted from test programs are represented as semantic vectors (typical outputs of code representation models). There is no automatic methods measuring their matching degrees. 
But we manually analyzed some cases to demonstrate the accuracy.

**During the fitting process, we meticulously monitored the dynamics of attention for each token. Tokens that exhibited heightened attention were visually marked in a vibrant shade of red, signifying an increase in attention. Conversely, tokens that experienced diminished attention were visually represented in a stark black hue, indicating a decrease in attention.**

* Case No.1
  

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/fcbfd409-1a01-4d63-90d4-a6ac9d26226d" alt="Case 3" width="300" height="300">

The fundamental **root cause** of the bug, which was triggered by the execution of this program, stems from three key factors:
1. Firstly, the definition of the structure itself, characterized by the presence of two integer member variables. 
2. Secondly, the distinct initial values assigned to the global variables *a* and *b*. 
3. And finally, the occurrence of a cross-assignment statement located on lines 05 and 06, contributing to the manifestation of the bug.
<!-- The **root cause** of the bug triggered by this program is
1. the structure's definition (two integer member variables),
2. the distinct initial values of global variables *a* and *b*
3. and the statement of cross-assignment on lines 05 and 06. -->

When the -O2 optimization level was applied, an erroneous optimization occurred leading to the incorrect elimination of line 5. This optimization was triggered after the cross-assignment of structure type objects a and b, both containing two member variables.

The heatmap representation of the **attention** diagram elucidates that:
1. The model accurately detects the presence of a particular structure type defined with two member variables at line 01.
2. Subsequently, the model identifies a discrepancy in the initialization of *a* and *b*, specifically observing that the values of the second member variables differ. This distinction is visually depicted as *0* appearing in black while *1* is highlighted in red within *{0,1}* at line 02.
3. The model exhibits heightened focus on the cross-assignment operation between *a* and *b* as indicated by line 05 and line 06.
   
In essence, through the meticulous analysis of attention patterns, the model comprehends the failure-relevant semantics associated with the cross-assignment of variables a and b within two distinct structure types, featuring divergent values for the second member variable. This alignment between attention and the root cause effectively validates the efficacy of the failure-relevant semantics extracted by BLADE.




* Case No.2

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/63ebc5e6-0dfb-4259-a460-1e9f18ab72b2" alt="Case 3" width="300" height="300">


This example triggers the same bug as case 1, and the **root cause** of the bug was identical in both cases.

Furthermore, by examining the heatmap of **attention**, it becomes evident that the model assigns a similar level of attention to this example as it did to case 1. This suggests that the model recognizes the shared patterns and relevance between the two cases, reinforcing the consistency of its attention mechanism.

* Case No.3
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/20bc675b-6d09-4e22-aa20-085305a014fb" alt="Case 4" width="300" height="300">

First and foremost, to ensure clarity of understanding, it is imperative to acknowledge a prerequisite knowledge in the realm of the C language. By default, the *char* type is considered *signed* within this language. Consequently, when comparing a *char* type with an *unsigned char* type, implicit conversion takes place, wherein the *unsigned char* is converted to a *char*.

The **root cause** of the bug, which was triggered by the execution of this program, can be attributed to the following factors:

1. The presence of a global variable, denoted as *a*.
2. Declaration of an *unsigned char* type variable, *b*, assigned a value of *254*. It is important to note that this value exceeds *127*, and when compared to a *char* variable, it should be converted to *-2* using the formula *254 - 266 = -2*.
3. A *char* type variable, denoted as *c*.
4. At least one of the variables *b* and *c* is declared as *const*.
5. The comparison between *b* and *c* takes place, and under the -O2 optimization level, the value of b fails to be correctly converted to *-2* but instead retains the value *254*.
6l Finally, the result of the comparison is assigned to the variable *a*.

Regarding the heatmap changes in **attention**, it is evident that:
1. The model discovers the declaration of *a*, but does not pay attention to its type.( *int* is black but *a* is red in line 01)
2. The model focuses on the whole declaration statement of *b*, including its type and the initial value assigned to it *254*, which is likely to indicate that the model has noticed the semantics that b is greater than *127*. (line 02)
3. The model finds the declaration of *c* but ignores its initial value. Combining the previous line, it can be seen that the model knows this semantics, that is, it only needs to consider whether the assignment of the unsigned char type is greater than *127*. (black *0* in line 03) (red *254* in line 02)
4. The model focuses on *const*. (line 03)
5. The model pays more attention to the execution of the *d* function with a comparison between *b* and *c*. (line 07)
6. Through the red *= e*, model knows the comparision result of *b* and *c* is assigned to the gloval variable *a* in function *d*. (line 04)

To summarize, a comprehensive analysis of attention reveals the failure-relevant semantics discerned by the model. Specifically, it identifies the disparity that arises when comparing an unsigned char type variable, surpassing the value of *127*, with a char type variable, while incorporating the presence of at least one of them as const. Notably, this discrepancy manifests differently across various optimization levels. This alignment between attention patterns and the underlying root cause substantiates the efficacy of the failure-relevant semantics extracted by BLADE, thus providing compelling evidence in support of its effectiveness.

* Case No.4
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/6ecdf7e2-c55c-49f8-ad10-eb983de8cfb0" alt="Case 4" width="300" height="300">

It was determined that this example encountered the same bug as case 3, with the **root cause** being identical.

Despite the *if-statements* in this program, its logical structure aligns with that of case 3. The model doesn't pay much attention to the *if-statement*. It primarily focuses on the definitions and comparisons involving *char* and *const unsigned char* and the value of *unsigned char*. This demonstrates the model's accurate identification of the genuine failure-relevant semantics within this set of examples.


* Case No.5


<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/2cd5e487-a07c-4a09-b11c-4122258c7616" alt="Case 2" width="500" height="500">

Let us embark on the analysis of a more intricate example. The bug triggered by this program can be traced back to its **root cause**, which consists of the following elements:
1. The presence of two char type pointers, *c* and *e*, alongside a general pointer, *d*.
2. Within the initial code block, the value pointed to by the address stored in *d* is altered to *1*.
3. Declaration of a *char* type variable, *g*, without assigning it a specific value.
4. Numerous complex factors contribute to the optimization process, such as the potential assignment of *e* or the utilization of the address of *g*. Ultimately, under the -O2 optimization level, *g*'s address is erroneously assigned by *d*, leading to *g* assuming a value of *1*.

By scrutinizing the heatmap of **attention**, we can extract the following insights:
1. The model identifies the declaration of *a*, *b*, the pointer *d*, and the *char* type of the variable *c*. This spans lines 01 to 03.
2. In the initial code block, the model exhibits attention solely towards the value alteration within the address pointed to by *d*, inadvertently overlooking other relevant aspects. This is indicated by the black coloration of *f* in lines 06 and 07.
3. The model places considerable emphasis on the entire declaration statement of the char variable *g*, as evidenced by the uniform red shading in line 10. This observation may signify the model's recognition of the varying values assigned to *g* across different optimization levels, with g's value ultimately influencing the program's output.
4. The model identifies additional assignment statements that play a pivotal role in triggering the optimization process, such as the potential utilization of *g*'s address in line 13 and the possibility of altering the value of the global char pointer, *e*. Furthermore, line 14 directly influences the final output.


In summary, through a meticulous examination of attention patterns, the model discerns the failure-relevant semantics, which revolve around an uninitialized *char* type variable occupying the address of a global pointer that points to another modified value. These findings align remarkably well with the root cause, thereby substantiating the model's exceptional capability to elucidate failure-relevant semantics.




