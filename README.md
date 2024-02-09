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

**We have tracked the changes in attention for each token in the fitting process. Tokens displayed in 
<font color="red">red indicate an increase in attention during fitting</font>
, while tokens displayed in black indicate a decrease in attention during fitting.**
* Case No.1
  
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/a8edf745-f992-4197-9843-5e8756e3745a" alt="Case 1" width="300" height="300">

After our **manual analysis**, this example revealed a compiler bug caused by the structure's definition (two integer member variables), the distinct initial values of global variables *a* and *b* and the statement of cross-copying on lines 05 and 06. During -O2 level optimization, the compiler selectively executes line 06 while disregarding line 05. Consequently, the wrong value of *b.d* is ultimately displayed through the print function, showcasing the influence of the compiler's optimization on -O2 level.

The heatmap diagram of **attention** illustrates the definition of the structure and highlights the differing initial values of the second member variable, d, within the global variables **a* and *b*. Since *a* does not provide an initial value, the model can only enhance attention when defining *a*. The cross assignment statement on lines 05 and 06, along with the final output statement for *b.d*, collectively contribute to triggering the semantics of this bug and attract the attention of the model. Surprisingly, in this example, the attention mechanism precisely recognizes the second value as the point of differentiation between variables *a* and *b* (indicated by the red 1 on line 02). This demonstrates the model's ability to accurately identify the crucial aspects of the code and highlight the relevant information for analysis.

This example demonstrates the alignment between attention and the results of manual analysis, thereby verifying the effectiveness of the failure-relevant semantics extracted by BLADE.




* Case No.2

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/e0e7d37e-3655-4da5-8162-01ca5277a5a0" alt="Case 3" width="300" height="300">

This example triggered the same bug as case 1. After **manual analysis**, they found that the root cause of the bug was also the same.

In addition, we can see from the heatmap of **attention** that the model pays the same attention to it as it does to case 1.

* Case No.3

<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/6ecdf7e2-c55c-49f8-ad10-eb983de8cfb0" alt="Case 4" width="300" height="300">




Our **manual analysis** of this example is as follows: In C language, the char type is regarded as a signed type by default. When the char type is compared with the unsigned char type, the unsigned char is converted to char by default, so the value of b in the if-statement should be -1, but when it is defined with const, b is not staged under the optimization level of O2, but is compared with c as an integer of value 255, and an incorrect output is obtained.

On the heatmap of **attention** changes, we can clearly find that the model pays more and more attention to the definitions of c and b, and in the if statement, the model pays more and more attention to the two variables c and b before and after, indicating that the model also believes that there is Type conversion operations can prove that the model captures failure-relevant semantics.


* Case No.4
<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/20bc675b-6d09-4e22-aa20-085305a014fb" alt="Case 4" width="300" height="300">

This example triggered the same bug as case 3. After **manual analysis**, they found that the root cause of the bug was also the same.

Although this program lacks the expression of conditional statements, it is logically the same as case 3. The model's **attention** change for this example is very similar to the case 3, and in this process, it only pays attention to char and const char definitions and also comparisons between them, but does not pay special attention to the expression of if-expressions in case4, which shows that the model accurately discovered the true failure-relevant semantics of this set of examples.


* Case No.5


<img src="https://github.com/anonymous0111118/De-duplication/assets/141200895/2cd5e487-a07c-4a09-b11c-4122258c7616" alt="Case 2" width="500" height="500">

Now let's analyze one example which is more complex. In this example, after our **manual analysis**, due to the assignment of the global pointer d in lines 05-08, the integer value at the address pointed to by d is 1. When g is declared in the 10th line, the address is given to g, so that the initial integer value of g is 1, which influences the output at last.

The heatmap of **attention** change is a bit hard to understand. Based on my experience, I will give you some tips.
  1. Analyze which variable types the model pays attention to.
  2. Analyze which constant values on the right side of the assignment statement the model pays attention to.
  3. Analyze which entire statement model pays special attention.

The answer is below:
  1. In this case, the variable types that the model pays attention to are necessary to trigger bugs, and the variable types that the model does not pay attention to can still trigger bugs after modification.
  2. The constants that the model pays attention to will not show bugs after modification, while the constants that the model does not pay attention to are unnecessary for the performance of bugs.
  3. The model pays special attention to the declaration of g, which is the top priority triggered by this bug. Because the triggering of this bug does not require an assignment statement, but the assignment through the address occurs quietly when it is declared.
     
In the heatmap changed by **attention**, we can see that in the main function, the model pays great attention to assignment statements. For integer assignments that have an impact on the output results, such as lines 6 and 14, the model pays special attention to the specific values on the right side. The specific assignment model of line 12 that has no impact on the results is not paid attention to. At the same time, we found that the main reason why this bug was triggered was the difference in the initial value of char g at different optimization levels in line 10. This feature was accurately captured by the model. Therefore, in this example, the model The explanation of failure-relevant semantics is quite correct.

**discussion**

BLADE's explanation method is very accurate, because we are explaining the last layer of the neural network, and attention is the parameter of the upstream transformer, which is mapped to the input. Not only is the accuracy loss of the downstream classification parameters , but also severe second part loss because it's mapped to the input, so it is very rough and therefore certainly not intuitive enough.
