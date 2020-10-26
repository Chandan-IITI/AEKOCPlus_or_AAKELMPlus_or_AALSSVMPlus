## AEKOC+ or AAKELM+ or AALSSVM+

Gautam, C., Tiwari, A., & Tanveer, M. (2020). AEKOC+: Kernel Ridge Regression-Based Auto-Encoder for One-Class Classification Using Privileged Information. Cognitive Computation, 1-14.

**AEKOC+ can also be treated as the variant of Kernel Extreme learning Machine or Least Squares SVM with zero bias, therefore, paper and method can also be named as follows:**

AAKELM+: Autoassociative Kernel Extreme Learning Machine based One-class Classification using Privileged Information 

or 

AALSSVM+: Autoassociative Least Square SVM with zero bias based One-class Classification using Privileged Information


## For reproducing the results of Heart datasets:

--  Open All_Heart_Experiments.ipynb in Python notebook and run all cells. It will save all results in .pkl files. Results on optimal   parameters along with optimal parameters values will be saved in a excel file.   

--  Be dfault these codes produce results for group attribute 'Age'. For other two group attributes (Sex and Electrocardiographic): change the value in cell number 3 and 4 as follows:

**For group attribute = Sex:**

Uncomment this line in cell 3:  
 privileged_space = privileged_space_tot.ix[:]['p1']

Uncomment this line in cell 4:
 feature_space = feature_space.drop('a2', axis=1)
 privfeat = 'Sex'

**For group attribute = Electrocardiographic:**

Uncomment this line in cell 3:  
 privileged_space = privileged_space_tot.ix[:]['p3']

Uncomment this line in cell 4:
 feature_space = feature_space.drop('a7', axis=1)
 privfeat = 'Elect'


### Following papers can be experimented by calling 'LUPI_oneclass_methods.py' file from the repository: 
    
 **Paper1 (KOC+):** Gautam, Chandan, Aruna Tiwari, and M. Tanveer. "KOC+: Kernel ridge regression based one-class classification using privileged information.", Information Sciences 504 (2019): 324-333.  

**Paper2 (OCKELM):** Leng, Qian, et al. "One-class classification with extreme learning machine.", Mathematical problems in engineering (2015).

**Paper3 (AEKOC+):** Gautam, Chandan, Aruna Tiwari, and M. Tanveer. "AEKOC+: Kernel ridge regression based Auto-Encoder for one-class classification using privileged information.", Cognitive Computation (2020): 1-14.

**Paper4 (AEKOC/AAKELM):** Gautam, Chandan, Aruna Tiwari, and Qian Leng. "On the construction of extreme learning machine for online and offline one-class classification-An expanded toolbox.", Neurocomputing 261 (2017): 126-143.

**Paper5 (SVDD+):** Zhang, Wenbo. "Support vector data description using privileged information.", Electronics Letters 51.14 (2015): 1075-1076.
        
**Paper6 (OCSVM+):** Burnaev, Evgeny, and Dmitry Smolyakov. "One-class SVM with privileged information and its application to malware detection.", 2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW). IEEE, 2016.


**For any query, you can reach me at chandangautam31@gmail.com , phd1501101001@iiti.ac.in**
