# Predictive Modeling (19 questions)
### 1 (Given a Dataset) Analyze this dataset and give me a model that can predict this response variable.

- A typical process will begin with a data integrity check, as well as data exploration. After getting to know the data set, I would like to understand on program we are solving. Since we have a label, there are many supervised learning models out there, which one solves this problem the best would be something I consider. After decided on the model, things to consider is do we have enough data to prevent over fitting. Then we will do some hyper parameter search, and cross validate the model. 

### 2. What could be some issues if the distribution of the test data is significantly different than the distribution of the training data?

-over fitting 
- not enough data in general.  (problem with data, indeed they have different distribution) 
- Covariate Shift : https://blog.bigml.com/2014/01/03/simple-machine-learning-to-detect-covariate-shift/  To test it there is a phi coefficient(mean square contingency coefficient ) If the phi coefficient of the evaluation is smaller than 0.2 then you can say that your training and production data are indistinguishable and they come from the same or at least very similar distribution. P(y|x) are the same but P(x) are different. (covariate shift) (problem with data)
-concept shift: means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This causes problems because the predictions become less accurate as time passes. P(y|x) are different. (concept shift)


### 3. What are some ways I can make my model more robust to outliers?

- use a tree based model 
- measure MAE instead of MSE. 
- log transform the dataset. Winsorizing the extremes
- regularization reduce variance (increase bias) 

### 4. What are some differences you would expect in a model that minimizes squared error, versus a model that minimizes absolute error? In which cases would each error metric be appropriate?
- MSE give more leverage to outliers where MAE is less sensitive to outliers.
- In terms of gradient, the MSE is always differentiable but MAE is not at the at 0. 

### 5. what error metric would you use to evaluate how good a binary classifier is? What if the class are imbalanced? What if there are more than 2 groups?
- The metric to evaluate classifier would be accuracy, precision, recall, miss rate and F1 score, AUC. Depend on what’ most important, I would used different metric for the specific case (when do you use F1?)
- if the class is imbalanced, we could down sample, the common class, or up sampling the rare class. Or use none parametric method such as bootstrap.  Another way is we can add different weighting, for the class we care more about.  (log loss)
- if there are more than 2 groups, I would use one vs the rest, and run the same as two group. 


