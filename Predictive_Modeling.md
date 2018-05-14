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

### 6. What are various ways to predict a binary response variable? Can you compare two of them and tell me when one would be more appropriate? What is the difference between these? (SVM, Logistic Regression, Naive Bayes, Decision Tree, etc.)
- SVM: a tradition way of classifier, have a solid equation to derive it. 
* Pro: I don’t know, many packages available. It can have a non linear decision boundary.
* Con: slow, decision can be driven by a boundary. 
- logistic regression: it is basically a linear regression plugged in a sigmoid function. 
* Pro: it is giving the probability directly. And can be used if you want a confidence of the classifier. 
* Con: accuracy is not the best, must be linear. Will perform bad if data is not linearly separable, don’t deal with categorical features well. 
- Naïve Bayes: one of the Go to method to get a quick result. Assumes independence of features. 
* pro: quick to run, will not over fit (always underfit.) train fast with large parameters.
* con: too simple. Sometimes assumption doesn’t hold
- Decision tree: make decision split based of certain criteria.
* pro: can parallelize

### 7. What is regularization and where might it be helpful? What is an example of using regularization in a model?
- regularization is used to prevent overfitting. The core concept is to implement a penalization term on the cost function. 
One example of using regularization is let’s say you have a lot of feature but not enough data, you can use Lasso to regularize and reduce some of the features. 
The two common way of regularization is Lasso and Ridge. (If you like combos Elastic net is a happy combination of the two.) Lasso suppress feature to 0 wile ridge only suppress the magnitude. 

### 8. Why might it be preferable to include fewer predictors over many?
- Too many predictors cause overfitting, where model earn the noise of training not the actual trend. And the training time increase as there are more parameters,
- make sense computationally, and easier to make business decision 
- curse of dimensionality, you will need more data to have a relevantly good feature. 

### 9. Given training data on tweets and their retweets, how would you predict the number of retweets of a given tweet after 7 days after only observing 2 days worth of data?
- There are two ways of build a machine learning model, first used the 2days worth of data as an input and 7 days as predictor and build a regression model. (assume we have 7 days of retweet historical data. 
-second way is to build a time series model and see how the number of retweets change with time. 

### 10. How could you collect and analyze data to use social media to predict the weather?
- First I want to identify possible feature we could extract and if there is any correlation between weather and people’s behavior. (for example if people talk about ice cream can indicate the weather is hot.) some example we want is: (topic, sentiment, direct indicator, etc.) 
- one we decide what feature we need, some place we can get data would be tweeter, Facebook, Instagram etc. 
- assuming weather have seasonality, I would look at historical weather as well as tweets. 

### 11. How would you construct a feed to show relevant content for a site that involves user interactions with items?
- I would write a personalized recommendation system. So the feed would base on user’s preference. We can create a item based similarity matrix, and based of the browsing history, recommend relevant item to user. Also we can do collaborative filtering using other user’s info. 
- cold start is an issue: there are several way to get around it. We could have the user fill out survey in the beginning, or we could use a generalize recommendation first and then get more personalized. 
- some of the modern tech such as wide&deep (facebook), or GBDT-CENT, and neural FM. https://arxiv.org/pdf/1707.07435.pdf, https://arxiv.org/abs/1708.05027

### 12. How would you design the people you may know feature on LinkedIn or Facebook?
- Before jump into the design, I would like to first define the business goal. People you may know paints a different picture in Linkedin comparing to Facebook. 
- For linkedin I would focus more of the professional network, I would take in features such as current title, company worked for, graduation years etc. and for facebook, I would look for location, common, friends. 
- Since this is a graph program, we can find simalrity between two nodes, and make recommendation based of similarity. I would like to say instead of people you may know, for linkedin is more like people you may want to know. 

### 13. How would you predict who someone may want to send a Snapchat or Gmail to?
- sender demographic, friend relation, past behavioral, feature we want would be relationship between the sender and receiver, we can make the receiver one vs all classification (random forest) 
- social media friend list, average Gmail usage per day, day of the week, (M-F paints a different feature compare to weekends). 
- use case, interoperability, 

### 14. How would you suggest to a franchise where to open a new store?
- assuming the business goal to a franchisees is to make more profit, since profit is revenue – cost, I would look at location information from both side of the equation. 
- On the cost side, I will look at the rental, equipment cost labor cost. I would get data from Zillow (housing price), average salary (labor cost)
- on the revenue side, I would further break down into price times quantity. I would do a demand forcasting model for around that area. And see how our competitor is doing. Weather we have any competition advantages. In terms of price, I would like to know what is our pricing power, and how much can we charge. 
- with all the above feature I would train a model to find the most optimal location ( maybe linear model)

### 15. In a search engine, given partial data on what the user has typed, how would you predict the user’s eventual search query?
-  I would do a sub string search of the historical search queries, perhaps taking account into time. I would weather the recent search more.
- I would get the NLP representation (tokenize tfidf) and see the frequency it appear with other search query and make a recommendation based of cosin3 similarity. 

### 16. Given a database of all previous alumni donations to your university, how would you predict which recent alumni are most likely to donate?
- I would collect features such as past donation frequency, time of the year (maybe before filing tax), current income (ability to donate), and engagement ( has the alumni come of events), social status (running for campaign?) 
- with all the data I would build a model (linear or tree based) to decide the likelihood of someone making a donation. 

### 17. You’re Uber and you want to design a heatmap to recommend to drivers where to wait for a passenger. How would you approach this?
- The goal of this problem is the increase the chance of getting a passenger in a given timeframe. (possession distribution) we would collect feature such as time of day, number of passengers per hour, number of drivers on spot, weather, day of week, event etc. 

### 18. How would you build a model to predict a March Madness bracket?
 - First I need to understand the basketball tournament, and then I will get the feature such as historical performance of a team, reputation, players, social media and build a predictive model. 

### 19. You want to run a regression to predict the probability of a flight delay, but there are flights with delays of up to 12 hours that are really messing up your model. How can you address this?
- this is a outlier problem, and the following three ways are suitable for this problem.
* choose a different model, tree based ones are not sensitive to outliers.
* log transform the data, so the outlier are not significant 
* chose a hard cutoff, so input the outliers with a resonable number. Mean + 2 std or quantile.  

