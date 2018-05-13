# Insurance-Classification
Project @c https://www.kaggle.com/c/cs412-insurance

This is a multi-classification task where you are expected to predict the response variable.

Given the attributes or features of a client, the task of the predictive system is to measure the level of risk in providing insurance to the client. Risk is categorized into 8 levels.

There are two data files to use.
  * training.csv: This contains data about 20000 patients, which is used to train the data
  * testing.csv: This is a little version which is used to have a quick test about accuracy of our prediction model.
  The real file used to rank on kaggle remains unknown.
  
We have about two files.
  * The first one is using sklearn. 
    Considering this is a high dimensional model with about 80 different features, we first use SVM
    model like One Vs All and One VS ONE. It's not surprising that ovo is more accurate that ovr, but is still not accurate enough.
    Since I think it is a project that mainly to help us understand the difference between models, instead of tuning the parameters, 
    we pay more atttention to choosing with different training model. The final decision is that we use Logistic Regression as final model.
    Prediciotn accuray on Kaggle is about 50%
    
  * The second one is writing a model from scratch, which means we cannot use packages like sklearn anymore. 
    As advisor suggests, Naive Bayes model and Decision Tree may be the two most common choices. When we compare these two mdoels, Naive Bayes requires us to better have
    a good understanding of data initialize it. However, we don't have much background about medicine, thus we choose the Decision Tree. Also, we use bagging to improve
    the results. Because of limited choices of model, the accuracy we get is lower than using Logistic model. Accuracy on trianing is about 40% and is about 30% on Kaggle.
    
Final output:
  * out.csv: This is our prediction result. Result may be different every time you run the file.
 
