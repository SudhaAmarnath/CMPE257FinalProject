# CMPE257FinalProject-NELA-GT-2018

## Google drive project colab location
https://drive.google.com/drive/u/3/folders/1UI2BgfNbLYIYp1qwSAJ5aSNCCrvgrdAr

## Datasets
NELA-GT-2018 data set - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ULHLCB

## News Coverage
https://www.kaggle.com/aashita/nyt-comments
https://www.kaggle.com/astoeckl/newsen
https://www.kaggle.com/rmisra/news-category-dataset
https://www.kaggle.com/dbs800/global-news-dataset

## Sensational Feature
 https://www.thepersuasionrevolution.com/380-high-emotion-persuasive-words/
 

## NELA-GT-2018: A Large Multi-Labelled News Dataset for the Study of Misinformation in News Articles". (2019-01-15)
Name: Sudha Amarnath  
Student ID: 013709956  
Business Problem / Data narrative  
News, Fake News, Misinformation Classification Selected on 194 sources in the NELA-GT-2018 dataset. A number of organizations and platforms have developed methods for assessing reliability and bias of news sources. These organizations come from both the research community and from practitioner communities. While each of these organizations and platforms provide useful assessments on their own, each uses different criteria and methods to make their assessments, and most of these assessments cover relatively few sources. Thus, in order to create a large, centralized set of veracity labels, the collected ground truth (GT) data from eight different sites, which all attempt to assess the reliability and/or the bias of news. These assessment sites are:

NewsGuard 
Pew Research Center 
Wikipedia 
OpenSources 
Media Bias/Fact Check (MBFC) 
AllSides 
BuzzFeed News 
Politifact 
Based on the labels\rating provided these provides on differnet sources, fakeness prediction can be made by on the given NELA-GT-2018 using NewsCoverage and Sensationalism features. Results can also be obtained in a modular way by creating import packages of the features classes and also by loading the automatically created PKL file while running the python scripts.  

## Data Collection
NELA-GT-2018 data set - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ULHLCB

## Dataset Articles
The articles gathered in this dataset is found in an sqlite database. The database has one table name articles. This table has 4 textual columns:

date: Date of article in yyyy-mm-dd format.
source: Source of article.
name: Title of article.
content: Clean text content of article.
The rows of the article are sorted first with respect to date and then with respect to source.

The dataset's articles are also provided in plain-text files, with file-structure and file naming convension:

date/
    source/
        <source>--<date>--<title>.txt
Dataset Labels
The labels of sources are stored in a comma-seperated file labels.csv and in a human-readable format in labels.txt. Each row in the files contain information about a source. The column names use the naming convention <site_name>,<label_name>, where <site_name> is the name of the site providing the label and <label_name> is the name of the particular label. The following lists all columns in the labels files. The columns use different value, which is described below. Note that all columns can also have missing value (no data for that particular source).

## Feature 1 - News Coverage
The main idea is to find the integerity of the NELA-GT-2018 dataset topics against a source which could be the actual media like News Papers. There are high chances for the positive corelation when the comparision is done with the more reliable source like the News Channels. For this task, I am considering the News Coverage Datasets from Kaggle.
https://www.kaggle.com/aashita/nyt-comments
https://www.kaggle.com/astoeckl/newsen
https://www.kaggle.com/rmisra/news-category-dataset
https://www.kaggle.com/dbs800/global-news-dataset
The NELA-GT-2018 dataset topics span over a year , that are from 2018. Since the above data setup after preprocessing is similar for the coverage in year wise(2018), there could he high chances of co-relation. We then use this feature to perform fakeness classification for the NELA-GT-2018 Data set. Then I used TFIDF Vectorizer and Random Forest algorithm. The Accuracy for this model achieved 55%.

## Different approaches to classify text based on the news coverage information. The different approaches are as below
CountVectorizer  
Doc2Vec Model  
TF-IDF Vectorizer  

## The Performance of these approaches are evaluated based on the accuracy score using the following algorithms.
Multinomial Naive Bayes  
SVM  
SGD  
Random Forest  
Logistic Regression  

## Data Preprocessing  
Remove non-letters/Special Characters and Punctuations  
Convert to lower case  
Remove punctuation  
Tokenize  
Remove stop words  
Lemmentize  
Stemming  
Remove small words of length < 3  

## What didn't work?
Since the number of rows(700K) are higher in NELA-GT-2018, the time it takes to process the dataset cleaning and running algorithms is considerably in many hours. The distillation of this dataset took around 86 minutes. Most of The News Coverage dataset were not completely available for the year 2018 in a single dataset.

## What worked later?
I am shuffling the original NELA-GT-2018 dataset with a random_state=1000. This would make sure everytime the notebook is run, same shuffeling is retained. Out of 700k+ rows, I'll be selecting around 15k rows for the project. The cleaning, doc2vec training of the of the dataset was around 45 minutes. Next I merged 4 datasets that had monthly news information from reputed news sources for the year 2018. This has nearly 40k rows.

## Feature 2 - Sensational Feature Prediction
With the close look of the words, and when some of them are combined selectively together, there are cues which would lead to emotions in the way the speaker has said in a certain context. Words when used correctly can transform an “eh whatever” into “wow that’s it!”. Words can make you go from literally ROFL to fuming with fury to an uncontrollable-urge-to-take-action-NOW-or-the-earth-may-stop-swinging -on-its-axis. Highly emotional words are capable capable of transforming an absolute no into almost yes and a “perhaps” into “for sure”! Words that are used:

When you are trying to sell people a solution
When you are trying to get them to take an action (like, share, subscribe, buy)
When you are trying to get people to click and read your article
When you are trying to get someone to agree with you
I am using a dataset from high emotion persiasive words [ https://www.thepersuasionrevolution.com/380-high-emotion-persuasive-words/ ] where there are 1400+ words that are both positive and negative emotions that will help to predict the sensational score for an article. The data enrichment is done using SentiNet library which provides polarity associated with 50,000 natural language concepts. A polarity is a floating number between -1 and +1. Minus one is extreme negativity, and plus one is extreme positivity. The knowledge base is free. It can be downloaded as XML file. SenticNet 5 reaches 100,000 commonsense concepts by employing recurrent neural networks to infer primitives by lexical substitution.

## Method used 
By performing cosine similarity for each news in the NELA-GT-2018 Data set with the Sensational words results in a particular score for each topic. These topics are then given a sensational label based on the 50% sensataional score. For the score above 50% value, the sensational label is predicted as 1 otherwise its 0. Then I used TFIDF Vectorizer and Multinomial Naive Bayes algorithm. The Accuracy for this model achieved to 60%.

## Feature 3 - NelaEncodedLabels

NewsGuard : Among the following, NewGuard overall_class is itself an encoded_label for the sources. A New column 'newsguard_label' is amalgamated based on the 0/1 values of overall_class
Does not repeatedly publish false content  
Gathers and presents information responsibly  
Regularly corrects or clarifies errors  
Handles the difference between news and opinion responsibly  
Avoids deceptive headlines  
Website discloses ownership and financing  
Clearly labels advertising  
Reveals who's in charge, including any possible conflicts of interest  
Provides information about content creators  
score  
overall_class  

Pew Research Center : Among the following, Pew Research Center inference columns known_by_40% has a binary value based on the poplarity of the source. A New column 'pewresearch_label' is amalagamated based on the 0/1 values of known_by_40%
known_by_40%  
total  
consistently_liberal  
mostly_liberal', 'Pew Research Center, mixed  
mostly conservative  
consistently conservative',  

Wikipedia : Label wikipedia_label is created for 0/1 value if fake its set to 0  
is_fake  
Open Sources: Among the following, Open Sources inference columns bias has a 1, 2, 3 score based on the bias of the source. A New column 'opensourcebias_label' is amalagamated based on the bias values 1-3  
reliable  
fake  
unreliable  
bias  
conspiracy  
hate  
junksci  
rumor  
blog  
clickbait  
political  
satire  
state  

Media Bias: Media Bias inference columns label has a specific facts on the source. A New column 'mediabias_label' is amalagamated based on the bias factors [ 'conspiracy_pseudoscience', 'left_center_bias', 'left_bias', 'right_bias', 'questionable_source', 'right_center_bias', 'least_biased', 'satire' ]  
label  
factual_reporting  
extreme_left  
right  
extreme_right  
propaganda  
fake_news  
some_fake_news  
failed_fact_checks  
conspiracy  
pseudoscience  
hate_group  
anti_islam  
nationalism  

Allsides: Among the following, Allsides inference columns community_label has a factors based on the public agreement for the source. A New column 'allsides_label' is amalagamated based on the values [ 'somewhat agree', 'somewhat disagree', 'strongly agree', 'agree', 'strongly disagree', 'absolutely agree', 'disagree' ]  
bias_rating  
community_agree  
community_disagree  
community_label  

BuzzFeed: Only one column based on left/right leaning for the source and a new label buzzfeed_label is encoded with binary values  
leaning  

PolitiFact: A new label politificat_label is encoded based on the true/ false counts of these columns for a source.  
Pants on Fire!  
False  
Mostly False  
Half-True  
Mostly True  
True  

For the rows having NaN values, it is retained as it is and not given any inference yet.

## Modular Approach
Modular approach is being considered now for the team in a centralized directory. Sensational Feature is integrated in assignment 2 Separate functions have been included for the features NewsCoverage() Class is defined based on TFIDF Vectorizer and Multinomial Naive Bayes algorithm to easily predict the fakeness. SensationalPrediction() Class is defined using TFIDF Vectorizer and Multinomial Naive Bayes algorithm to easily predict the fakeness. NelaEncodeLabelPrediction() Class is defined using TFIDF Vectorizer and Multinomial Naive Bayes algorithm to easily predict the fakeness.

Redefined the NewsCoverage() and SensationalPrediction() classes. Changed the algorithm for NewsCoverage Prediction to use the top document match from doc2vector output. For the NewsCoverage() Class Object pickle file is created at ../models/newscoverage_feature.pkl For the SensationalPrediction() Class Object pickle file is created at ../models/sensational_feature.pkl NelaEncodeLabelPrediction() Class Object pickle file is created at ../models/sensational_feature.pkl All the data sets and Models are located in AlternusVeraDataSets2019/FinalExam/Spartans/Sudha/input_data The Models are located in AlternusVeraDataSets2019/FinalExam/Spartans/Sudha/models Pickle load the NewsCoverage() Class Object and test the train_news for prediction Pickle load the SensationalPrediction() Class Object and test the train_news prediction Pickle load the NelaEncodedLabelPrediction() Class Object and test the train_news prediction

New python files are created in directory ./classes. init.py file defined for class imports NewsCoverage.py is defined for News Coverage Feature SensationalPrediction.py is defined for Sensational Prediction Feature Pickle models are saved when the script is run (guarded in main) Import class packages Instantiate class object Verify train set clean Prediction Probabilites are defined in the respective classes Checking Prediction score Calcualtion of polynomial eqation for the 3 features Performance Analysis.

## Machine Learning Life-cycle
### 1. Configuration of the System : Iterative, Notebook, code structure, data, where will it reside, folders, cloud buckets etc.
### 2. Data Collection : initial Data Set
### 3. Set Data Narrative : Set Business Objectives, what use case are you solving for
### 4. Exploratory Data Analysis and Visualization

1. feature analysis and engineering (for ML, for DL it's feature extraction)
2. Analyze data
3. Visualize data
4. Run Stats: mean, median, mode, correlation, variance
5. .cor
6. pairplot()
7. gini score
8. feature_importance with xgboost

### 5. Data Prep: Curation

1. Feature Selection and Extraction : what are the main features to use in this data set?
2. Data Verification: Do we have enough data?
3. Possibility of Amalgamation1: Add Dataset 2
4. Data Cleansing
5. Data Regularization
6. Data Normalization

### 6. Unsupervised Exploration : Find relevant Clusters in Your Data

1. How many clusters? Explore different k’s…
2. Select Clustering algorithms, run several and compare in a table
3. What does each cluster mean? How do they contribute to your Data Narrative (Story)
4. Measure goodness of your clusters (e.g., BICs)

### 7. Supervised Training Preparation: Data Curation : label your data set

1. Classify Your Data Sets : Run different classification algorithms
2. Measure Classification Success
3. What regression objectives should we have? Complete your , add to your Data Story
4. Run Regressions using various algorithmsv5. Measure Success of Regressions and
6. Compare Regressions in a table

### 8. Metrics and Evaluation

1. F1, R2, RMSE,
2. Precision, Recall, Accuracy
3. Confusion Matrix
4. Other metric as applicable to your project

### 9. Distillation

1.Entity Identification  
2.Customer Rank  
3.Sentiment  
4.Topic Modeling  

## Conclusion
### Amalgamted NELA dataset consists of Doc2Vec inferred vector values of the NewsCoverage dataset, SensationalScores and NelaEncodedLabels
TFIDF Multinomial Naive Bayes Algorithm was selected for the 3 features-  
NelaEncodedLabel accuracy: 58%  
NewsCoverageFeature - accuracy: 48%  
Sensational Feature - accuracy: 53%  

### Performance analysis of the valid news for Nela encoded label:
truePos= 1845  
trueNeg= 640  
falsePos= 68  
falseNeg= 514  
ignored= 917  
accuracy= 78%  
Polynomial score of the 3 fearures came up to 63%  

### For a modular approach 3 classes were created:
NewsCoverage.py  
SensationalPrediction.py  
NelaEncodedLabelPrediction.py  

### Import packages were created for all these 3 classes.
Guard function was defined in the classes to create an instance to save the pkl files.  
Import of the classes was done to directly to retun accuracy and predicted probability and input name. Pkl files were created when the python scripts were run without the imports.
