import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier


class NewsCoverage():

    def __init__(self):        

        basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
        trainfile = basedir + 'input_data/processed/trainnews_d2v_processed.csv'
        testfile = basedir + 'input_data/processed/testnews_d2v_processed.csv'

        global dataTrain
        global accscore
        dataTrain = pd.read_csv(trainfile, sep=',')
        dataTest = pd.read_csv(testfile, sep=',')

        tfidfV = TfidfVectorizer(ngram_range = (1,2), sublinear_tf = True)

        '''
        self.random_forest_ngram = Pipeline([
            ('vector', tfidfV),
            ('mname', RandomForestClassifier(n_estimators=100, n_jobs=3))
        ])

        self.random_forest_ngram.fit(dataTrain['clean'].values.astype('U'), dataTrain['doc2vecsimilarity'])
        predicted_rf_ngram = self.random_forest_ngram.predict(dataTest['clean'])
        accscore = metrics.accuracy_score(dataTest['doc2vecsimilarity'], predicted_rf_ngram)
        print("News Coverage Feature Prediction - accuracy:   %0.6f" % accscore)
        '''

        self.nb_pipeline_ngram = Pipeline([
        ('vector', tfidfV),
        ('mname',MultinomialNB())])

        self.nb_pipeline_ngram.fit(dataTrain['name'], dataTrain['doc2vecsimilarity'])
        predicted_nb_ngram = self.nb_pipeline_ngram.predict(dataTest['name'])
        accscore = metrics.accuracy_score(dataTest['doc2vecsimilarity'], predicted_nb_ngram)
        print("News Coverage Feature Prediction - accuracy:   %0.6f" % accscore)



    def predict(self, text):
        idx = dataTrain.index[dataTrain['name'] == text].tolist()[0]
        cleantxt = dataTrain['clean'][idx]
        predicted = self.nb_pipeline_ngram.predict([cleantxt])
        predicedProb = self.nb_pipeline_ngram.predict_proba([cleantxt])[:,1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, text):
        idx = dataTrain.index[dataTrain['name'] == text].tolist()[0]
        cleantxt = dataTrain['clean'][idx]
        predicedProb = self.nb_pipeline_ngram.predict_proba([cleantxt])[:,1]
        return float(predicedProb)

    def getScore(self):
        return accscore

if __name__ == "__main__":
    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    newscoverage_filename_pkl = basedir + 'models/newscoverage_feature_av4.pkl'
    nc = NewsCoverage()
    text1 = dataTrain['name'][0]
    text2 = dataTrain['name'][1]
    print(nc.predict(text1), text1)
    print(nc.predict(text2), text2)
    pickle.dump(nc, open(newscoverage_filename_pkl, 'wb'))
    del nc
