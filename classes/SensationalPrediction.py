import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle

class SensationalPrediction():

    def __init__(self):

        basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
        trainfile = basedir + 'input_data/processed/trainnews_sensational_processed.csv'
        testfile = basedir + 'input_data/processed/testnews_sensational_processed.csv'

        global dataTrain
        global accscore

        dataTrain = pd.read_csv(trainfile, sep=',')
        dataTest = pd.read_csv(testfile, sep=',')

        tfidfV = TfidfVectorizer(ngram_range = (1,2), sublinear_tf = True)

        self.nb_pipeline_ngram = Pipeline([
        ('vector', tfidfV),
        ('mname',MultinomialNB())])

        self.nb_pipeline_ngram.fit(dataTrain['name'], dataTrain['sensational_label'])
        predicted_nb_ngram = self.nb_pipeline_ngram.predict(dataTest['name'])
        accscore = metrics.accuracy_score(dataTest['sensational_label'], predicted_nb_ngram)
        print("Sensational Feature Prediction - accuracy:   %0.6f" % accscore)

    def predict(self, text):
        predicted = self.nb_pipeline_ngram.predict([text])
        predicedProb = self.nb_pipeline_ngram.predict_proba([text])[:,1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, text):
        predicedProb = self.nb_pipeline_ngram.predict_proba([text])[:,1]
        return float(predicedProb)

    def getScore(self):
        return accscore

if __name__ == "__main__":

    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    sensational_filename_pkl = basedir + 'models/sensational_feature_av4.pkl'
    print(sensational_filename_pkl)
    sp = SensationalPrediction()
    text1 = dataTrain['name'][0]
    text2 = dataTrain['name'][1]
    print(sp.predict(text1), text1)
    print(sp.predict(text2), text2)
    pickle.dump(sp, open(sensational_filename_pkl, 'wb'))
    del sp

