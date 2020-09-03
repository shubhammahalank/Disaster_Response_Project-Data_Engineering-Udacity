import sys
import re
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

# Data Analysis
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Natural-Language Processing
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



def load_data(database_filepath):
    '''
    i/p :   database_filepath   :   Database absolute file path
            
    o/p :   X                   :   Training messages
            y                   :   Training targets
            category_names      :   Names of categories
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('FigureEight', engine)
    X = df.message.values
    y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, y, category_names

def tokenize(text):
    '''
    i/p :   text                :   Text input to be cleaned and tokenized
            
    o/p :   clean_tokens        :   Tokenized and processed list of tokens
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    i/p :   N/A
            
    o/p :   cv                  :   Model result
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    parameters = {
                'clf__estimator__estimator__C': [1,2,5],
                'tfidf__smooth_idf':[True,False]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    i/p :   model               :   Designed model to be evaluated
            X_test              :   Testing messages
            Y_test              :   Actual testing targets
            category_names      :   Names of categories
            
    o/p :   N/A
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))
    for i in range(Y_test.shape[1]):
        print('%25s Accuracy: %.3f' %(category_names[i], accuracy_score(Y_test[:,i], Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    i/p :   model               :   Designed model to be saved
            model_filepath      :   Absolute path of a model
            
    o/p :   N/A
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()