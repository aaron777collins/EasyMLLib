from cgi import test
from datetime import datetime
import os.path as path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF

from DataGatherer import DataGatherer
from helper import Helper
from DataCleaner import DataCleaner
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from matplotlib import pyplot as plt

from DataSplitter import DataSplitter
from logger import Logger

DIB_NAME = "DIB1"
DATASETS_PATH = path.join(DIB_NAME, 'dataRaw', 'ID')
SELECTED_DATA_SET_PATH = path.join("Single", "ET")
CONCAT_FILE_PATH = DIB_NAME
CONCAT_FILE_NAME = "concat-data"
CONCAT_FILE_EXT = ".csv"
CONCAT_FULLPATH_WITHOUT_EXT = path.join(CONCAT_FILE_PATH, CONCAT_FILE_NAME)

class ModelTester:
    def main(self):
        
        print("Running models on each participant (1-28)")
        for id in range(1,29):
            
            
            print(f"Creating Logger for participant {id}")
            self.logger = Logger(f"Test-Models-Scores-{id}.txt")
            
            self.logger.log("Getting clean data..")
            data: pd.DataFrame = DataCleaner().getCleanData(id)
            self.logger.log("Quick stats on clean data")
            Helper.quickDfStat(data)
            
            self.logger.log("Getting Data Sets..")        
            startTime = datetime.now()
            features, answers = DataSplitter().getAllFeaturesAndAnswers(data)            
            X_train, Y_train, X_val, Y_val, X_test, Y_test = DataSplitter().getTrainValTestSplit(data)
            self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {datetime.now()-startTime}")

            
            self.logger.log("Quick stats on features and answers for the train-val-test split")
            Helper.quickDfArrStat([X_train, Y_train])
            Helper.quickDfArrStat([X_val, Y_val])
            Helper.quickDfArrStat([X_test, Y_test])
            
            self.logger.log("Verifying the features and answers for the sets add up")
            # self.logger.log("Verifying X..")
            featureArr = []
            for df in [X_train, X_val, X_test]:
                val = round(len(df.index)/len(features.index), 3)
                featureArr.append(val)
                # self.logger.log(f"{val}")
            
            # self.logger.log("Verifying Y..")
            answerArr = []
            for df in [Y_train, Y_val, Y_test]:
                val = round(len(df.index)/len(answers.index), 3)
                answerArr.append(val)
                # self.logger.log(f"{val}")
                
            self.logger.log("Adding up X")
            sum = 0
            for x in featureArr:
                sum += x
            self.logger.log(f"Sum: {sum}")    
            self.logger.log("Adding up Y")
            sum = 0
            for y in answerArr:
                sum += y
            self.logger.log(f"Sum: {sum}")   
            
            classifierNames = [
                # "Nearest Neighbors (3)",
                # "Linear SVM (kernel='linear', C=0.025)",
                # "RBF SVM (gamma=2, C=1)",
                # "Gaussian Process (1.0 * RBF(1.0)",
                "Decision Tree (max_depth=5)",
                "Random Forest (max_depth=5, n_estimators=10, max_features=1)",
                "Neural Net (alpha=1, max_iter=1000)",
                "AdaBoost",
                "Naive Bayes",
                "QDA",
                "Nearest Neighbors",
                "Linear SVM",
                "RBF SVM",
                # "Gaussian Process",
                "Decision Tree",
                "Random Forest",
                "Neural Net",
            ]

            
            classifiers = [
                # KNeighborsClassifier(3),
                # SVC(kernel="linear", C=0.025),
                # SVC(gamma=2, C=1),
                # GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis(),
                KNeighborsClassifier(),
                SVC(),
                SVC(),
                # GaussianProcessClassifier(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                MLPClassifier(),
                # KNeighborsClassifier(3),
                # SVC(kernel="linear", C=0.025),
                # SVC(gamma=2, C=1),
                # GaussianProcessClassifier(1.0 * RBF(1.0)),
                # DecisionTreeClassifier(max_depth=5),
                # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                # MLPClassifier(alpha=1, max_iter=1000),
                # AdaBoostClassifier(),
                # GaussianNB(),
                # QuadraticDiscriminantAnalysis(),
            ]
            
            self.logger.log("Building many models from list the list of classifiers: ", classifierNames)
            
            for i, classifier in enumerate(classifiers):
                self.logger.log(f"Building Model on: {classifierNames[i]}")
                
                startTime = datetime.now()
                model = self.buildModel(X_train, Y_train, classifier)
                self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {datetime.now()-startTime}")

                self.logger.log(f"Possible tests:", metrics.SCORERS.keys())        
                self.logger.log("Testing model on Train")
                for test_type in ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']:
                    startTime = datetime.now()
                    self.logger.log(f"Testing {test_type}", cross_val_score(model, X_train, Y_train, cv=5, scoring=test_type))
                    self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {datetime.now()-startTime}")
                    
                self.logger.log("Testing model on Val")
                for test_type in ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']:
                    startTime = datetime.now()
                    self.logger.log(f"Testing {test_type}", cross_val_score(model, X_val, Y_val, cv=5, scoring=test_type))
                    self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {datetime.now()-startTime}")
                    
                self.logger.log("Testing model on Test")
                for test_type in ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']:
                    startTime = datetime.now()
                    self.logger.log(f"Testing {test_type}", cross_val_score(model, X_test, Y_test, cv=5, scoring=test_type))
                    self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {datetime.now()-startTime}")
                
            # FEATURE TESTING CODE BELOW
            
                # get importance
                # importance = model.feature_importances_
                
                # # Summarize feature importance
                # for f,s in enumerate(importance):
                #     print('Feature: %0d, Score: %.5f' % (f,s))

                # # plot feature importance
                # fig, ax = plt.subplots(figsize=(30, 10))
                # ax.bar([x for x in range(len(importance))], importance)
                # print(features.columns.values)
                # ax.set_xticks([x for x in range(len(importance))])
                # ax.set_xticklabels(labels=features.columns.values)
                # #plt.show()
                # plt.savefig(path.join("Figures", "feature-test-"+ classifierNames[i] +"-"+ datetime.now().strftime(R"%m-%d-%Y, %H-%M-%S") + ".png"), format='png', dpi=200)
        
                
    def buildModel(self, features:pd.DataFrame, answers:pd.DataFrame, model):
        # from tutorial: https://machinelearningmastery.com/calculate-feature-importance-with-python/
        
        # fit the model
        model.fit(features, answers)
        
        return model

if __name__ == "__main__":
    ModelTester().main()