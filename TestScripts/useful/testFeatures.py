from datetime import datetime
import os.path as path
from typing import Tuple

import numpy as np
import pandas as pd

from gatherData import DataGatherer
from helper import Helper
from cleanData import DataCleaner
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

from DataSplitter import DataSplitter

# Not used
DIB_NAME = "DIB1"
DATASETS_PATH = path.join(DIB_NAME, 'dataRaw', 'ID')
SELECTED_DATA_SET_PATH = path.join("Single", "ET")
CONCAT_FILE_PATH = DIB_NAME
CONCAT_FILE_NAME = "concat-data"
CONCAT_FILE_EXT = ".csv"
CONCAT_FULLPATH_WITHOUT_EXT = path.join(CONCAT_FILE_PATH, CONCAT_FILE_NAME)

class FeatureTester:
    def main(self):
        print("Getting clean data..")
        data: pd.DataFrame = DataCleaner().getCleanData(10)
        print("Quick stats on clean data")
        Helper.quickDfStat(data)
        print("Getting Data Sets..")        
        features, answers = DataSplitter().getAllFeaturesAndAnswers(data)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = DataSplitter().getTrainValTestSplit(data)
        print("Quick stats on features and answers for the train-val-test split")
        Helper.quickDfArrStat([X_train, Y_train])
        Helper.quickDfArrStat([X_val, Y_val])
        Helper.quickDfArrStat([X_test, Y_test])
        
        print("Verifying the features and answers for the sets add up")
        print("Verifying X..")
        featureArr = []
        for df in [X_train, X_val, X_test]:
            val = round(len(df.index)/len(features.index), 3)
            featureArr.append(val)
            print(f"{val}")
        
        print("Verifying Y..")
        answerArr = []
        for df in [Y_train, Y_val, Y_test]:
            val = round(len(df.index)/len(answers.index), 3)
            answerArr.append(val)
            print(f"{val}")
            
        print("Adding up X")
        sum = 0
        for x in featureArr:
            sum += x
        print(f"Sum: {sum}")    
        print("Adding up Y")
        sum = 0
        for y in answerArr:
            sum += y
        print(f"Sum: {sum}")   
        
        # # Calculating number of invalid rows -> [removed now]
        # print("Number of invalid rows FPOGV:")
        # print(data.groupby(by='FPOGV').count())
        # print("Number of invalid rows LPV:")
        # print(data.groupby(by='LPV').count())
        # print("Number of invalid rows RPV:")
        # print(data.groupby(by='RPV').count())
        # print("Number of invalid rows BPOGV:")
        # print(data.groupby(by='BPOGV').count())
        
        print("Calculating classification numbers")
        print(data.groupby(by='classifier').count())
        
        print("Building Model..")
        model = self.buildModel(X_train, Y_train)
        
        # get importance
        importance = model.feature_importances_
        
        # Summarize feature importance
        for f,s in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (f,s))

        # plot feature importance
        fig, ax = plt.subplots(figsize=(30, 10))
        ax.bar([x for x in range(len(importance))], importance)
        print(features.columns.values)
        ax.set_xticks([x for x in range(len(importance))])
        ax.set_xticklabels(labels=features.columns.values)
        #plt.show()
        plt.savefig(path.join("Figures", "feature-test-"+ datetime.now().strftime(R"%m-%d-%Y, %H-%M-%S") + ".png"), format='png', dpi=200)
        
                
    def buildModel(self, features:pd.DataFrame, answers:pd.DataFrame):
        # from tutorial: https://machinelearningmastery.com/calculate-feature-importance-with-python/
        
        # define the model
        model = RandomForestClassifier()
        
        # fit the model
        model.fit(features, answers)
        
        return model

if __name__ == "__main__":
    FeatureTester().main()