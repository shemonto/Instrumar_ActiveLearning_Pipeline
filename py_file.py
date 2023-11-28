import random
import numpy as np
import pandas as pd
from active_learning import UncStrategy, QBCStrategy
# models
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

#from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
# generate some random data but control the seed to observe if the strategies
# are running properly with the seeds
data = pd.read_csv('/Users/shemontod/Desktop/Instrumar/aquafil_short/scaled_labeled_data.csv')

data.drop(columns=["TimeStamp_start","TimeStamp_end","ThreadlineId", "count"], inplace=True)

X = data.loc[:, data.columns != 'Label'] 
y = data['Label']

X_all = np.array(X)
y_all = np.array(y)


le = LabelEncoder()
# after this

""" Now let's start preparing to run the experiment several times
"""
# Prepare the variables to iterate over
#seeds = [1415,9265,3589,7932,3846,2643,3832,7950,2884,1971,6939,9375,1058,2097,4944,5923,781,6406,2862,899]
seeds = [1415,9265,3589,7932,3846,2643,3832,7950,2884,1971,6939,9375]

#models = [XGBClassifier(),ExtraTreesClassifier(),tree.DecisionTreeClassifier(),RandomForestClassifier(n_estimators=10)]
#models = [XGBClassifier(),ExtraTreesClassifier(),RandomForestClassifier(n_estimators=10),GradientBoostingClassifier(n_estimators=300)]
models = [XGBClassifier(), GradientBoostingClassifier(n_estimators=300)]
# 1450 hoitese train data for 15 min php data
total_budget, step_size =  6000,500          #4355, 335 ## 12 ta step ashbe

print('Aquafill ',' 13 Alarms', step_size)

# file to save experiments 7200,360
file_name = '21_Nov_AL_pipeline_result.csv'
# Run experiments for UncertaintySampling
with open(file_name, 'a') as f:    
    header = 'al_strategy,keep_balanced,model,seed,total_budget,step,f_score,accuracy\n'
    f.write(header)
    for s in seeds:
        al_strategies = [UncStrategy(seed=s, keep_balanced=False), UncStrategy(seed=s, keep_balanced=True), QBCStrategy(seed=s, keep_balanced=False), QBCStrategy(seed=s, keep_balanced=True)]
        for al_strategy in al_strategies:            
            for m in models:
                # separate the subset for testing
                X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state = s) 
                y_encoded = le.fit_transform(y)
                y_test_encoded = le.fit_transform(y_test)
                if m != models[0]:
                    #print('check ', step_size)
                    out = al_strategy.run_experiment(X, y, X_test, y_test, m, total_budget, step_size, s)
                else:
                    #print('check ', step_size)
                    out = al_strategy.run_experiment(X, y_encoded, X_test, y_test_encoded, m, total_budget, step_size, s)
    
                # dump values from out
    #             print(out)
                for line in out:
                    f.write(line)



print(' End Game !!! ')
