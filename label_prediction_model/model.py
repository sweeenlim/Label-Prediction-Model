import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../data/contract_dataset_v20220109.csv')
X = dataset['provision']
y=dataset['label']
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42,stratify=y)

# Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_data['provision'])
y_train = train_data['label']

X_test = vectorizer.transform(test_data['provision'])
y_true = test_data['label']

# Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.toarray())
X_test=scaler.transform(X_test.toarray())

# Handling Class imbalance
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
X_train_res, y_train_res=sm.fit_resample(X_train,y_train)


# Training the model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced',n_estimators=100, random_state=42, max_depth=5) 
rf_model.fit(X_train_res, y_train_res)



pickle.dump(rf_model, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
vec_file = 'vectorizer.pkl'
pickle.dump(vectorizer, open(vec_file, 'wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))
new = vectorizer.transform([train_data.iloc[0]['provision']]) #returns a generator 
new= scaler.transform(new.toarray())
print(model.predict(new))
