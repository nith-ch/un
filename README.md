# Introduction
This dataset is donated by the UNHCR to deal with complex household survey data. The variables collected at household level 
(including enumerator appreciation of vulnerability) and variables collected at personal levels.


## Install
```
pip install --upgrade category_encoders
pip install graphviz
```
## Importing the libraries
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import graphviz 
import category_encoders as ce
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,roc_auc_score,recall_score,f1_score,plot_confusion_matrix
%matplotlib inline
warnings.filterwarnings('ignore')
```

## Data Exploration
Loading the dataset
```
member = pd.read_csv('unhcr_rbac_pm_2020_member_v2.1_anonymised.csv', index_col=False)
```

## Select data
```
member2 =  member.iloc[0:57759,0:63]
```

## Age with pregnant
```
age_preg =  member2.iloc[0:57759,[2,6,7,8,9,11]]
age_preg.head()
age_preg.info()
```

## Sex with pregnant
```
sex_preg =  member2.iloc[0:57759,[1,8,9,11]]
sex_preg.head()
sex_preg.info()
```

## Count number of values
```
age_preg['vulnerabilities_pregnant'].value_counts()
age_preg['vulnerabilities_pregnant'].value_counts(normalize=True)
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/vul_preg.PNG)
```
age_preg['vulnerabilities_lactating'].value_counts()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/vul_lac.PNG)
```
age_preg['vulnerabilities_medical'].value_counts()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/vul_med.PNG)
```
age_preg['vulnerabilities_medical_threat'].value_counts()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/vul_med_thr.PNG)
```
age_preg['vulnerabilities_disability'].value_counts()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/vul_dis.PNG)

## Check age groups
```
age_preg['age'].value_counts().plot(kind="bar", rot=0, color='green', figsize=(8,5), title="Age Groups")
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/age_group.PNG)

## Check sex groups
```
sex_preg.sex.value_counts().plot(kind='bar', rot=0, color='red', title="Count sex")
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/sex_group.PNG)

## Check Amount of Age Groups
```
cnt_srs = age_preg['age'].value_counts().head(25)
plt.figure(figsize=(8,12))
sns.barplot(y=cnt_srs.index, x=cnt_srs.values)
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.title("Age Groups", fontsize=15)
plt.show()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/age_amount.png)

## Check Amount of Sex Groups
```
cnt_srs = sex_preg['sex'].value_counts().head(25)
plt.figure(figsize=(6,3))
sns.barplot(y=cnt_srs.index, x=cnt_srs.values)
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Sex', fontsize=12)
plt.title("Sex Groups", fontsize=15)
plt.show()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/sex_amount.png)

-----------------------------------

## Frequency distribution of values in variables_age
```
inputCols = ['vulnerabilities_pregnant','vulnerabilities_lactating',
             'vulnerabilities_medical','vulnerabilities_disability','age']

for col in inputCols:
    print(age_preg[col].value_counts())
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/frequency_dis.PNG)

## Explore age (target) variable
```
age_preg['age'].value_counts()
```

## Remove null from occu_preg
```
age_preg = occu_preg.dropna()
```

## Check missing values
```
age_preg.isnull().sum()
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/sum_agenull.PNG)

## Declare datasets
```
X = age_preg.drop(['age'], axis=1)
y = age_preg['age']
```

## Split data into separate training and test set
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
```
We will split data to train and test set as follows: Train set 70% and Test set 30%

## Check the shape of X_train and X_test
```
X_train.shape, X_test.shape
```
See the shape of train and test set

![alt text](https://github.com/nith-ch/un/blob/master/pic/shape_age.PNG)

## Check data types in X_train
```
X_train.dtypes
X_train.head()
```

## Encode variables with ordinal encoding
```
encoder = ce.OrdinalEncoder(cols=['vulnerabilities_pregnant','vulnerabilities_lactating','vulnerabilities_medical_threat',
                                  'vulnerabilities_medical','vulnerabilities_disability'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```

```
X_train.head()
X_test.head()
```

## DecisionTreeClassifier model with criterion gini index
```
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
```
## Fit the model
```
clf_gini.fit(X_train, y_train)
```

## Predict the Test set results with criterion gini index
```
y_pred_gini = clf_gini.predict(X_test)
```

## Check accuracy score
```
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
```
y_test are the true class labels and y_pred_gini are the predicted class labels in the test-set.
With GINI index, thr model accuracy score is 0.6259 

![alt text](https://github.com/nith-ch/un/blob/master/pic/acc_gini.PNG)
## Compare the train-set and test-set accuracy
```
y_pred_train_gini = clf_gini.predict(X_train)
y_pred_train_gini
```
```
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/training_acc_sco.PNG)

## Print the scores on training and test set
```
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
```

## Change series to array
```
y_train = y_train.values
```

## Visualize decision-trees
```
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train)) 
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/age_dec_tree.png)

```
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini,
               feature_names = X_train.columns, 
               class_names=y_train,
               filled = True,rounded=True);
```
![alt text](https://github.com/nith-ch/un/blob/master/pic/age_dec_tree2.png)

-----------------------------------

## Frequency distribution of values in variables_sex
```
inputCols = ['vulnerabilities_disability','vulnerabilities_medical',
             'vulnerabilities_medical_threat','sex']

for col in inputCols:
    print(sex_preg[col].value_counts())
```

## Explore sex (target) variable
```
sex_preg['sex'].value_counts()
```

## Remove null from occu_preg
```
sex_preg = occu_preg.dropna()
```

## Check missing values
```
sex_preg.isnull().sum()
```

## Declare datasets
```
X = sex_preg.drop(['sex'], axis=1)
y = sex_preg['sex']
```

## Split data into separate training and test set 
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
```
We will split data to train and test set as follows: Train set 70% and Test set 30%

## Check the shape of X_train and X_test
```
X_train.shape, X_test.shape
```
See the shape of train and test set

## Check data types in X_train
```
X_train.dtypes
X_train.head()
```

## DecisionTreeClassifier model with criterion entropy
```
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
```
#Fit the model
```
clf_en.fit(X_train, y_train)
```

## Predict the Test set results with criterion gini index
```
y_pred_gini = clf_gini.predict(X_test)
```

## Check accuracy score
```
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
```

## Compare the train-set and test-set accuracy
```
y_pred_train_gini = clf_gini.predict(X_train)
y_pred_train_gini
```

```
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
```

## Print the scores on training and test set
```
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
```

## Change series to array
```
y_train = y_train.values
```

## Visualize decision-trees
```
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train)) 
```

```
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini,
               feature_names = X_train.columns, 
               class_names=y_train,
               filled = True,rounded=True);
			   ```