
######## Import Libraries ##########################
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

######## Data Reading ############################
scf=pd.read_excel('SCFP2016.xlsx')


scf.drop( scf[scf['DEBT'] == 0 ].index , inplace=True)
scf=scf.reset_index()             #Resetting the index
scf1=scf.drop('index',axis=1)     #Removing the 'index' column

a=scf1.describe().transpose()

X=scf1.drop('LATE',axis=1)
Y=scf1['LATE']

feat_labels=list(X.columns)


######### RF feature selection ####################################3
# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100,random_state=0)

# Train the classifier
clf.fit(X_train, y_train)


# Print the name and gini importance of each feature
feature=[]
for i in zip(feat_labels, clf.feature_importances_):
    feature.append(i)
    
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.01
scores = clf.feature_importances_
sfm = SelectFromModel(clf, threshold=0.01)

# Train the selector
sfm.fit(X_train, y_train)
# Print the names of the most important features
main_features=[]
for j in sfm.get_support(indices=True):
    main_features.append(feat_labels[j])
    

top_13_feat=list(main_features)
top_13=scf1[top_13_feat]

####Variance Inflation Factor ################################
from statsmodels.stats.outliers_influence import variance_inflation_factor
# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(top_13.values, i) for i in range(top_13.shape[1])]
vif["features"] = top_13.columns
index = vif[vif["VIF Factor"]>5].index.tolist()
print(index)
vif.round(1)

###############################################################

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(top_13)
rescaledX = scaler.transform(top_13)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rescaledX, Y, test_size=0.2, random_state=30)

######################## Decision Tree Classiier ###########################
from sklearn.tree import DecisionTreeClassifier
dec_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dec_classifier.fit(X_train, y_train)

#############################################################################

######################## Random Forest Classifier #########################
from sklearn.ensemble import RandomForestClassifier
rand_classifier=RandomForestClassifier(n_estimators=100,random_state=15,n_jobs=-1)
rand_classifier.fit(X_train, y_train)
###########################################################################

######################## Xg Boost Classifier ###############################
from xgboost import XGBClassifier
xg_classifier = XGBClassifier(n_estimators=100, max_depth=6, silent=False)
xg_classifier.fit(X_train, y_train)

###########################################################################

##################### Logistic Regression ##################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

############################################################################

########## Predicting the Train set results #################################
y_pred_dec_train = dec_classifier.predict(X_train)
y_pred_rand_train=rand_classifier.predict(X_train)
y_pred_xg_train=xg_classifier.predict(X_train)
y_pred_logreg_train=logreg.predict(X_train)

############################################################################

########## Predicting the Test set results #################################
y_pred_dec = dec_classifier.predict(X_test)
y_pred_rand=rand_classifier.predict(X_test)
y_pred_xg=xg_classifier.predict(X_test)
y_pred_logreg=logreg.predict(X_test)

############################################################################

################ Making the Confusion Matrix ###############################
from sklearn.metrics import confusion_matrix
dec_cm = confusion_matrix(y_test, y_pred_dec)
rand_cm = confusion_matrix(y_test, y_pred_rand)
xg_cm = confusion_matrix(y_test, y_pred_xg)
logreg_cm = confusion_matrix(y_test, y_pred_logreg)

############################################################################

############## Train Accuracy (tp + tn) / (p + n)#################################
from sklearn.metrics import accuracy_score
dec_train_accuracy=accuracy_score(y_train,y_pred_dec_train)
rand_train_accuracy=accuracy_score(y_train,y_pred_rand_train)
xg_train_accuracy=accuracy_score(y_train,y_pred_xg_train)
logreg_train_accuracy=accuracy_score(y_train,y_pred_logreg_train)

train_accuracy=[dec_train_accuracy,rand_train_accuracy,xg_train_accuracy,logreg_train_accuracy]
a1=pd.DataFrame(train_accuracy)
############################################################################

############## Accuracy (tp + tn) / (p + n)#################################
dec_accuracy=accuracy_score(y_test,y_pred_dec)
rand_accuracy=accuracy_score(y_test,y_pred_rand)
xg_accuracy=accuracy_score(y_test,y_pred_xg)
logreg_accuracy=accuracy_score(y_test,y_pred_logreg)

accuracy=[dec_accuracy,rand_accuracy,xg_accuracy,logreg_accuracy]
a2=pd.DataFrame(accuracy)
############################################################################


############### Precision  tp / (tp + fp)###################################
from sklearn.metrics import precision_score
dec_precision = precision_score(y_test, y_pred_dec)
rand_precision = precision_score(y_test, y_pred_rand)
xg_precision = precision_score(y_test, y_pred_xg)
logreg_precision = precision_score(y_test, y_pred_logreg)

precision=[dec_precision,rand_precision,xg_precision,logreg_precision]
a3=pd.DataFrame(precision)
############################################################################

################ Recall: tp / (tp + fn) ###################################
from sklearn.metrics import recall_score
dec_recall = recall_score(y_test,y_pred_dec)
rand_recall = recall_score(y_test,y_pred_rand)
xg_recall = recall_score(y_test,y_pred_xg)
logreg_recall = recall_score(y_test,y_pred_logreg)

recall=[dec_recall,rand_recall,xg_recall,logreg_recall]
a4=pd.DataFrame(recall)
############################################################################

################# F1: 2 *(precision*recall)/ (precision+recall) ############
from sklearn.metrics import f1_score
dec_f1 = f1_score(y_test,y_pred_dec)
rand_f1 = f1_score(y_test,y_pred_rand)
xg_f1 = f1_score(y_test,y_pred_xg)
logreg_f1 = f1_score(y_test,y_pred_logreg)

f1_score=[dec_f1,rand_f1,xg_f1,logreg_f1]
a5=pd.DataFrame(f1_score)
#############################################################################

################# ROC AUC####################################################
from sklearn.metrics import roc_auc_score
dec_roc_auc = roc_auc_score(y_test,y_pred_dec)
rand_roc_auc = roc_auc_score(y_test,y_pred_rand)
xg_roc_auc = roc_auc_score(y_test,y_pred_xg)
logreg_roc_auc = roc_auc_score(y_test,y_pred_logreg)

roc_auc=[dec_roc_auc,rand_roc_auc,xg_roc_auc,logreg_roc_auc]
a6=pd.DataFrame(roc_auc)
############################################################################
#############################################################################
model=['Decision Tree','Random Forest','XG Boost','Logistic Regression']
aa=pd.DataFrame(model)
Result=pd.concat([aa,a1,a2,a3,a4,a5,a6],axis=1)
Result.columns=['MODEL','Train_Accuracy','Test_Accuracy','Precision','Recall','F1_Score','ROC_AUC']
#Result.to_csv("RF_13_Feature_Selection2.csv",index=False)
