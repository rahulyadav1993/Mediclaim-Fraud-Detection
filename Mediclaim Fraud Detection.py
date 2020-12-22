#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings     # for supressing a warning when importing large files
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn.metrics import confusion_matrix,roc_curve,accuracy_score,roc_auc_score,classification_report
import pickle

from sklearn.model_selection import GridSearchCV,KFold

from pylab import rcParams


sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42


# In[2]:


# Load Train Dataset

Train=pd.read_csv("Train-1542865627584.csv")
Train_Beneficiarydata=pd.read_csv("Train_Beneficiarydata-1542865627584.csv")
Train_Inpatientdata=pd.read_csv("Train_Inpatientdata-1542865627584.csv")
Train_Outpatientdata=pd.read_csv("Train_Outpatientdata-1542865627584.csv")

# Load Test Dataset

Test=pd.read_csv("Test-1542969243754.csv")
Test_Beneficiarydata=pd.read_csv("Test_Beneficiarydata-1542969243754.csv")
Test_Inpatientdata=pd.read_csv("Test_Inpatientdata-1542969243754.csv")
Test_Outpatientdata=pd.read_csv("Test_Outpatientdata-1542969243754.csv")


# In[3]:


## Lets Check Shape of datasets 

print('Shape of Train data :',Train.shape)
print('Shape of Train_Beneficiarydata data :',Train_Beneficiarydata.shape)
print('Shape of Train_Inpatientdata data :',Train_Inpatientdata.shape)
print('Shape of Train_Outpatientdata data :',Train_Outpatientdata.shape)

print('Shape of Test data :',Test.shape)
print('Shape of Test_Beneficiarydata data :',Test_Beneficiarydata.shape)
print('Shape of Test_Inpatientdata data :',Test_Inpatientdata.shape)
print('Shape of Test_Outpatientdata data :',Test_Outpatientdata.shape)


# #### Train and Test Dataset understanding

# In[4]:


print('\033[1m'"Train Dataset"+ "\033[0m","\n",Train.head(4),'\n')

print('\033[1m'+"Test Dataset"+ "\033[0m")

print(Test.head(4)) # We don't have Target Variable Fraud in the test dataset and this target variable we need to predict


# In[5]:


#To Check the summary of the train dataset

Train.describe()


# In[6]:


## Lets check whether  providers details are unique or not in train data
print(Train.Provider.value_counts(sort=True,ascending=False).head(5))  # number of unique providers in train data.Check for duplicates

print('\n Total missing values in Train :',Train.isna().sum().sum())

print('\n Total missing values in Train :',Test.isna().sum().sum())


# ### Data Preprocessing on Beneficiary Dataset

# In[7]:


print('\033[1m'+"Train Dataset"+ "\033[0m")

#display(Train_Beneficiarydata.head(5))

print('\033[1m'+"Test Dataset"+ "\033[0m")

#display(Test_Beneficiarydata.head(5))


# In[8]:


#Lets Check missing values in each column in beneficiary data :


print('\033[1m'+"Train Beneficiary Dataset"+ "\033[0m")

print(Train_Beneficiarydata.isna().sum())

print('\033[1m'+"Test Beneficiary Dataset"+ "\033[0m")

print(Train_Beneficiarydata.isna().sum())


# In[9]:


# Lets check data types of each column in beneficiary data

Train_Beneficiarydata.dtypes


# In[10]:


Train_Beneficiarydata.describe(include='all')


# In[11]:


##Replacing 2 with 0 for chronic conditions ,that means chronic condition No is 0 and yes is 1

Train_Beneficiarydata = Train_Beneficiarydata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

Train_Beneficiarydata = Train_Beneficiarydata.replace({'RenalDiseaseIndicator': 'Y'}, 1)


## Same thing do in the Test Dataset also 
Test_Beneficiarydata = Test_Beneficiarydata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

Test_Beneficiarydata = Test_Beneficiarydata.replace({'RenalDiseaseIndicator': 'Y'}, 1)


# #### Feature Engineering on Beneficiary Dataset

# In[12]:


## Lets Create Age column to the Train and Test dataset

Train_Beneficiarydata['DOB'] = pd.to_datetime(Train_Beneficiarydata['DOB'] )
Train_Beneficiarydata['DOD'] = pd.to_datetime(Train_Beneficiarydata['DOD'],errors='ignore')
Train_Beneficiarydata['Age'] = round(((Train_Beneficiarydata['DOD'] - Train_Beneficiarydata['DOB']).dt.days)/365)


Test_Beneficiarydata['DOB'] = pd.to_datetime(Test_Beneficiarydata['DOB'])
Test_Beneficiarydata['DOD'] = pd.to_datetime(Test_Beneficiarydata['DOD'],errors='ignore')
Test_Beneficiarydata['Age'] = round(((Test_Beneficiarydata['DOD'] - Test_Beneficiarydata['DOB']).dt.days)/365)


# In[13]:


Train_Beneficiarydata.head(10)


# In[14]:


## As we can see above Age column have some Nan values, This is due to DOD is Nan for that record.
## As we see that last DOD value is 2017-12-01 ,which means Beneficiary Details data is of year 2017.
## so we will calculate age of other benficiaries for year 2017.

Train_Beneficiarydata.Age.fillna(round(((pd.to_datetime('2017-12-01' ) - Train_Beneficiarydata['DOB']).dt.days)/365),
                                 inplace=True)


Test_Beneficiarydata.Age.fillna(round(((pd.to_datetime('2017-12-01') - Test_Beneficiarydata['DOB']).dt.days)/365),
                                 inplace=True)


# In[15]:


Train_Beneficiarydata.head(5)


# #### Add Flag column 'WhetherDead' using DOD values to tell whether beneficiary is dead on not

# In[16]:


#Lets create a new variable 'WhetherDead' with flag 1 means Dead and 0 means not Dead

Train_Beneficiarydata.loc[Train_Beneficiarydata.DOD.isna(),'WhetherDead']=0
Train_Beneficiarydata.loc[Train_Beneficiarydata.DOD.notna(),'WhetherDead']=1



Test_Beneficiarydata.loc[Test_Beneficiarydata.DOD.isna(),'WhetherDead']=0
Test_Beneficiarydata.loc[Test_Beneficiarydata.DOD.notna(),'WhetherDead']=1


# In[17]:


print('\033[1m'+"Train Dataset"+ "\033[0m")

print(Train_Beneficiarydata.loc[:,'WhetherDead'].head(7))

print('\033[1m'+"Test Dataset"+ "\033[0m")

print(Train_Beneficiarydata.loc[:,'WhetherDead'].head(7))


# ### Data Preprocessing on Inpatient Dataset

# In[18]:


# Summary of Inpatient Dataset

print('\033[1m'+"Train Inpatient Dataset"+ "\033[0m")

#display(Train_Inpatientdata.head(5))

print('\033[1m'+"Test Inpatient Dataset"+ "\033[0m")

#display(Train_Inpatientdata.head(5))


# In[19]:


#Lets check missing values in each column in inpatient data

print('\033[1m'+"Train Inpatient Dataset"+ "\033[0m")

print(Train_Inpatientdata.isna().sum())

print('\033[1m'+"Test Inpatient Dataset"+ "\033[0m")

print(Test_Inpatientdata.isna().sum())


# In[20]:


Train_Inpatientdata.info()


# #### Feature Engineering on Inpatient Dataset

# Create new column 'AdmitForDays' indicating number of days patient was admitted in hospital

# In[21]:


## As patient can be admitted for only for 1 day,we will add 1 to the difference of Discharge Date and Admission Date 

Train_Inpatientdata['AdmissionDt'] = pd.to_datetime(Train_Inpatientdata['AdmissionDt'])
Train_Inpatientdata['DischargeDt'] = pd.to_datetime(Train_Inpatientdata['DischargeDt'])
Train_Inpatientdata['AdmitForDays'] = ((Train_Inpatientdata['DischargeDt'] - Train_Inpatientdata['AdmissionDt']).dt.days.abs())+1


Test_Inpatientdata['AdmissionDt'] = pd.to_datetime(Test_Inpatientdata['AdmissionDt'])
Test_Inpatientdata['DischargeDt'] = pd.to_datetime(Test_Inpatientdata['DischargeDt'])
Test_Inpatientdata['AdmitForDays'] = ((Test_Inpatientdata['DischargeDt'] - Test_Inpatientdata['AdmissionDt']).dt.days.abs())+1


# In[22]:


Train_Inpatientdata.loc[:,['AdmissionDt','DischargeDt','AdmitForDays']]


# In[23]:


## Lets check Min and Max values of AdmitforDays column in Train and Test.
print('Min AdmitForDays Train:- ',Train_Inpatientdata.AdmitForDays.min())
print('Max AdmitForDays Train:- ',Train_Inpatientdata.AdmitForDays.max())
print(Train_Inpatientdata.AdmitForDays.isnull().sum() )  #Check Null values.

print('Min AdmitForDays Test:- ',Test_Inpatientdata.AdmitForDays.min())
print('Max AdmitForDays Test:- ',Test_Inpatientdata.AdmitForDays.max())
print(Test_Inpatientdata.AdmitForDays.isnull().sum())   #Check Null values.


# ### Data Preprocessing on Outpatient Dataset

# In[24]:


# Summary of Outpatient Dataset

print('\033[1m'+"Train Outpatient Dataset"+ "\033[0m")

#display(Train_Outpatientdata.head(5))

print('\033[1m'+"Test Outpatient Dataset"+ "\033[0m")

#display(Train_Outpatientdata.head(5))


# In[25]:


# Lets check the null values in each column of Outpatient Dataset

print('\033[1m'+"Train Outpatient Dataset"+ "\033[0m")

print(Train_Outpatientdata.isna().sum())

print('\033[1m'+"Test Outpatient Dataset"+ "\033[0m")

print(Test_Outpatientdata.isna().sum())


# In[26]:


## Lets Check Shape of datasets after adding new variables

print('Shape of Train data :',Train.shape)
print('Shape of Train_Beneficiarydata data :',Train_Beneficiarydata.shape)
print('Shape of Train_Inpatientdata data :',Train_Inpatientdata.shape)
print('Shape of Train_Outpatientdata data :',Train_Outpatientdata.shape)

print('Shape of Test data :',Test.shape)
print('Shape of Test_Beneficiarydata data :',Test_Beneficiarydata.shape)
print('Shape of Test_Inpatientdata data :',Test_Inpatientdata.shape)


# ### Merge Beneficiary, Inpatient and Outpatient Dataset into a single dataset 
# 

# #### Merging of Train Datasets 

# In[27]:



Train_patient_merge_id = [i for i in Train_Outpatientdata.columns if i in Train_Inpatientdata.columns]

# Merge Inpatient, Outpatient and beneficiary dataframe into a single patient dataset
Train_Patient_data = pd.merge(Train_Inpatientdata, Train_Outpatientdata,
                    left_on = Train_patient_merge_id,
                    right_on = Train_patient_merge_id,
                    how = 'outer').\
          merge(Train_Beneficiarydata,left_on='BeneID',right_on='BeneID',how='inner')


# #### Merging of Test Dataset

# In[28]:


Test_patient_merge_id = [i for i in Test_Outpatientdata.columns if i in Test_Inpatientdata.columns]

# Merge Inpatient, Outpatient and beneficiary dataframe into a single patient dataset
Test_Patient_data = pd.merge(Test_Inpatientdata, Test_Outpatientdata,
                    left_on = Test_patient_merge_id,
                    right_on = Test_patient_merge_id,
                    how = 'outer').\
          merge(Test_Beneficiarydata,left_on='BeneID',right_on='BeneID',how='inner')


# In[29]:


# Shape of Merging Dataset 

print("Train Dataset Shape after merge:",Train_Patient_data.shape)

print("Test Dataset Shape after merge:",Test_Patient_data.shape)


# ### Exploratory Data Analysis on Train_Patient_data dataset

# In[30]:


Train_Patient_data.info()


# #### Handling Missing values 

# In[31]:


# To check the number of missing values in the Train_Pateint_data

Train_Patient_data.isnull().sum()


# In[32]:


# Summary of the dataset 

Train_Patient_data.describe(include='all')


# In[33]:


### There are missing values in AttendingPhysician, OperatingPhysician and OtherPhysician columns, so we need to handle these varaibles 

Train_Patient_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']]


# In[34]:


Train_Patient_data[['AttendingPhysician','OperatingPhysician', 'OtherPhysician']].describe()


# In[35]:


## We are replacing these columns value with 0 and 1 where we have value we are replacing it with 1 and in place of null value we replace it with 0.


Train_Patient_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']] = np.where(Train_Patient_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']].isnull(), 0, 1)


# In[36]:


Train_Patient_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']]


# In[37]:


### Add a new variable in which it tells us how many total types of physicians used for the particular claim or patient.


Train_Patient_data['N_Types_Physicians'] = Train_Patient_data['AttendingPhysician'] +  Train_Patient_data['OperatingPhysician'] + Train_Patient_data['OtherPhysician']


# In[38]:


Train_Patient_data['N_Types_Physicians']


# In[39]:


Train_Patient_data.isnull().sum() #We can see here new variable "N_Type_Physicians" is added


# In[40]:


### Handling Missing values on"DiagnosisGroupCode"

Train_Patient_data['DiagnosisGroupCode'].describe()


# In[41]:


# Here we are finding out each DignosisGroupCode Count

Count_DiagnosisGroupCode=Train_Patient_data['DiagnosisGroupCode'].value_counts()
Count_DiagnosisGroupCode=Count_DiagnosisGroupCode[:20] # To show only top 20 codes 
Count_DiagnosisGroupCode


# In[42]:


### Visualization of top 20 DignosisGroupCode

fig=plt.figure(figsize=(20,8))
sns.barplot(Count_DiagnosisGroupCode.index,Count_DiagnosisGroupCode.values)
fig.tight_layout()

## From here we can see that DignosisGroupCode 882 has maximum count that is 179 


# In[43]:


## Since in this columns we have maximum values as null, so we are handling this by creating a new column
## so we are creating a new variable/column "IsDiagnosisCode" in which value will either "1" or "0" 
## if in a claim there is a groupDiagnosiscode  has null value then in "IsDiagnosisCode" column value is 0 otherwise 1

Train_Patient_data['IsDiagnosisCode'] = np.where(Train_Patient_data.DiagnosisGroupCode.notnull(), 1, 0)
Train_Patient_data = Train_Patient_data.drop(['DiagnosisGroupCode'], axis = 1) # We are droping the column "DiagnosisGroupCode"


# In[44]:


Train_Patient_data['IsDiagnosisCode']


# In[45]:


### Handling missing values for "DeductibleamtPaid" column

Train_Patient_data['DeductibleAmtPaid'].isnull().sum()  #Check number of missing values in this variable


# In[46]:


# Describing this column by omiting the Nan, to check mean , variance , skewness etc

sc.stats.describe(Train_Patient_data['DeductibleAmtPaid'],nan_policy='omit')


# In[47]:


Train_Patient_data['DeductibleAmtPaid'].median() # Median of this variable 


# In[48]:


## Count Plot of "DeductibleAmtPaid" maximum values are 0 in this 

fig=plt.figure(figsize=(15,10))
sns.countplot(Train_Patient_data['DeductibleAmtPaid'])


# In[49]:


## Box plot of this "DeductibleAmtPaid", maximum values are 0 that shows here.

fig=plt.figure(figsize=(8,6))
sns.boxplot(Train_Patient_data['DeductibleAmtPaid'])
fig.tight_layout()


# In[50]:


## So from the above analysis we can reach to the conclusion that we replace missing values with 0 

Train_Patient_data['DeductibleAmtPaid'].fillna(0,inplace=True)


# In[51]:


### We are also creating one new variable "IsDeductibleAmtPaid" which tells us that particular claim has any DeductibleAmtPaid or not

Train_Patient_data['IsDeductibleAmtPaid']=np.where(Train_Patient_data['DeductibleAmtPaid']==0,0,1) 


# In[52]:


# So from this plot we can say that maximum claims doesn't have any "DeductibleAmtPaid"

fig=plt.figure(figsize=(8,6))
sns.countplot(Train_Patient_data['IsDeductibleAmtPaid'])

print(Train_Patient_data['IsDeductibleAmtPaid'].value_counts())


# In[53]:


### Handling missing values for "AdmitForDays" column

Train_Patient_data['AdmitForDays'].isnull().sum() # Count of missing values in this column


# In[54]:


# Replace all value with 0 as these all are the patients that didn't admit in the hospital

Train_Patient_data['AdmitForDays'].fillna(0,inplace=True)


# In[55]:


Train_Patient_data['AdmitForDays'].isnull().sum()


# In[56]:


#In this dataset now we have some Date columns in which missing values are there, which we do not need to handle and we can drop those columns also. 

Train_Patient_data.isnull().sum() 


# Now we need to handle missing values of ClmDiagnosisCodes and ClmProcedureCode columns 


# In[57]:


## First we handle ClmProcedureCodes variables 

ClmProcedure_vars = ['ClmProcedureCode_{}'.format(x) for x in range(1,7)]
ClmProcedure_vars


# In[58]:


Train_Patient_data[ClmProcedure_vars]


# In[59]:


Train_Patient_data[ClmProcedure_vars].describe()


# In[60]:


## To Check how many null values are in each Clmprocedurecodes
## By this we find out that in code_6 column all are Nan values 

Train_Patient_data[ClmProcedure_vars].isnull().sum()


# In[61]:


# This function helps us find the length of unique values in each row/record
def N_unique_values(df):
    return np.array([len(set([i for i in x[~pd.isnull(x)]])) for x in df.values])


# In[62]:


# We count the number of procedureCode for each claim and store these value in a new variable
Train_Patient_data['N_Procedure'] = N_unique_values(Train_Patient_data[ClmProcedure_vars])


# In[63]:


## So from here we get to know that 534901 claims/records has 0 claim procedure codes, 17820 claims/records has 1 claimprocedurecodes and so on

Train_Patient_data['N_Procedure'].value_counts()


# In[64]:


### Handling of 'ClmDiagnosisCode'

# We count the number of claims
ClmDiagnosisCode_vars =['ClmAdmitDiagnosisCode'] + ['ClmDiagnosisCode_{}'.format(x) for x in range(1, 11)]


ClmDiagnosisCode_vars


# In[65]:


# We count the number of CLMDiagnosisCode for each claim and store these value in a new variable

Train_Patient_data['N_UniqueDiagnosis_Claims'] = N_unique_values(Train_Patient_data[ClmDiagnosisCode_vars])


# In[66]:


Train_Patient_data['N_UniqueDiagnosis_Claims'].value_counts()


# In[67]:


Train_Patient_data.info()


# #### EDA on other remaining variables 
# 
# #### 1.Gender

# In[68]:


Train_Patient_data.Gender.describe()  


# In[69]:


Train_Patient_data.Gender.value_counts() # here we have only 1 and 2, so we can change it to binary as 0 or 1 


# In[70]:


Train_Patient_data['Gender']=Train_Patient_data['Gender'].replace(2,0) # replacing 2 with 0 


# In[71]:


## Countplot of Gender Column, Here we can consider 0 as Female and 1 as Male

fig=plt.figure(figsize=(8,6))
sns.countplot(Train_Patient_data['Gender'])
fig.tight_layout()


# #### 2.Race

# In[72]:


Train_Patient_data['Race'].describe()


# In[73]:


### Countplot of Race variable 
### From here we can find out that majority of claims are from Race 1
fig=plt.figure(figsize=(8,6))
sns.countplot(Train_Patient_data['Race'])
fig.tight_layout()


# In[74]:


### Now in Race column we do 'one hot encoding' so that ranking of values doesn't occur here 

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(Train_Patient_data.Race.values.reshape(-1, 1)).toarray()


# In[75]:


df_OneHot = pd.DataFrame(x, columns = ["Race_"+str(int(i)) for i in range(1,5)]) 
df_OneHot


# In[76]:


df_OneHot.drop('Race_1',axis=1,inplace=True) ## Drop the first column "Race_1" this we need to drop when we do oneHotEncoding
df_OneHot


# In[77]:


## Concatenation of dataframe "df_oneHot" that we created above in our main dataset

Train_Patient_data = pd.concat([Train_Patient_data, df_OneHot], axis=1)


# In[78]:


Train_Patient_data.drop(['Race'], axis=1,inplace=True)  #So now we do not need this race column so we are droping this also 


# #### 3. RenealDiseaseIndicator

# In[79]:


Train_Patient_data['RenalDiseaseIndicator'].describe()


# In[80]:


## Countplot of "RenalDiseaseIndicator" variable from here we can findout that maximu disease doesn't have any RenalDisease
fig=plt.figure(figsize=(8,6))
sns.countplot(Train_Patient_data['RenalDiseaseIndicator'])
fig.tight_layout()


# In[81]:


Train_Patient_data['RenalDiseaseIndicator']=Train_Patient_data.RenalDiseaseIndicator.astype(int) # Change of datatype from object to int


# In[82]:


Train_Patient_data['RenalDiseaseIndicator'].describe()


# #### 4. State and County

# In[83]:


Train_Patient_data[['State','County']].describe()


# In[84]:


#Find out which state has maximum count of claims

state_count=Train_Patient_data['State'].value_counts()
state_count=state_count[:20]
state_count


# In[85]:


##Count plot of top 20 states which have maximum claims  

## from here we can see that state code 5 has maximum number of claims 

fig=plt.figure(figsize=(10,6))
sns.barplot(state_count.index,state_count.values,order=state_count.index)
fig.tight_layout()


# In[86]:


#Find out which County has maximum count of claims
county_count=Train_Patient_data['County'].value_counts()
county_count=county_count[:20]
county_count


# In[87]:


##Count plot of top 20 County which have maximum claims  

## from here we can see that County code 200 has maximum number of claims 


fig=plt.figure(figsize=(12,6))
sns.barplot(county_count.index,county_count.values,order=county_count.index)
fig.tight_layout()


# #### 5. Chronic_cond

# In[88]:


## Visulization of ChronicCond Variables 

## From this we can findout that how many claims has ChronicCond diseases, for eg: In ChronicCond_Alzheimer more than 3 lacs claims doesn't have this and remaining claims approx( 2 lacs) have ChronicCond_Alzheimer

fig=plt.figure(figsize=(20,20))

for col in range(1,12):
    plt.subplot(6,2,col)
    sns.countplot(Train_Patient_data.iloc[:,37+col])
    
fig.tight_layout()


# #### Boxplots of some numerical features to check the distribution of data 

# In[89]:


## Boxplot of "IPAnnualReimbursementAmt" and we can see in this boxplot data is not normally distributed and it is left skewed 

fig=plt.figure(figsize=(8,6))
sns.boxplot(Train_Patient_data['IPAnnualReimbursementAmt'])
fig.tight_layout()


# In[90]:



## Boxplot of "IPAnnualDeductibleAmt" and we can see in this boxplot data is not normally distributed and it is left skewed


fig=plt.figure(figsize=(8,6))
sns.boxplot(Train_Patient_data['IPAnnualDeductibleAmt'])
fig.tight_layout()


# In[91]:


## Boxplot of "OPAnnualReimbursementAmt" and we can see in this boxplot data is not normally distributed and it is left skewed

fig=plt.figure(figsize=(8,6))
sns.boxplot(Train_Patient_data['OPAnnualReimbursementAmt'])
fig.tight_layout()


# In[92]:



## Boxplot of "OPAnnualDeductibleAmt" and we can see in this boxplot data is not normally distributed and it is left skewed

fig=plt.figure(figsize=(8,6))
sns.boxplot(Train_Patient_data['OPAnnualDeductibleAmt'])
fig.tight_layout()


# ### Handling  Missing values and add new features in Test_Patient_data

# In[93]:


Test_Patient_data.isnull().sum()


# In[94]:


## We are replacing these columns value with 0 and 1 where we have value we are replacing it with 1 and in place of null value we replace it with 0.


Test_Patient_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']] = np.where(Test_Patient_data[['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']].isnull(), 0, 1)

Test_Patient_data['N_Types_Physicians'] = Test_Patient_data['AttendingPhysician'] +  Test_Patient_data['OperatingPhysician'] + Test_Patient_data['OtherPhysician']


# In[95]:


Test_Patient_data['N_Types_Physicians']


# In[96]:


Test_Patient_data['IsDiagnosisCode'] = np.where(Test_Patient_data.DiagnosisGroupCode.notnull(), 1, 0)
Test_Patient_data = Test_Patient_data.drop(['DiagnosisGroupCode'], axis = 1)


# In[97]:


Test_Patient_data.isnull().sum()


# In[98]:


Test_Patient_data['DeductibleAmtPaid'].describe()


# In[99]:


Test_Patient_data['DeductibleAmtPaid'].fillna(0,inplace=True)


# In[100]:


Test_Patient_data['IsDeductibleAmtPaid']=np.where(Test_Patient_data['DeductibleAmtPaid']==0,0,1) 


# In[101]:


Test_Patient_data['IsDeductibleAmtPaid'].value_counts()


# In[102]:


Test_Patient_data['AdmitForDays'].isnull().sum()


# In[103]:


Test_Patient_data['AdmitForDays'].fillna(0,inplace=True)


# In[104]:


Test_Patient_data.Gender.describe()


# In[105]:


Test_Patient_data['Gender']=Test_Patient_data['Gender'].replace(2,0)


# In[106]:


Test_Patient_data['Race'].describe()


# In[107]:


onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(Test_Patient_data.Race.values.reshape(-1, 1)).toarray()


# In[108]:


df_test_OneHot = pd.DataFrame(x, columns = ["Race_"+str(int(i)) for i in range(1,5)]) 
df_test_OneHot


# In[109]:


df_test_OneHot.drop('Race_1',axis=1,inplace=True)


# In[110]:


Test_Patient_data = pd.concat([Test_Patient_data, df_test_OneHot], axis=1)

#droping the country column 


# In[111]:


Test_Patient_data.drop(['Race'], axis=1,inplace=True) 


# In[112]:


Test_Patient_data.info()


# In[113]:


Test_Patient_data['RenalDiseaseIndicator'].describe()


# In[114]:


Test_Patient_data['RenalDiseaseIndicator']=Test_Patient_data.RenalDiseaseIndicator.astype(int)


# In[115]:


Test_Patient_data[ClmProcedure_vars].describe()


# In[116]:


# We count the number of procedures for each claim
Test_Patient_data['N_Procedure'] = N_unique_values(Test_Patient_data[ClmProcedure_vars])


# In[117]:


Test_Patient_data['N_Procedure'].value_counts()


# In[118]:


# We count the number of CLMDiagnosisCode for each claim and store these value in a new variable

Test_Patient_data['N_UniqueDiagnosis_Claims'] = N_unique_values(Test_Patient_data[ClmDiagnosisCode_vars])

Test_Patient_data['N_UniqueDiagnosis_Claims'].value_counts()


# In[119]:


print('\033[1m'+"Train Patient Dataset"+ "\033[0m")

print(Train_Patient_data.info())

print('\033[1m'+"Test Patient Dataset"+ "\033[0m")

print(Test_Patient_data.info())


# ## Merging of Train and Test dataframe with Train_Patient_data and Test_Patient_data respectively to create a Final Dataframe for Train and Test for modelling  

# In[120]:


### Count number of records
## From here we get the count of BeneID and ClaimId for each provider

## For Train 
Train_Count = Train_Patient_data[['BeneID', 'ClaimID']].groupby(Train_Patient_data['Provider']).nunique().reset_index()
Train_Count.rename(columns={'BeneID':'BeneID_count','ClaimID':'ClaimID_count'},inplace=True)


## For Test
Test_Count = Test_Patient_data[['BeneID', 'ClaimID']].groupby(Test_Patient_data['Provider']).nunique().reset_index()
Test_Count.rename(columns={'BeneID':'BeneID_count','ClaimID':'ClaimID_count'},inplace=True)


# In[121]:


Train_Count


# In[122]:


Test_Count


# In[123]:


## Here we are calculating the total sum of values for each unique provider 


Train_Data_Sum = Train_Patient_data.groupby(['Provider'], as_index = False)[['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'RenalDiseaseIndicator', 
                                                     'AttendingPhysician','OperatingPhysician','OtherPhysician','AdmitForDays',
                                                    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure','ChronicCond_Cancer', 
                                                    'ChronicCond_KidneyDisease', 'ChronicCond_ObstrPulmonary',
                                                   'ChronicCond_Depression','ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',   
                                                    'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                                                    'ChronicCond_stroke', 'IPAnnualReimbursementAmt','IPAnnualDeductibleAmt',
                                                    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'WhetherDead',
                                                    'N_Types_Physicians','IsDiagnosisCode', 'N_Procedure', 'N_UniqueDiagnosis_Claims']].sum()



Test_Data_Sum = Test_Patient_data.groupby(['Provider'], as_index = False)[['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'RenalDiseaseIndicator', 
                                                     'AttendingPhysician','OperatingPhysician','OtherPhysician','AdmitForDays',
                                                    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure','ChronicCond_Cancer', 
                                                    'ChronicCond_KidneyDisease', 'ChronicCond_ObstrPulmonary',
                                                   'ChronicCond_Depression','ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',   
                                                    'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                                                    'ChronicCond_stroke', 'IPAnnualReimbursementAmt','IPAnnualDeductibleAmt',
                                                    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'WhetherDead',
                                                    'N_Types_Physicians','IsDiagnosisCode', 'N_Procedure', 'N_UniqueDiagnosis_Claims']].sum()


# In[124]:


Train_Data_Sum


# In[125]:


Test_Data_Sum


# In[126]:


## Here we are calculating the mean of values for some variables for each unique provider.

Train_Data_Mean=round(Train_Patient_data.groupby(['Provider'], as_index = False)[['NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
                                                                            'Age']].mean())


Test_Data_Mean=round(Test_Patient_data.groupby(['Provider'], as_index = False)[['NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
                                                                            'Age']].mean())


# In[127]:


Train_Data_Mean


# In[128]:


Test_Data_Mean


# #### Now we merge Count,sum and mean dataframes with the main train dataframe

# In[129]:


## Merging of Train Datasets
Train_df=pd.merge(Train_Count,Train_Data_Sum,on='Provider',how='left').                merge(Train_Data_Mean,on='Provider',how='left').                merge(Train,on='Provider',how='left')

## Merging of Test Datasets

Test_df=pd.merge(Test_Count,Test_Data_Sum,on='Provider',how='left').                merge(Test_Data_Mean,on='Provider',how='left').                merge(Test,on='Provider',how='left')


# In[130]:


Train_df #Target column PotentialFraud is avaialble here


# In[131]:


Test_df #Target column PotentialFraud is not avaialble here


# In[132]:


Train_df.isnull().sum() ## No null value is present in this dataset 


# In[133]:


Test_df.isnull().sum() ## No null value is present in this dataset 


# In[134]:


#In Train Dataset Target variable PotentialFraud has value in category i.e "Yes" and "No" need to replace with 1 and 0.

Train_df['PotentialFraud']=np.where(Train_df.PotentialFraud == "Yes", 1, 0)


# In[135]:


Train_df


# In[136]:


# Here we can the count of Dependent variable values 
plt.figure(figsize=(10,8))
sns.countplot(Train_df.PotentialFraud)


# ###  Bivariant Data Analysis

# In[137]:


## Here we can se the barplot of PotentialFraud v/s BeneID_Count and here bar shows mean of BeneID_Count for Potential Fraud value 1 and 0
## From this barplot we can conclude that there is a Potential Fraud when the BeneID_Count is more as its mean is more as shown.

plt.figure(figsize=(12,8))
sns.barplot(Train_df["PotentialFraud"],Train_df["BeneID_count"], hue=Train_df["PotentialFraud"])
plt.suptitle('PotentialFraud v/s BeneID_count')
plt.xlabel('PotentialFraud')
plt.ylabel('BeneID_count')


# As we can see Fraudulant claims have higher number of Beneficiary ID as they tend to commit fraud with multiple beneficiary id.

# In[138]:


## Here we can se the barplot of PotentialFraud v/s ClaimID_Count and here bar shows mean of ClaimID_Count for Potential Fraud value 1 and 0
## From this barplot we can conclude that there is a Potential Fraud when the ClaimID_Count is more as its mean is more as shown.

plt.figure(figsize=(12,8))
sns.barplot(Train_df["PotentialFraud"],Train_df["ClaimID_count"], hue=Train_df["PotentialFraud"])


# Same as the above observation, potential fraud claims tend to have higher number of Claim ID.

# In[139]:


## Here we can se the barplot of PotentialFraud v/s InscClaimAmtReimbursed and here bar shows mean of InscClaimAmtReimbursed for Potential Fraud value 1 and 0
## From this barplot we can conclude that there is a Potential Fraud when the InscClaimAmtReimbursed is more as its mean is more as shown.

plt.figure(figsize=(12,8))

sns.barplot(Train_df["PotentialFraud"],Train_df["InscClaimAmtReimbursed"], hue=Train_df["PotentialFraud"])


# In[140]:


plt.figure(figsize=(12,8))

sns.barplot(Train_df["PotentialFraud"],Train_df["DeductibleAmtPaid"], hue=Train_df["PotentialFraud"])


# As we have observed both in InscClaimAmtReimbursed and DeductibleAmtPaid are way higher than the legitimate claims.

# In[141]:



plt.figure(figsize=(12,8))
sns.barplot(Train_df["PotentialFraud"],Train_df["RenalDiseaseIndicator"], hue=Train_df["PotentialFraud"])


# In[142]:



plt.figure(figsize=(12,8))
sns.barplot(Train_df["PotentialFraud"],Train_df["AdmitForDays"], hue=Train_df["PotentialFraud"])


# In[143]:


plt.figure(figsize=(12,8))
sns.barplot(Train_df["PotentialFraud"],Train_df["WhetherDead"], hue=Train_df["PotentialFraud"])


# In category 0, the bar is between 0 and 1 because there are some people who are dead and some are alive, but in category 1 the bar has gone above 3 that means fraudulant claims are more likely to happen where people are dead.

# In[144]:


Train_df.info()


# ### Correlation Matrix

# #### Correlation is used to find out the relationship between Dependent Variable with independent variables 
# 

# In[145]:


plt.figure(figsize=(12,8))
Train_corr=Train_df.corr()
sns.heatmap(Train_corr)


# In[146]:


Train_corr=Train_df.corr()
Train_corr['PotentialFraud']


# So from here we can see that Age, NoOfMonths_PartBCov and NoOfMonths_PartACov are not making any pattern/relationship with dependent variable 'PotentialFraud', hence we will not consider these variables in our model 

# We will make a final dataset on which we will do modelling,In this dataset we keep only those variable which we will use in our machine learning modelling algorithms. So from our Train_df dataset we will remove all ID type variables like Provider,BeneID_count and ClaimID_count and also remove those variable which are not making any pattern with the dependent variable this we can see correlation matrix that is shown above 

# In[147]:


df_clf=Train_df.iloc[:,3:]
df_clf


# In[148]:


df_clf.drop(['NoOfMonths_PartACov','NoOfMonths_PartBCov','Age'],axis=1,inplace=True)


# ### Final Train Dataset on which we trained our model 

# In[149]:


df_clf #This is final Trained Dataset


# ### Final Test Dataset on which we will do final Prediction 

# In[150]:


Test_df


# In[151]:


def test(test_data):
    test_data=test_data.iloc[:,3:]
    test_data=test_data.drop(['NoOfMonths_PartACov','NoOfMonths_PartBCov','Age'],axis=1)
    return test_data


# In[152]:


Test_data=test(Test_df)
Test_data ## In this target varaible is not there we need to predict this after we trained our model 


# In[153]:


## Send this test dataset to our local system in csv format 

Test_data.to_csv("C:\\Users\\Rahul\\Desktop\\Fraud Detection\\Dataset\\Test_data.csv",index=False)


# ### Working on our Train Dataset 

# In[154]:


#Split the dataset into Independent and Dependent Features

x=df_clf.drop("PotentialFraud",axis=1)
y=df_clf.PotentialFraud


# In[155]:


print("Independent Variable shape:",x.shape)
print("Dependent Variable shape:",y.shape)


# ### Train Test Split

# In[156]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)


# In[157]:


print("Independent variables train:",x_train.shape)
print("Target variable train:",y_train.shape)
print("Independent variables test:",x_test.shape)
print("Target variables test:",y_test.shape)


# ## Modelling 

# In[158]:


#Here we can see that our target vairable is imbalanced as "0" class is is majority and "1" class is in minority 
plt.figure(figsize=(10,8))
sns.countplot(y_train)


# Since our target variable is imbalanced hence we need to do the modelling using sampling techniques 

# ### Sampling Techniques 

# #### 1. Under Sampling

# In[159]:


from collections import Counter
from imblearn.under_sampling import NearMiss



ns=NearMiss(0.8)
x_train_ns,y_train_ns=ns.fit_sample(x_train,y_train) ## Create new train dataset after fitting undersmapling
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))


# In[160]:


plt.figure(figsize=(10,8))
sns.countplot(y_train_ns)


# Here we can see that our target variable is balanced now 

# In[161]:


print("Shape of independent trained dataset after under sampling",x_train_ns.shape)
print("Shape of dependent trained dataset after under sampling", y_train_ns.shape)


# ### Modelling on Under sampled Dataset  

# ### Random Forest on Undersampled Data

# In[162]:


from sklearn.ensemble import RandomForestClassifier


clf=RandomForestClassifier()
clf_fit=clf.fit(x_train_ns,y_train_ns)

y_pred_rf=clf_fit.predict(x_test)


# In[163]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_rf))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_rf))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_rf))


# Here we can see that this model accuracy is low or not good.This model is able to classify "1" class but not able to classify "0" class 

# ### SVM on Undersampled Data 

# In[164]:


from sklearn.svm import SVC


clf_svc=SVC()
clf_svc_fit=clf_svc.fit(x_train_ns,y_train_ns)

y_pred_svc=clf_svc_fit.predict(x_test)


# In[165]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_svc))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_svc))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_svc))


# In SVM also accuracy score is okay, but here are many miscalssification in "0" class, Hence we go for another sampling technique 
# 

# ### 2. Over Sampling

# In[166]:


from imblearn.over_sampling import RandomOverSampler


os=RandomOverSampler(0.75)
x_train_os,y_train_os=os.fit_sample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_os)))


# In[167]:


plt.figure(figsize=(10,8))
sns.countplot(y_train_os)


# Here we can see that our target variable is balanced now 

# In[168]:


print("Shape of independent trained dataset after Over sampling",x_train_os.shape)
print("Shape of dependent trained dataset after Over sampling", y_train_os.shape)


# ### Modelling on Over Sampled Dataset 

# ### Random Forest on Over Sampled Dataset 

# In[169]:


from sklearn.ensemble import RandomForestClassifier


os_clf=RandomForestClassifier()
os_clf_fit=os_clf.fit(x_train_os,y_train_os)

y_pred_rf_os=os_clf_fit.predict(x_test)


# In[170]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_rf_os))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_rf_os))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_rf_os))


# Here Random Forest gives us good accuracy score but it is not able to classify majority of "1" class correctly  

# ### SVM on Over Sampled Data 

# In[171]:


from sklearn.svm import SVC


clf_svc_os=SVC()
clf_svc_fit_os=clf_svc_os.fit(x_train_os,y_train_os)

y_pred_svc_os=clf_svc_fit_os.predict(x_test)


# In[172]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_svc_os))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_svc_os))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_svc_os))


# Here SVM gives us good accuracy score but there are some misclassification in "1" class, Now we want to minimize this misclassification, Hence we go to another sampling technique 

# ### 3. Synthetic Minority Oversampling Technique (SMOTE) 

# In[173]:


from imblearn.combine import SMOTETomek



os=SMOTETomek(0.75)
x_train_st,y_train_st=os.fit_sample(x_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_st)))


# In[174]:


plt.figure(figsize=(10,8))
sns.countplot(y_train_st)


# In[175]:


print("Shape of independent trained dataset after SMOTE sampling",x_train_st.shape)
print("Shape of dependent trained dataset after SMOTE sampling", y_train_st.shape)


# ### Modelling on SMOTE Sampling 

# ### Decision Tree on SMOTE Sampled Data 

# In[176]:


from sklearn.tree import DecisionTreeClassifier


clf_dt=DecisionTreeClassifier()
model_dt=clf_dt.fit(x_train_st,y_train_st)
y_pred_dt=model_dt.predict(x_test)


# In[177]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_dt))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_dt))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_dt))


# In Decision Tree model accuracy is good but there are many misclassification for both the classes.

# ### Naive bayes 

# In[178]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB


clf_nb=GaussianNB()
model_nb=clf_nb.fit(x_train_st,y_train_st)
y_pred_nb=model_nb.predict(x_test)


# In[179]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_nb))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_nb))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_nb))


# ### Gradient Boosting Classifier 

# In[180]:


from sklearn.ensemble import GradientBoostingClassifier

clf_GB=GradientBoostingClassifier()
model_GB=clf_GB.fit(x_train_st,y_train_st)
y_pred_GB=model_GB.predict(x_test)


# In[181]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_GB))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_GB))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_GB))


# ### Random Forest on SMOTE sampled Data 

# In[182]:


clf_st=RandomForestClassifier()
model_rf=clf_st.fit(x_train_st,y_train_st)
y_pred_rf=model_rf.predict(x_test)


# In[183]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_rf))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_rf))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_rf))


# ### ROC Curve

# In[184]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[185]:


probs = model_rf.predict_proba(x_test)
probs = probs[:, 1]
probs


# In[186]:


fpr1, tpr1, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr1, tpr1)


print('\033[1m'+"AUC Score \n"+'\033[0m', roc_auc_score(y_test, probs))


# ### SVM on SMOTE Sampled Data 

# In[187]:


clf_svm=SVC(probability=True)
model_svm=clf_svm.fit(x_train_st,y_train_st)
y_pred_svm=model_svm.predict(x_test)


# In[188]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_svm))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_svm))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_svm))


# SVM is working good here as we can see misclassification for classes is also less 

# In[189]:


probs_svm = model_svm.predict_proba(x_test)
probs_svm = probs_svm[:, 1]
probs_svm


# #### ROC Curve for SVM 

# In[190]:


fpr2, tpr2, thresholds = roc_curve(y_test, probs_svm)

plot_roc_curve(fpr2, tpr2)


print('\033[1m'+"AUC Score \n"+'\033[0m', round(roc_auc_score(y_test, probs_svm),2))


# ### Comparison of Models Accuracy 

# In[191]:


rf_accuracy=round(accuracy_score(y_test,y_pred_rf),4)
svm_accuracy=round(accuracy_score(y_test,y_pred_svm),4)
GB_accuracy=round(accuracy_score(y_test,y_pred_GB),4)
DT_accuracy=round(accuracy_score(y_test,y_pred_dt),4)
NB_accuracy=round(accuracy_score(y_test,y_pred_nb),4)


# In[192]:


Accuracy=pd.DataFrame({"Model":["Decision Tree","Naive Bayes","Random Forest","SVM","Gradient Boosting"],"Accuracy":[DT_accuracy,NB_accuracy,rf_accuracy,svm_accuracy,GB_accuracy]})
Accuracy


# In[193]:


sns.barplot(x=Accuracy.Model,y=Accuracy.Accuracy,)


# ### Comparison of ROC Curve between RandomForest and SVM 

# In[194]:


plt.plot(fpr2, tpr2,color='orange',label='SVM')
plt.plot(fpr1, tpr1,color='green',label='Random Forest')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.legend()
plt.show();


# #### Best result is coming from SVM till now, But there are other models also which are giving us good accuracy score but misclassification is more in that like Random Forest and Gradient Boosting. So now we will do Hyperparameter tunning, by which we try to increase the accuracy and reduce the misclassification of classes for Random Forest and Gradient Boosting.

# ### Hyperparameter Tunning 

# #### Random Forest Hyperparameter Tunning

# In[195]:


model_rf.get_params()


# In[196]:



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]

# Number of features to consider at every split
max_features = ['auto','log2']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 15, num = 5)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}

cv=KFold(n_splits=3,random_state=None,shuffle=False) #Cross validation


# In[197]:


param_grid


# In[269]:


model_rf_grid=RandomForestClassifier()

grid_model=GridSearchCV(estimator=model_rf_grid,param_grid = param_grid,cv=cv,n_jobs=1)
grid_model=grid_model.fit(x_train_st,y_train_st)


# In[270]:


print(grid_model.best_params_) # here we can get the best parmameters of the above model which gives us the best accuracy 


# In[271]:


y_pred_rfg=grid_model.predict(x_test)


# In[272]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_rfg))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_rfg))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_rfg))


# #### Gradient Boosting Hyperparameter Tunning 

# In[198]:


model_GB.get_params()


# In[199]:



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 400, num = 4)]

learning_rate = [0.1,0.05,0.2]

# Number of features to consider at every split
loss = ['deviance', 'exponential']





# Create the random grid
param_grid_GB = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'loss': loss}

cv=KFold(n_splits=3,random_state=None,shuffle=False) #Cross validation

param_grid_GB


# In[275]:


model_gb_grid=GradientBoostingClassifier()

grid_model_gb=GridSearchCV(estimator=model_gb_grid,param_grid = param_grid_GB,cv=cv,n_jobs=1)
grid_model_gb=grid_model_gb.fit(x_train_st,y_train_st)


# In[279]:


grid_model_gb.best_estimator_


# In[276]:


y_pred_gb_grid=grid_model_gb.predict(x_test)


# In[277]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_gb_grid))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_gb_grid))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_gb_grid))


# #### After Hyperparameter tunning also misclassification of classes did not decrease on both Random Forest and Gradient Boosting, Hence our final model will be SVM.

# ### Importing of Pickle file of SVM Model for Deployment 

# In[200]:


import pickle


# Save to file in the specified location
pkl_filename = "C:\\Users\\Rahul\\Desktop\\Fraud Detection\\fraud_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_svm, file)


# ### Prediction of Potential Fraud (Target Variable) on our main Test Data 

# In[201]:


Test_data


# In[202]:


PotentialFraud=model_svm.predict(Test_data)


# In[203]:


Potential_Fraud=pd.DataFrame(PotentialFraud,columns=['PotentialFraud'])


# In[204]:


Potential_Fraud


# In[205]:


Predicted_Test_data=pd.concat([Test_df,Potential_Fraud],axis=1)


# In[206]:


Predicted_Test_data


# In[207]:


### Export this predicted test data file to local system 

Predicted_Test_data.to_csv("C:\\Users\\Rahul\\Desktop\\Fraud Detection\\Dataset\\Predicted_Test_data.csv")


# ## Feature Importance 

# #### Get important features from RandomForest Model

# In[208]:


# Get numerical feature importances from Random Forest model
importances = list(model_rf.feature_importances_)
print(importances)


# In[209]:


feature_list=list(df_clf.columns)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#print(feature_importances)
# Print out the feature and importances 
print([print('Variable: {:20} Importance: {} '.format(*pair))  for pair in feature_importances])


# #### Get important Features from Gradient Boosting Model

# In[210]:


# Get numerical feature importances from Random Forest model
importances = list(model_GB.feature_importances_)
print(importances)


# In[211]:


feature_list=list(df_clf.columns)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#print(feature_importances)
# Print out the feature and importances 
print([print('Variable: {:20} Importance: {} '.format(*pair))  for pair in feature_importances])


# In[212]:


from xgboost import XGBClassifier


xg_model=XGBClassifier()
sg_model_fit=xg_model.fit(x_train_st,y_train_st)


# In[213]:


from xgboost import plot_importance
plot_importance(sg_model_fit)


# In[214]:


#### Hence from above we can conclude that important features are :

x_train_imp=x_train_st[['InscClaimAmtReimbursed','AdmitForDays','DeductibleAmtPaid','N_Procedure','IsDiagnosisCode']]
x_test_imp=x_test[['InscClaimAmtReimbursed','AdmitForDays','DeductibleAmtPaid','N_Procedure','IsDiagnosisCode']]
x_train_imp.head()


# #### Now we will do the modelling with important features only and then generate the pickle file

# In[215]:


## SVM model

svm_imp=SVC(probability=True)
svm_imp=svm_imp.fit(x_train_imp,y_train_st)
y_pred_svm2=svm_imp.predict(x_test_imp)


# In[216]:



print('\033[1m'+"Confusion Matrix \n"+'\033[0m',confusion_matrix(y_test,y_pred_svm2))
print('\033[1m'+"\n Accuracy Score \n"+'\033[0m',accuracy_score(y_test,y_pred_svm2))
print('\033[1m'+"\n Classification Report \n"+'\033[0m',classification_report(y_test,y_pred_svm2))


# #### ROC Curve of SVM Model on Important Features 

# In[217]:


probs_new = svm_imp.predict_proba(x_test_imp)
probs_new = probs_new[:, 1]
probs_new


# In[218]:


fpr, tpr, thresholds = roc_curve(y_test, probs_new)

plot_roc_curve(fpr, tpr)


print('\033[1m'+"AUC Score \n"+'\033[0m', round(roc_auc_score(y_test, probs_new),2))


# ### Importing of Pickle file of SVM model for Deployement 

# In[219]:


import pickle


# Save to file in the specified location
pkl_filename = "C:\\Users\\Rahul\\Desktop\\Fraud Detection\\fraud_detect.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svm_imp, file)


# In[224]:


output=0
txt3 = "My name is {yes}".format(yes=output)
print(txt3)


# In[226]:


if output==1:
    print("yes")
else:
    print("No")


# In[230]:


pre=["Yes" if output==1 else "NO"]


# In[232]:


pre[0]


# In[233]:


df_clf.columns


# In[ ]:




