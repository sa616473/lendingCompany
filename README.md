# lendingCompany
Objective: Can we predict trustworthiness?

## Table of contents
- Define Problem
    - Can we predict the right loan applicant from their background?
    - Goals

- Discover Data
   - Exploratory Data Analysis(EDA)
   - Data Visualization

- Develop solutions
    - Establish a baseline
    - Machine Learning Models
    - Hyperparameter Tuning & Optimization
    - Best model 

# Defining The Problem

#### Can we predict the right loan applicant from their background?

Through loans many people can pursue their future education or buy a home, car, favorite furniture or anything they like that will bring joy and happiness to them and their family.
- **But the most important question is who is going to pay back?**. 
- **How can a bank identify the right candiate apart from someone who is trying to scam the bank?** 
- **Can AI help us with this problem**

#### Goals
- Help lending club make the right decision when giving out loans.
- Help the right applicants to recive loans.

## Discovering the Data

#### Exploratory Data Analysis (EDA)

- The following are the raw features and their descriptions
LoanStatNew| Description
------------------- |------------
loan_amnt |The listed amount of the loan applied for by t...
term |The number of payments on the loan. Values are...
int_rate |Interest Rate on the loan
installment |The monthly payment owed by the borrower if th...
grade |LC assigned loan grade
sub_grade |LC assigned loan subgrade
emp_title |The job title supplied by the Borrower when ap...
emp_length |Employment length in years. Possible values ar...
home_ownership |The home ownership status provided by the borr...
annual_inc |The self-reported annual income provided by th...
verification_status |Indicates if income was verified by LC, not ve...
issue_d |The month which the loan was funded
loan_status |Current status of the loan
purpose |A category provided by the borrower for the lo...
title |The loan title provided by the borrower
zip_code |The first 3 numbers of the zip code provided b...
addr_state |The state provided by the borrower in the loan...
dti |A ratio calculated using the borrower’s total ...
earliest_cr_line |The month the borrower's earliest reported cre...
open_acc |The number of open credit lines in the borrowe...
pub_rec |Number of derogatory public records
revol_bal |Total credit revolving balance
revol_util |Revolving line utilization rate, or the amount...
total_acc |The total number of credit lines currently in ...
initial_list_status |The initial listing status of the loan. Possib...
application_type |Indicates whether the loan is an individual ap...
mort_acc |Number of mortgage accounts.
pub_rec_bankruptcies |Number of public record bankruptcies

- Missing data in percentage
feature |                mis_perc
-------------|-------------------
mort_acc    |            9.543469
emp_title   |            5.789208
emp_length   |           4.621115

As we can see the mort_acc has the most missing data so we looked at the <br/>
**correlations** based on **mort_acc** and imputed the missing values with <br/>
the **total_acc** features median value.

feature |               corr
--------------|----------------
mort_acc   |             1.000000
total_acc   |            0.381072
annual_inc   |           0.236320
loan_amnt     |          0.222315
revol_bal      |         0.194925
installment     |        0.193694
open_acc         |       0.109205
loan_repaid       |      0.073111
pub_rec_bankruptcies|    0.027239
pub_rec              |   0.011552
revol_util            |  0.007514
dti                    |-0.025439
int_rate               |-0.082583

#### Data Visualization

-- **Imbalance of the data plot**

- As we can see in the following plot we have more positve data compared to negative data this natuaral in this scenario because the companies that are lending money usally lend to people that they think will most likely pay back.

![count plot loan status](/reports/figures/graphs/png/loan_status_count.png)

-- **Correlation plot using heat map**
- Some basic visualization to check out the correlations

![correlations plot](/reports/figures/graphs/png/corr.png)

-- **count plot of grade F and G**
- Taking a closer look at the loans that usally default

![F and G](/reports/figures/graphs/png/sub_grade_count.png)

### Developing Solutions

#### Establishing the Baseline
- We used the Grade feature to predict our baseline model. Our base model by default predicts 0 for grade F and G
- With this model we acheived a accuracy of 80%
-- **Baseline confusion matrix**
![Baseline Confusion Matrix](/reports/figures/graphs/png/baseline_confusion_matrix.png)

#### Machine Learning Models
- Hypothesized models
- **XGBoost Classifies** with hyperparameter tuning
- **Densely connected Neural Network(DNN)** with Bayesian optimization
- **Densely connected Neural Network(DNN)** with hand tuning Hyperparameters

### Best Model
- Our Best models for now:
- **Densely connected Neural Network(DNN)** with Bayesian optimization
 |precision |   recall | f1-score  | support
 |----------|----------|-----------|---------
 0   |    0.47  |    0.93  |    0.62  |    3925
1    |   0.99   |   0.89  |    0.94  |   35597
 accuracy   |           |           |  0.89   |  39522
 macro avg   |    0.73   |   0.91    |  0.78   |  39522
weighted avg    |   0.94   |   0.89     | 0.90   |  39522

||  0    |  1  |
|---|--------|------|
|0| 3635 |  290|
|1 |4081 |31516|


![Baseline Confusion Matrix](/reports/figures/graphs/png/_confusion_matrix.png)


- **Densely connected Neural Network(DNN)** with hand tuning Hyperparameters
 |precision |   recall | f1-score  | support
 |----------|----------|-----------|---------
 0   |    0.43  |    1.00  |    0.62  |    3364
1    |   1.00  |   0.88  |    0.94  |   36158
 accuracy   |           |           |  0.89   |  39522
 macro avg   |    0.72   |   0.94    |  0.77   |  39522
weighted avg    |   0.95   |   0.89     | 0.91   |  39522

||  0    |  1  |
|---|--------|------|
|0| 3348 |  16|
|1 |4368 |31790|

![Baseline Confusion Matrix](/reports/figures/graphs/png/baseline_confusion_matrix.png)



