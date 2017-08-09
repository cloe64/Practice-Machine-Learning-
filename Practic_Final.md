
# Final Project Report: Practical Machine Learning

## Introduction

1.1 Background 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

1.2 Data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

1.3 Goal
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

## Data Load and Cleaning

```r
library(caret)
library(rpart)
library(rpart.plot)  
library(randomForest)
library(corrplot)
```


```r
urltrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urltest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train <- read.csv(url(urltrain), na.strings=c("NA","#DIV/0!",""))
valid <- read.csv(url(urltest), na.strings=c("NA","#DIV/0!",""))
set.seed(12345)
inTrain<-createDataPartition(train$classe,p=0.7,list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]
dim(training)
```

```
## [1] 13737   160
```

```r
dim(testing)
```

```
## [1] 5885  160
```

## Cleaning the data

There are 160 variables in the datasets. 1. Using Near Zero variance removed variables that contain one or few unique values or large ratio of most frequence common value to second frequence common value. 2. Remove the predictors which have percentage of missing values larger than 95%.3.Remove the time variables and row index (1-4).

```r
dataNZV<-nearZeroVar(training)
training<-training[,-dataNZV]
testing<-testing[,-dataNZV]
valid<-valid[,-dataNZV]
NAmost<-apply(training,2,function(x) mean(is.na(x))>0.95)
training<-training[,NAmost==FALSE]
testing<-testing[,NAmost==FALSE]
valid<-valid[,NAmost==FALSE]
training<-training[,-c(1,2,3,4)]
testing<-testing[,-c(1,2,3,4)]
valid<-valid[,-c(1,2,3,4)]
dim(training)
```

```
## [1] 13737    55
```

```r
dim(testing)
```

```
## [1] 5885   55
```

```r
dim(valid)
```

```
## [1] 20 55
```
Investigate the correlations among the predictors.There are 34 percentage of predictors correlated. No need to consider PCA in the preprocess of the data.

```r
index<-grep('classe',names(training))
tdata<-training[,-index]
number<-sapply(tdata,is.numeric)
Cordata<-tdata[,number]
Matrix<-abs(cor(Cordata))
diag(Matrix)<-0
length(which(Matrix>0.8))/2/55
```

```
## [1] 0.3454545
```
## Model Building

For this project frist repeated k-fold cross-validation for basic parameter tunining. There3 differnt model algorithms and then choose the one with the best out-of-sample accuracy. The three model types to test are:

1.Decision trees with CART (rpart)
2.gradient boosting trees (gbm)
3.Random forest decision trees (rf)

```r
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 3,
                           ## repeated ten times
                           repeats = 10)
dtreeFit <- train(classe~ ., data = training, 
                 method = "rpart", 
                 trControl = fitControl)
gbmFit <- train(classe~ ., data = training, 
                 method = "gbm", 
                 trControl = fitControl)
```

```
## Warning: package 'gbm' was built under R version 3.3.3
```

```
## Warning: package 'survival' was built under R version 3.3.3
```

```r
rfFit <- train(classe~ ., data = training, 
                 method = "rf", 
                 trControl = fitControl,
                 ntree=100)
```
## Model Accuracy

Compare the models accuracy by using testing datasets.

```r
pred_dt<-predict(dtreeFit,testing)
dt<-confusionMatrix(pred_dt,testing$classe)
pred_gbm<-predict(gbmFit,testing)
gbm<-confusionMatrix(pred_gbm,testing$classe)
pred_rf<-predict(rfFit,testing)
rf<-confusionMatrix(pred_rf,testing$classe)
results <- data.frame(
  Model = c('Decisiontrees', 'GBM', 'RF'),
  Accuracy = rbind(dt$overall[1], gbm$overall[1], rf$overall[1])
)
print(results)
```

```
##           Model  Accuracy
## 1 Decisiontrees 0.5359388
## 2           GBM 0.9911640
## 3            RF 0.9983008
```
Based on the assessements of the three models, the gradient boosting and random forests have the high accuracy. The following 5 features are most important from random forests.

```r
varImp(rfFit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 72)
## 
##                                Overall
## num_window                     100.000
## roll_belt                       67.766
## pitch_forearm                   41.702
## yaw_belt                        31.180
## magnet_dumbbell_z               30.598
## magnet_dumbbell_y               26.494
## pitch_belt                      25.088
## roll_forearm                    23.170
## cvtd_timestamp28/11/2011 14:13  17.860
## cvtd_timestamp30/11/2011 17:12  17.315
## magnet_dumbbell_x               17.219
## cvtd_timestamp02/12/2011 14:58  16.640
## cvtd_timestamp05/12/2011 14:24  14.866
## cvtd_timestamp28/11/2011 14:15  13.151
## cvtd_timestamp02/12/2011 13:33  12.508
## accel_belt_z                    11.385
## accel_dumbbell_y                10.663
## accel_forearm_x                  9.528
## roll_dumbbell                    9.464
## accel_dumbbell_z                 8.655
```



```r
predictTEST <- predict(rfFit, newdata=valid)
predictTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
