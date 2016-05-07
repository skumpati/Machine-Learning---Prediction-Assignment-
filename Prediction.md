---
title: "Prediction Assignment Writeup"
author: "Suresh Babu Kumpati"
date: "May 7, 2016"
output: html_document
---
## Prediction Assignment


### Background

Using devices such as JawboneUp, NikeFuelBand, and Fitbitit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
   
In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).   

### Preparing the data and R packages  

#### Load packages, set caching 

```{r, message=FALSE}
library(caret)
library(corrplot)
library(Rtsne)
library(stats)
library(knitr)
library(ggplot2)
knitr::opts_chunk$set(cache=TRUE)
```
  

#### Getting Data

```{r}
# URL of the training and testing data
train.url ="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test.url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# file names
train.name = "./data/pml-training.csv"
test.name = "./data/pml-testing.csv"
# if directory does not exist, create new
if (!file.exists("./data")) {
  dir.create("./data")
}

# load the CSV files as data.frame 
train = read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test = read.csv("pml-testing.csv",  na.strings = c("NA", "#DIV/0!", ""))

dim(train)

## [1] 19622   160

dim(test)

## [1]  20 160

names(train)

```

The raw training data has 19622 rows of observations and 158 features (predictors). Column `X` is unusable row number. While the testing data has 20 rows and the same 158 features. There is one column of target outcome named `classe`.   

#### Data cleaning

First, extract target outcome (the activity quality) from training data, so now the training data contains only the predictors (the activity monitors).   
```{r, echo = TRUE}
# target outcome (label)
outcome.org = train[, "classe"]
outcome = outcome.org 
levels(outcome)
```

Outcome has 5 levels in character format.   
Convert the outcome to numeric, because XGBoost gradient booster only recognizes numeric data.   
```{r, echo = TRUE}
# convert character levels to numeric
num.class = length(levels(outcome))
levels(outcome) = 1:num.class
head(outcome)
```

The outcome is removed from training data.   
```{r, echo = TRUE}
# remove outcome from train
train$classe = NULL
```

The assignment rubric asks to use data from accelerometers on the `belt`, `forearm`, `arm`, and `dumbell`, so the features are extracted based on these keywords.   
  
```{r, echo = TRUE}
# filter columns on: belt, forearm, arm, dumbell
filter = grepl("belt|arm|dumbell", names(train))
train = train[, filter]
test = test[, filter]
```

Instead of less-accurate imputation of missing data, remove all columns with NA values.   
```{r}
# remove columns with NA, use test data as referal for NA
cols.without.na = colSums(is.na(test)) == 0
train = train[, cols.without.na]
test = test[, cols.without.na]
```

### Preprocessing  

#### Check for features's variance

Based on the principal component analysis PCA, it is important that features have maximum variance for maximum uniqueness, so that each feature is as distant as possible (as orthogonal as possible) from the other features.   
```{r, echo= TRUE}
# check for zero variance
zero.var = nearZeroVar(train, saveMetrics=TRUE)
zero.var

```
There is no features without variability (all has enough variance). So there is no feature to be removed further.  

#### Plot of relationship between features and outcome  

Plot the relationship between features and outcome. From the plot below, each features has relatively the same distribution among the 5 outcome levels (A, B, C, D, E).   
```{r fig.width=12, fig.height=8, dpi=72}
featurePlot(train, outcome.org, "strip")
```

(https://github.com/skumpati/Machine-Learning---Prediction-Assignment-/blob/master/Figures/unnamed-chunk-9-1.png)

#### Plot of correlation matrix  

Plot a correlation matrix between features.   
A good set of features is when they are highly uncorrelated (orthogonal) each others. The plot below shows average of correlation is not too high, so I choose to not perform further PCA preprocessing.   
```{r fig.width=12, fig.height=12, dpi=72}
corrplot.mixed(cor(train), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
```

(https://github.com/skumpati/Machine-Learning---Prediction-Assignment-/blob/master/Figures/unnamed-chunk-10-1.png)

#### tSNE plot 

A tSNE (t-Distributed Stochastic Neighbor Embedding) visualization is 2D plot of multidimensional features, that is multidimensional reduction into 2D plane. In the tSNE plot below there is no clear separation of clustering of the 5 levels of outcome (A, B, C, D, E). So it hardly gets conclusion for manually building any regression equation from the irregularity.   

```{r fig.width=12, fig.height=8, dpi=72}
# t-Distributed Stochastic Neighbor Embedding
tsne = Rtsne(as.matrix(train), check_duplicates=FALSE, pca=TRUE, 
              perplexity=30, theta=0.5, dims=2)

## Read the 19622 x 39 data matrix successfully!
## Using no_dims = 2, perplexity = 30.000000, and theta = 0.500000
## Computing input similarities...
## Building tree...
##  - point 0 of 19622
##  - point 10000 of 19622
## Done in 5.90 seconds (sparsity = 0.005774)!
## Learning embedding...
## Iteration 50: error is 106.633840 (50 iterations in 9.74 seconds)
## Iteration 100: error is 97.718591 (50 iterations in 10.70 seconds)
## Iteration 150: error is 82.962726 (50 iterations in 7.74 seconds)
## Iteration 200: error is 78.169002 (50 iterations in 7.45 seconds)
## Iteration 250: error is 3.803975 (50 iterations in 7.43 seconds)
## Iteration 300: error is 3.086925 (50 iterations in 7.28 seconds)
## Iteration 350: error is 2.675746 (50 iterations in 7.20 seconds)
## Iteration 400: error is 2.385472 (50 iterations in 7.20 seconds)
## Iteration 450: error is 2.168501 (50 iterations in 7.13 seconds)
## Iteration 500: error is 2.000504 (50 iterations in 7.15 seconds)
## Iteration 550: error is 1.866260 (50 iterations in 7.15 seconds)
## Iteration 600: error is 1.755478 (50 iterations in 7.16 seconds)
## Iteration 650: error is 1.662327 (50 iterations in 7.18 seconds)
## Iteration 700: error is 1.583451 (50 iterations in 7.21 seconds)
## Iteration 750: error is 1.515918 (50 iterations in 7.22 seconds)
## Iteration 800: error is 1.458107 (50 iterations in 7.28 seconds)
## Iteration 850: error is 1.407774 (50 iterations in 7.28 seconds)
## Iteration 900: error is 1.363542 (50 iterations in 7.31 seconds)
## Iteration 950: error is 1.324365 (50 iterations in 7.39 seconds)
## Iteration 999: error is 1.290877 (50 iterations in 7.23 seconds)
## Fitting performed in 151.42 seconds.


embedding = as.data.frame(tsne$Y)
embedding$Class = outcome.org
g = ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE 2D Embedding of 'Classe' Outcome") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank())
print(g)
```

(https://github.com/skumpati/Machine-Learning---Prediction-Assignment-/blob/master/Figures/unnamed-chunk-11-1.png)


### Build machine learning model 

Now build a machine learning model to predict activity quality (`classe` outcome) from the activity monitors (the features or predictors) by using XGBoost extreme gradient boosting algorithm.    

#### Expected error rate 

Expected error rate is less than `1%` for a good classification. Do cross validation to estimate the error rate using 4-fold cross validation, with 200 epochs to reach the expected error rate of less than `1%`.  

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    1    2    3    4    5
##          1 5567    8    5    0    0
##          2   12 3770   15    0    0
##          3    0   24 3382   16    0
##          4    2    0   23 3190    1
##          5    0    2    3   10 3592
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9938          
##                  95% CI : (0.9926, 0.9949)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9922          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
## Sensitivity            0.9975   0.9911   0.9866   0.9919   0.9997
## Specificity            0.9991   0.9983   0.9975   0.9984   0.9991
## Pos Pred Value         0.9977   0.9929   0.9883   0.9919   0.9958
## Neg Pred Value         0.9990   0.9979   0.9972   0.9984   0.9999
## Prevalence             0.2844   0.1939   0.1747   0.1639   0.1831
## Detection Rate         0.2837   0.1921   0.1724   0.1626   0.1831
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9983   0.9947   0.9921   0.9952   0.9994
```

Confusion matrix shows concentration of correct predictions is on the diagonal, as expected.  
  
The average accuracy is `99.38%`, with error rate is `0.62%`. So, expected error rate of less than `1%` is fulfilled.  

Time elapsed is around 35 seconds.  

#### Post-processing

Output of prediction is the predicted probability of the 5 levels (columns) of outcome.  
Decode the quantitative 5 levels of outcomes to qualitative letters (A, B, C, D, E).   
  
# plot
![alt tag](https://github.com/skumpati/Machine-Learning---Prediction-Assignment-/blob/master/Figures/unnamed-chunk-21-1.png)



Feature importance plot is useful to select only best features with highest correlation to the outcome(s). To improve model fitting performance (time or overfitting), less important features can be removed.   

### Creating submission files 

```{r}
path = "./answer"
pml_write_files = function(x) {
    n = length(x)
    for(i in 1: n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file=file.path(path, filename), 
                    quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}
```
------------------   

