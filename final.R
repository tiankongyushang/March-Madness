#final

library(car) #package to calculate Variance Inflation Factor
library(leaps) #best subsets regression
library(glmnet) #allows ridge regression, LASSO and elastic net
library(caret) #this will help identify the appropriate parameters
require(readxl)
require(tidyverse)
require(gridExtra)
require(broom)
require(dplyr)
require(MLmetrics)
require(ggplot2)
require(reshape2)
require(GGally)

theme.info <- theme(plot.title = element_text(size=16, hjust=0.5),
                   axis.title = element_text(size=14),
                   axis.text = element_text(size=14))

data <- read_delim("merge_r_t.csv", col_names = TRUE, delim = ',')
test_lasso <- read_delim("test(merge_r_t).csv", col_names = TRUE, delim = ',')
#-----drop unused columns-------
data <- data[,-c(1,2,4,5,6,7,30:32)]
data <- data[,-c(1)]

test_lasso <- test_lasso[,-c(1:6,29:31)]

ggcorr(data, geom = "circle", nbreaks = 5)

#----------test VIF--------------
lm_vif <- glm(Outcome ~., family = "binomial", data = data)
vif(lm_vif)

data <- data[,-c(3,4,13,14,26,28,36,43,45,53)]
test_lasso <- test_lasso[,-c(3,4,13,14,26,28,36,43,45,53)]
outcome_true <- test_lasso[,c(47)]
test_lasso <- test_lasso[,-c(47)]
#----------lasso------------
x1 <- as.matrix(data[,1:46])
y1 <- as.matrix(data[,47])

#----交叉验证选取模型时希望最小化的目标参量 (最终还是选择deviance)
#------- type.measure = default (devaiance)----即 - 2 倍的 log-likelihood
lasso_dev <- cv.glmnet(x = x1, y = y1, family = "binomial")
plot(lasso_dev)
#variable being chosed by lasso
coef(lasso_dev, s = "lambda.min")
par(mfrow=c(2,1))
A <- plot(lasso_dev, xvar = "lambda", label = TRUE)#两条虚线分别指示了两个特殊的 λ 值
B <- plot(lasso_dev$glmnet.fit, xvar = "lambda", label = TRUE)

lasso_dev$lambda.min #lambda.min 是指在所有的 λ 值中，得到最小目标参量均值的那一个
lasso_dev$lambda.1se #指在 lambda.min 一个方差范围内得到最简单模型的那一个 λ 值
test_lasso <- as.matrix(test_lasso)
outcome_predict <- predict(lasso_dev, newx = test_lasso, type = "response", s = "lambda.1se")

#----type.measure = auc------使用area under the ROC curve -----变量太少不用
lasso_auc <- cv.glmnet(x = x1, y = y1, family = "binomial",type.measure = "auc", nfolds = 30)
coef(lasso_auc, s = "lambda.min")
plot(lasso_auc, xvar = "lambda", label = TRUE)#两条虚线分别指示了两个特殊的 λ 值
plot(lasso_auc$glmnet.fit, xvar = "lambda")
lasso_auc$lambda.min #lambda.min 是指在所有的 λ 值中，得到最小目标参量均值的那一个
lasso_auc$lambda.1se #指在 lambda.min 一个方差范围内得到最简单模型的那一个 λ 值
test_lasso <- as.matrix(test_lasso)
outcome_predict <- predict(lasso_auc, newx = test_lasso, type = "response", s = "lambda.min")

#-------type.measure = class-----使用模型分类的错误率（missclassification error）---变量太少不用
lasso_class <- cv.glmnet(x = x1, y = y1, family = "binomial",type.measure = "class")
coef(lasso_class, s = "lambda.min")
plot(lasso_class, xvar = "lambda", label = TRUE)#两条虚线分别指示了两个特殊的 λ 值
plot(lasso_class$glmnet.fit, xvar = "lambda")
lasso_class$lambda.min #lambda.min 是指在所有的 λ 值中，得到最小目标参量均值的那一个
lasso_class$lambda.1se #指在 lambda.min 一个方差范围内得到最简单模型的那一个 λ 值
outcome_predict <- predict(lasso_class, newx = test_lasso, type = "response", s = "lambda.1se")

#------type.measure="mae" 使用 mean absolute error-----
lasso_mae <- cv.glmnet(x = x1, y = y1, family = "binomial",type.measure="mae")
coef(lasso_mae, s = "lambda.min")
plot(lasso_mae, xvar = "lambda", label = TRUE)#两条虚线分别指示了两个特殊的 λ 值
plot(lasso_mae$glmnet.fit, xvar = "lambda")
lasso_mae$lambda.min #lambda.min 是指在所有的 λ 值中，得到最小目标参量均值的那一个
lasso_mae$lambda.1se #指在 lambda.min 一个方差范围内得到最简单模型的那一个 λ 值
predict(lasso_mae, newx = test_lasso, type = "response", s = "lambda.1se")
write.csv(outcome_true,'final123.csv')
write.csv(outcome_predict,'final.csv')
outcome_true <- as.matrix(outcome_true)
LogLoss(outcome_predict, outcome_true)

