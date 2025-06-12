# ----------------------------
# Fraud Detection
# ----------------------------

library(tidyverse)
library(caret)
library(DMwR)  # For SMOTE
library(ROSE)  # For ROSE sampling

# Load dataset (simplified example)
data <- read.csv("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")

# Highly imbalanced data (fraud = 0.17%)
table(data$Class)

# Undersampling majority class
set.seed(123)
data_balanced <- ovun.sample(Class ~ ., data = data, method = "under")$data

# Train-test split
train_index <- createDataPartition(data_balanced$Class, p = 0.8, list = FALSE)
train_data <- data_balanced[train_index, ]
test_data <- data_balanced[-train_index, ]

# Train models
models <- list(
  logistic = train(Class ~ ., data = train_data, method = "glm", family = "binomial"),
  isolation_forest = train(Class ~ ., data = train_data, method = "iforest")
)

# Evaluate
results <- map(models, ~confusionMatrix(predict(.x, test_data), test_data$Class))
print(results)

# Precision-Recall curve
prob <- predict(models$logistic, test_data, type = "prob")[,2]
pr <- pr.curve(scores.class0 = prob, weights.class0 = as.numeric(test_data$Class) - 1)
plot(pr, main = "Precision-Recall Curve")

# Save model
saveRDS(models$isolation_forest, "fraud_model.rds")