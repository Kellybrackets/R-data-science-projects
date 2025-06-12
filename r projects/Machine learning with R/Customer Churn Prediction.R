# ----------------------------
# Customer Churn Prediction
# ----------------------------

# Load libraries
library(tidyverse)
library(caret)
library(ROCR)

# Load dataset (simulated churn data)
data <- read.csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/telecom_churn.csv")

# Data preprocessing
data <- data %>%
  mutate(churn = as.factor(churn),
         total_charges = ifelse(is.na(total_charges), median(total_charges, na.rm = TRUE), total_charges))

# Train-test split
set.seed(123)
train_index <- createDataPartition(data$churn, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train models
models <- list(
  logistic = train(churn ~ ., data = train_data, method = "glm", family = "binomial"),
  random_forest = train(churn ~ ., data = train_data, method = "rf")
)

# Evaluate
results <- lapply(models, function(model) {
  pred <- predict(model, test_data)
  confusionMatrix(pred, test_data$churn)
})

# ROC curve
prob <- predict(models$logistic, test_data, type = "prob")[,2]
pred_roc <- prediction(prob, test_data$churn)
perf <- performance(pred_roc, "tpr", "fpr")
plot(perf, col = "blue", main = "ROC Curve")

# Feature importance
varImp(models$random_forest) %>% plot(main = "Random Forest Feature Importance")

# Save best model
saveRDS(models$random_forest, "churn_model.rds")