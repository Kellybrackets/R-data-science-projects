# ----------------------------
# House Price Prediction
# ----------------------------

library(tidyverse)
library(caret)
library(xgboost)

# Load Boston housing data
data("BostonHousing", package = "mlbench")
data <- BostonHousing

# Train-test split
set.seed(123)
train_index <- createDataPartition(data$medv, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train models
models <- list(
  linear = train(medv ~ ., data = train_data, method = "lm"),
  xgb = train(medv ~ ., data = train_data, method = "xgbTree")
)

# Evaluate
metrics <- map_dfr(models, function(model) {
  pred <- predict(model, test_data)
  postResample(pred, test_data$medv)
}, .id = "model")

print(metrics)

# Residual plot
pred <- predict(models$linear, test_data)
plot(test_data$medv, pred, main = "Actual vs Predicted Prices", xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = "red")

# Feature importance
ggplot(varImp(models$xgb)) + 
  ggtitle("XGBoost Feature Importance")

# Save model
saveRDS(models$xgb, "house_price_model.rds")