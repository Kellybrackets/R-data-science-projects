#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Iris Classification with SVM and Comparative Machine Learning
# Features:
# - Comprehensive data exploration
# - Advanced feature engineering
# - Multiple model comparison
# - Hyperparameter tuning
# - Detailed performance metrics
# - Explainable AI components
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## 1. SETUP AND DATA LOADING ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load required libraries
library(datasets)
library(caret)
library(ggplot2)
library(GGally)
library(pROC)
library(e1071)
library(kernlab)
library(gridExtra)
library(dplyr)
library(recipes)

# Set random seed for reproducibility
set.seed(100)

# Load Iris dataset
data(iris)

## 2. COMPREHENSIVE DATA EXPLORATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 2.1 Basic Data Inspection
cat("Dataset Structure:\n")
str(iris)

cat("\nSummary Statistics:\n")
summary(iris)

cat("\nClass Distribution:\n")
table(iris$Species) %>% 
  prop.table() %>% 
  round(3) %>% 
  kable() %>% 
  kable_styling(bootstrap_options = "striped")

# 2.2 Data Quality Check
cat("\nData Quality Assessment:\n")
missing_values <- sum(is.na(iris))
cat("Missing Values:", missing_values, "\n")

duplicates <- sum(duplicated(iris))
cat("Duplicate Rows:", duplicates, "\n")
if(duplicates > 0) {
  iris <- distinct(iris)
  cat("Duplicates removed.\n")
}

# 2.3 Advanced Visualization
cat("\nCreating Visualizations...\n")

# Feature distribution by class
feature_dist <- lapply(names(iris)[1:4], function(x) {
  ggplot(iris, aes_string(x = x, fill = "Species")) +
    geom_density(alpha = 0.6) +
    theme_minimal() +
    labs(title = paste("Distribution of", x))
})
grid.arrange(grobs = feature_dist, ncol = 2)

# Scatterplot matrix
ggpairs(iris, columns = 1:4, aes(color = Species, alpha = 0.7)) +
  theme_bw() +
  ggtitle("Feature Relationships by Species")

# Correlation heatmap
cor_matrix <- cor(iris[,1:4])
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")

## 3. DATA PREPARATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~

# 3.1 Train-Test Split
cat("\nCreating Train-Test Split (80-20)...\n")
TrainingIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
TrainingSet <- iris[TrainingIndex, ]
TestingSet <- iris[-TrainingIndex, ]

# Verify split proportions
cat("\nTraining Set Dimensions:", dim(TrainingSet), "\n")
cat("Testing Set Dimensions:", dim(TestingSet), "\n")

cat("\nClass Distribution in Training Set:\n")
table(TrainingSet$Species) %>% 
  prop.table() %>% 
  kable() %>% 
  kable_styling()

cat("\nClass Distribution in Testing Set:\n")
table(TestingSet$Species) %>% 
  prop.table() %>% 
  kable() %>% 
  kable_styling()

# 3.2 Feature Engineering
cat("\nApplying Feature Engineering...\n")

# Create recipe for preprocessing
iris_recipe <- recipe(Species ~ ., data = TrainingSet) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_corr(all_numeric(), threshold = 0.9) %>%
  step_BoxCox(all_numeric()) %>%
  prep()

# Apply transformations
TrainingSet <- bake(iris_recipe, new_data = TrainingSet)
TestingSet <- bake(iris_recipe, new_data = TestingSet)

## 4. MODEL BUILDING AND EVALUATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 4.1 Define Training Control
cat("\nSetting Up Model Training...\n")
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     classProbs = TRUE,
                     summaryFunction = multiClassSummary,
                     savePredictions = TRUE)

# 4.2 SVM with Polynomial Kernel (Enhanced)
cat("\nTraining SVM with Polynomial Kernel...\n")

# Expanded tuning grid
svmPolyGrid <- expand.grid(
  degree = c(1, 2, 3),
  scale = c(0.1, 1, 10),
  C = c(0.1, 1, 10)
)

# Train model with tuning
svmPolyModel <- train(Species ~ ., 
                      data = TrainingSet,
                      method = "svmPoly",
                      trControl = ctrl,
                      tuneGrid = svmPolyGrid,
                      metric = "Accuracy")

# Display tuning results
cat("\nSVM (Poly) Tuning Results:\n")
print(svmPolyModel)
plot(svmPolyModel, main = "SVM Polynomial Kernel Tuning")

# 4.3 Comparative Models
cat("\nTraining Comparative Models...\n")

# Random Forest
rfModel <- train(Species ~ .,
                 data = TrainingSet,
                 method = "rf",
                 trControl = ctrl,
                 importance = TRUE,
                 tuneLength = 5)

# Logistic Regression (Multinomial)
logitModel <- train(Species ~ .,
                    data = TrainingSet,
                    method = "multinom",
                    trControl = ctrl,
                    trace = FALSE)

# 4.4 Model Evaluation
cat("\nEvaluating Model Performance...\n")

# Collect all models
models <- list(
  "SVM_Poly" = svmPolyModel,
  "Random_Forest" = rfModel,
  "Logistic_Regression" = logitModel
)

# Resample comparison
resamps <- resamples(models)
summary(resamps)

# Visual comparison
dotplot(resamps, metric = "Accuracy", main = "Model Accuracy Comparison")
bwplot(resamps, metric = "Accuracy", main = "Model Accuracy Distribution")

# 4.5 Test Set Evaluation
evaluate_model <- function(model, data) {
  predictions <- predict(model, data)
  probs <- predict(model, data, type = "prob")
  cm <- confusionMatrix(predictions, data$Species)
  
  # Multiclass ROC
  roc <- multiclass.roc(data$Species, probs)
  
  return(list(
    predictions = predictions,
    probabilities = probs,
    confusion = cm,
    roc = roc
  ))
}

test_results <- lapply(models, evaluate_model, data = TestingSet)

# Print test results
for(model_name in names(test_results)) {
  cat("\n", model_name, "Performance:\n")
  print(test_results[[model_name]]$confusion)
  cat("AUC:", auc(test_results[[model_name]]$roc), "\n")
}

## 5. MODEL INTERPRETATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 5.1 Feature Importance
cat("\nAnalyzing Feature Importance...\n")

# SVM Feature Importance
svm_imp <- varImp(svmPolyModel, scale = FALSE)
plot(svm_imp, main = "SVM Feature Importance")

# Random Forest Feature Importance
rf_imp <- varImp(rfModel, scale = FALSE)
plot(rf_imp, main = "Random Forest Feature Importance")

# 5.2 Partial Dependence Plots
cat("\nGenerating Partial Dependence Plots...\n")
partial_dep <- lapply(names(iris)[1:4], function(feature) {
  plot_partial_dependence <- partial(rfModel, pred.var = feature, which.class = "versicolor")
  autoplot(plot_partial_dependence) + 
    theme_minimal() +
    labs(title = paste("Partial Dependence on", feature))
})
grid.arrange(grobs = partial_dep, ncol = 2)

## 6. MODEL DEPLOYMENT PREPARATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 6.1 Select Best Model
test_accuracies <- sapply(test_results, function(x) x$confusion$overall['Accuracy'])
best_model_name <- names(which.max(test_accuracies))
best_model <- models[[best_model_name]]

cat("\nBest Performing Model:", best_model_name, "\n")
cat("Test Accuracy:", max(test_accuracies), "\n")

# 6.2 Save Model Pipeline
final_model <- list(
  model = best_model,
  recipe = iris_recipe,
  performance = test_results[[best_model_name]]$confusion
)

saveRDS(final_model, file = "iris_classification_model.rds")

# 6.3 Prediction Function
predict_iris <- function(new_data, model_obj) {
  # Preprocess new data
  preprocessed_data <- bake(model_obj$recipe, new_data = new_data)
  
  # Make predictions
  predictions <- predict(model_obj$model, preprocessed_data)
  
  # Add probabilities if needed
  if(model_obj$model$modelType == "Classification") {
    probs <- predict(model_obj$model, preprocessed_data, type = "prob")
    return(data.frame(Class = predictions, probs))
  } else {
    return(predictions)
  }
}

# Example usage
new_samples <- data.frame(
  Sepal.Length = c(5.1, 6.7),
  Sepal.Width = c(3.5, 3.0),
  Petal.Length = c(1.4, 5.2),
  Petal.Width = c(0.2, 2.3)
)

cat("\nExample Predictions:\n")
predict_iris(new_samples, final_model) %>% 
  kable() %>% 
  kable_styling()

## 7. FINAL REPORT ##
#~~~~~~~~~~~~~~~~~~~~

# Generate performance report
generate_report <- function(results) {
  cat("\nFINAL MODEL PERFORMANCE REPORT\n")
  cat("==============================\n\n")
  
  metrics <- data.frame()
  for(model_name in names(results)) {
    cm <- results[[model_name]]$confusion
    metrics <- rbind(metrics, data.frame(
      Model = model_name,
      Accuracy = cm$overall['Accuracy'],
      Kappa = cm$overall['Kappa'],
      AUC = auc(results[[model_name]]$roc),
      stringsAsFactors = FALSE
    ))
  }
  
  print(kable(metrics) %>% kable_styling())
  
  cat("\nBest Model:", best_model_name, "\n")
  cat("Best Test Accuracy:", round(max(test_accuracies), 4), "\n")
  cat("\nConfusion Matrix for Best Model:\n")
  print(test_results[[best_model_name]]$confusion$table)
}

generate_report(test_results)