#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Iris Classification Project
# Features:
# - Comprehensive EDA
# - Multiple modeling approaches
# - Hyperparameter tuning
# - Detailed performance evaluation
# - Advanced visualizations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load required libraries
library(caret)
library(ggplot2)
library(GGally)
library(e1071)
library(pROC)
library(kernlab)
library(gridExtra)
library(dplyr)
library(corrplot)

## 1. DATA LOADING AND INITIAL INSPECTION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load Iris dataset
data(iris)

# Basic dataset inspection
cat("Dataset Structure:\n")
str(iris)

cat("\nDataset Summary:\n")
summary(iris)

cat("\nClass Distribution:\n")
table(iris$Species)

## 2. DATA QUALITY CHECK ##
#~~~~~~~~~~~~~~~~~~~~~~~~~

# Check for missing values
missing_data <- sum(is.na(iris))
cat("\nMissing Values Summary:\n")
if (missing_data == 0) {
  cat("No missing values found in the dataset.\n")
} else {
  cat("Total missing values:", missing_data, "\n")
  print(colSums(is.na(iris)))
}

# Check for duplicated rows
duplicates <- sum(duplicated(iris))
cat("\nDuplicate Rows:", duplicates, "\n")
if (duplicates > 0) {
  cat("Duplicate rows found. Consider removing them.\n")
  iris <- distinct(iris)
}

## 3. EXPLORATORY DATA ANALYSIS (EDA) ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3.1 Univariate Analysis
cat("\nUnivariate Analysis:\n")

# Create histograms for each feature
hist_plots <- list()
for (col in names(iris)[1:4]) {
  hist_plots[[col]] <- ggplot(iris, aes_string(x = col)) +
    geom_histogram(aes(fill = Species), bins = 30, alpha = 0.7) +
    labs(title = paste("Distribution of", col)) +
    theme_minimal()
}
grid.arrange(grobs = hist_plots, ncol = 2)

# 3.2 Bivariate Analysis
cat("\nBivariate Analysis:\n")

# Scatter plot matrix with species differentiation
ggpairs(iris, columns = 1:4, aes(color = Species, alpha = 0.7)) +
  theme_bw()

# Boxplots for each feature by species
box_plots <- list()
for (col in names(iris)[1:4]) {
  box_plots[[col]] <- ggplot(iris, aes_string(x = "Species", y = col)) +
    geom_boxplot(aes(fill = Species)) +
    labs(title = paste(col, "by Species")) +
    theme_minimal()
}
grid.arrange(grobs = box_plots, ncol = 2)

# 3.3 Correlation Analysis
cat("\nCorrelation Analysis:\n")

# Calculate correlations
cor_matrix <- cor(iris[,1:4])
print(cor_matrix)

# Visualize correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")

## 4. DATA PREPARATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~

# Set seed for reproducibility
set.seed(100)

# Stratified train-test split (80-20)
TrainingIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
TrainingSet <- iris[TrainingIndex, ]
TestingSet <- iris[-TrainingIndex, ]

# Verify split proportions
cat("\nTraining Set Dimensions:", dim(TrainingSet), "\n")
cat("Testing Set Dimensions:", dim(TestingSet), "\n")

cat("\nClass Distribution in Training Set:\n")
print(prop.table(table(TrainingSet$Species)))

cat("\nClass Distribution in Testing Set:\n")
print(prop.table(table(TestingSet$Species)))

# Compare summary statistics
cat("\nTraining Set Summary:\n")
print(summary(TrainingSet))

cat("\nTesting Set Summary:\n")
print(summary(TestingSet))

## 5. MODEL BUILDING ##
#~~~~~~~~~~~~~~~~~~~~~~

# Define common training control
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     classProbs = TRUE,
                     summaryFunction = multiClassSummary,
                     savePredictions = TRUE)

# 5.1 SVM with Polynomial Kernel (with hyperparameter tuning)
cat("\nBuilding SVM with Polynomial Kernel...\n")

# Define tuning grid
svmPolyGrid <- expand.grid(degree = c(1, 2, 3),
                           scale = c(0.1, 1, 10),
                           C = c(0.1, 1, 10))

# Train model
svmPolyModel <- train(Species ~ ., 
                      data = TrainingSet,
                      method = "svmPoly",
                      trControl = ctrl,
                      tuneGrid = svmPolyGrid,
                      preProcess = c("scale", "center"),
                      metric = "Accuracy")

# Print model details
cat("\nSVM (Poly) Model Summary:\n")
print(svmPolyModel)
plot(svmPolyModel)

# 5.2 SVM with Radial Kernel (for comparison)
cat("\nBuilding SVM with Radial Kernel...\n")

svmRadialGrid <- expand.grid(sigma = c(0.1, 1, 10),
                             C = c(0.1, 1, 10))

svmRadialModel <- train(Species ~ .,
                        data = TrainingSet,
                        method = "svmRadial",
                        trControl = ctrl,
                        tuneGrid = svmRadialGrid,
                        preProcess = c("scale", "center"),
                        metric = "Accuracy")

cat("\nSVM (Radial) Model Summary:\n")
print(svmRadialModel)
plot(svmRadialModel)

# 5.3 Random Forest (as alternative model)
cat("\nBuilding Random Forest Model...\n")

rfModel <- train(Species ~ .,
                 data = TrainingSet,
                 method = "rf",
                 trControl = ctrl,
                 tuneLength = 5,
                 importance = TRUE,
                 metric = "Accuracy")

cat("\nRandom Forest Model Summary:\n")
print(rfModel)
plot(rfModel)

## 6. MODEL EVALUATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~

# 6.1 Training Set Performance
cat("\nTraining Set Performance:\n")

models <- list("SVM_Poly" = svmPolyModel,
               "SVM_Radial" = svmRadialModel,
               "Random_Forest" = rfModel)

# Collect resamples for comparison
resamps <- resamples(models)
summary(resamps)

# Visualize model comparison
dotplot(resamps, metric = "Accuracy")
bwplot(resamps, metric = "Accuracy")

# 6.2 Test Set Evaluation
cat("\nTest Set Evaluation:\n")

# Function to evaluate models on test set
evaluate_model <- function(model, data) {
  predictions <- predict(model, data)
  cm <- confusionMatrix(predictions, data$Species)
  return(list(predictions = predictions, confusion = cm))
}

# Evaluate all models
test_results <- lapply(models, evaluate_model, data = TestingSet)

# Print confusion matrices
for (model_name in names(test_results)) {
  cat("\n", model_name, "Confusion Matrix:\n")
  print(test_results[[model_name]]$confusion)
}

# 6.3 ROC Analysis (for multiclass)
cat("\nROC Analysis:\n")

# Calculate probabilities for each model
prob_poly <- predict(svmPolyModel, TestingSet, type = "prob")
prob_radial <- predict(svmRadialModel, TestingSet, type = "prob")
prob_rf <- predict(rfModel, TestingSet, type = "prob")

# Create ROC curves
roc_poly <- multiclass.roc(TestingSet$Species, prob_poly)
roc_radial <- multiclass.roc(TestingSet$Species, prob_radial)
roc_rf <- multiclass.roc(TestingSet$Species, prob_rf)

# Print AUC values
cat("\nSVM (Poly) AUC:", auc(roc_poly), "\n")
cat("SVM (Radial) AUC:", auc(roc_radial), "\n")
cat("Random Forest AUC:", auc(roc_rf), "\n")

## 7. FEATURE IMPORTANCE ##
#~~~~~~~~~~~~~~~~~~~~~~~~~

cat("\nFeature Importance Analysis:\n")

# 7.1 SVM Feature Importance
svm_imp <- varImp(svmPolyModel, scale = FALSE)
plot(svm_imp, main = "SVM (Poly) - Feature Importance")

# 7.2 Random Forest Feature Importance
rf_imp <- varImp(rfModel, scale = FALSE)
plot(rf_imp, main = "Random Forest - Feature Importance")

# Compare feature importance
imp_plots <- list(
  ggplot(svm_imp$importance, aes(x = rownames(svm_imp$importance), 
                                 y = Overall)) +
    geom_col(fill = "steelblue") +
    labs(title = "SVM Feature Importance", x = "Features", y = "Importance") +
    theme_minimal(),
  
  ggplot(rf_imp$importance, aes(x = rownames(rf_imp$importance), 
                                y = Overall)) +
    geom_col(fill = "darkgreen") +
    labs(title = "RF Feature Importance", x = "Features", y = "Importance") +
    theme_minimal()
)

grid.arrange(grobs = imp_plots, ncol = 2)

## 8. MODEL DEPLOYMENT PREPARATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Select best model based on test accuracy
test_accuracies <- sapply(test_results, function(x) x$confusion$overall['Accuracy'])
best_model_name <- names(which.max(test_accuracies))
best_model <- models[[best_model_name]]

cat("\nBest Model:", best_model_name, "\n")
cat("Test Accuracy:", max(test_accuracies), "\n")

# Save the best model
saveRDS(best_model, file = "best_iris_classifier.rds")

# Function to classify new samples
classify_iris <- function(model, new_data) {
  if (ncol(new_data) != 4) {
    stop("Input data must have exactly 4 features")
  }
  colnames(new_data) <- names(iris)[1:4]
  predict(model, new_data)
}

# Example usage:
new_samples <- data.frame(
  Sepal.Length = c(5.1, 6.7),
  Sepal.Width = c(3.5, 3.0),
  Petal.Length = c(1.4, 5.2),
  Petal.Width = c(0.2, 2.3)
)

cat("\nClassifying New Samples:\n")
print(new_samples)
cat("\nPredictions:\n")
print(classify_iris(best_model, new_samples))

## 9. FINAL REPORT ##
#~~~~~~~~~~~~~~~~~~~~

# Generate a comprehensive performance report
generate_report <- function(results) {
  cat("\nFINAL MODEL PERFORMANCE REPORT\n")
  cat("==============================\n\n")
  
  for (model_name in names(results)) {
    cm <- results[[model_name]]$confusion
    cat("Model:", model_name, "\n")
    cat("Accuracy:", cm$overall['Accuracy'], "\n")
    cat("Kappa:", cm$overall['Kappa'], "\n")
    cat("Precision:\n")
    print(cm$byClass[,'Precision'])
    cat("\nRecall:\n")
    print(cm$byClass[,'Recall'])
    cat("\nF1 Score:\n")
    print(cm$byClass[,'F1'])
    cat("\n--------------------------------\n")
  }
  
  cat("\nBest Model:", best_model_name, "\n")
  cat("Best Test Accuracy:", max(test_accuracies), "\n")
}

generate_report(test_results)