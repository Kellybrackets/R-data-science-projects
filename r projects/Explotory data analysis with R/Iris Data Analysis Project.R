#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Iris Data Analysis Project
# Features:
# - Multiple data loading methods with validation
# - Comprehensive data quality checks
# - Advanced statistical analysis
# - Interactive visualizations
# - Machine learning model comparison
# - Detailed reporting
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## 1. SETUP AND CONFIGURATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Install required packages (if not already installed)
required_packages <- c("datasets", "RCurl", "skimr", "dplyr", "ggplot2", 
                       "caret", "GGally", "plotly", "DataExplorer", "psych", 
                       "corrplot", "gridExtra", "knitr", "kableExtra")

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load libraries
library(datasets)
library(RCurl)
library(skimr)
library(dplyr)
library(ggplot2)
library(caret)
library(GGally)
library(plotly)
library(DataExplorer)
library(psych)
library(corrplot)
library(gridExtra)
library(knitr)
library(kableExtra)

# Set global options
options(scipen = 999)  # Disable scientific notation
set.seed(123)          # For reproducibility

## 2. DATA LOADING AND VALIDATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Method 1: Built-in dataset
data(iris)
iris_builtin <- iris

# Method 2: Using datasets package
iris_direct <- datasets::iris

# Method 3: From GitHub (with error handling)
iris_github <- tryCatch({
  read.csv(text = getURL("https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv"))
}, error = function(e) {
  message("Failed to load from GitHub: ", e$message)
  NULL
})

# Data validation
validate_datasets <- function(df1, df2, df3) {
  cat("\nDataset Validation Results:\n")
  
  # Check if all datasets loaded successfully
  if(is.null(df3)) {
    cat("GitHub dataset failed to load. Proceeding with built-in datasets.\n")
    datasets <- list(df1, df2)
    names <- c("Built-in", "Direct")
  } else {
    datasets <- list(df1, df2, df3)
    names <- c("Built-in", "Direct", "GitHub")
  }
  
  # Compare datasets
  identical_flag <- TRUE
  for(i in 1:(length(datasets)-1)) {
    if(!identical(datasets[[i]], datasets[[i+1]])) {
      cat("\nWARNING: Datasets", names[i], "and", names[i+1], "are not identical!\n")
      identical_flag <- FALSE
      
      # Find differences
      diff_cols <- setdiff(names(datasets[[i]]), names(datasets[[i+1]]))
      if(length(diff_cols) > 0) {
        cat("- Different columns:", paste(diff_cols, collapse = ", "), "\n")
      }
    }
  }
  
  if(identical_flag) {
    cat("\nAll datasets are identical.\n")
  }
  
  # Return the first valid dataset
  return(datasets[[1]])
}

# Validate and select dataset
iris <- validate_datasets(iris_builtin, iris_direct, iris_github)

# Clean up
rm(iris_builtin, iris_direct, iris_github)

## 3. DATA EXPLORATION AND QUALITY CHECKS ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3.1 Initial Inspection
cat("\nDataset Structure:\n")
str(iris)

cat("\nFirst 5 rows:\n")
kable(head(iris, 5)) %>% kable_styling(bootstrap_options = "striped", full_width = F)

cat("\nLast 5 rows:\n")
kable(tail(iris, 5)) %>% kable_styling(bootstrap_options = "striped", full_width = F)

# 3.2 Data Quality Checks
cat("\nData Quality Checks:\n")

# Missing values
missing_summary <- data.frame(
  Feature = names(iris),
  Missing_Values = sapply(iris, function(x) sum(is.na(x))),
  Percentage = sapply(iris, function(x) round(mean(is.na(x)) * 100, 2))
)

kable(missing_summary) %>% kable_styling()

# Duplicates
duplicates <- sum(duplicated(iris))
cat("\nDuplicate Rows:", duplicates, "\n")
if(duplicates > 0) {
  cat("Removing duplicates...\n")
  iris <- distinct(iris)
}

# Outlier detection
outlier_check <- function(x) {
  qnt <- quantile(x, probs = c(0.25, 0.75))
  iqr <- IQR(x)
  lower <- qnt[1] - 1.5 * iqr
  upper <- qnt[2] + 1.5 * iqr
  sum(x < lower | x > upper)
}

outliers <- sapply(iris[,1:4], outlier_check)
cat("\nPotential Outliers:\n")
print(outliers)

## 4. STATISTICAL ANALYSIS ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 4.1 Summary Statistics
cat("\nSummary Statistics:\n")
skim(iris) %>% kable() %>% kable_styling()

# Grouped statistics
cat("\nSummary Statistics by Species:\n")
iris %>% 
  group_by(Species) %>% 
  skim() %>% 
  kable() %>% 
  kable_styling()

# Detailed descriptive stats
cat("\nDetailed Descriptive Statistics:\n")
describeBy(iris[,1:4], group = iris$Species) %>% 
  lapply(function(x) kable(x, digits = 2) %>% kable_styling()) %>% 
  print()

## 5. DATA VISUALIZATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~

# 5.1 Univariate Analysis
cat("\nUnivariate Analysis:\n")

# Histograms
hist_plots <- lapply(names(iris)[1:4], function(col) {
  ggplot(iris, aes_string(x = col)) +
    geom_histogram(aes(fill = Species), bins = 30, alpha = 0.7) +
    geom_density(aes(y = ..count..), color = "darkblue", size = 1) +
    labs(title = paste("Distribution of", col)) +
    theme_minimal()
})

grid.arrange(grobs = hist_plots, ncol = 2)

# 5.2 Bivariate Analysis
cat("\nBivariate Analysis:\n")

# Scatterplot matrix
ggpairs(iris, columns = 1:4, aes(color = Species, alpha = 0.7)) +
  theme_bw() +
  ggtitle("Scatterplot Matrix by Species")

# Interactive 3D scatterplot
plot_ly(iris, x = ~Sepal.Length, y = ~Sepal.Width, z = ~Petal.Length,
        color = ~Species, colors = c("#BF382A", "#1C91C0", "#0C4B8E")) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Sepal Length'),
                      yaxis = list(title = 'Sepal Width'),
                      zaxis = list(title = 'Petal Length')))

# 5.3 Multivariate Analysis
cat("\nMultivariate Analysis:\n")

# Correlation matrix
cor_matrix <- cor(iris[,1:4])
corrplot(cor_matrix, method = "number", type = "upper", 
         tl.col = "black", tl.srt = 45)

# Parallel coordinates plot
ggplot(iris, aes(x = variable, y = value, color = Species)) +
  geom_line(aes(group = id), alpha = 0.3) +
  geom_point(alpha = 0.5) +
  facet_grid(. ~ Species) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## 6. AUTOMATED EXPLORATORY ANALYSIS ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Generate comprehensive EDA report
DataExplorer::create_report(iris, 
                            output_file = "iris_EDA_report.html",
                            output_dir = getwd(),
                            y = "Species")

## 7. ADVANCED ANALYSIS ##
#~~~~~~~~~~~~~~~~~~~~~~~~~

# 7.1 Principal Component Analysis
pca_result <- prcomp(iris[,1:4], scale. = TRUE)
summary(pca_result)

# PCA biplot
ggbiplot(pca_result, groups = iris$Species, ellipse = TRUE) +
  theme_minimal() +
  ggtitle("PCA Biplot of Iris Dataset")

# 7.2 Cluster Analysis
# K-means clustering
kmeans_result <- kmeans(scale(iris[,1:4]), centers = 3, nstart = 20)
table(Actual = iris$Species, Predicted = kmeans_result$cluster)

# Visualize clusters
ggplot(iris, aes(x = Sepal.Length, y = Petal.Length, color = as.factor(kmeans_result$cluster))) +
  geom_point(size = 3) +
  geom_point(aes(shape = Species), size = 4) +
  ggtitle("K-means Clustering vs Actual Species") +
  theme_minimal()

## 8. MACHINE LEARNING MODELING ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 8.1 Data Preparation
# Create train-test split (80-20)
train_index <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

# 8.2 Model Training
# Define training control
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     classProbs = TRUE,
                     summaryFunction = multiClassSummary)

# Train multiple models
models <- list()

# Logistic Regression (multinomial)
models$logit <- train(Species ~ ., data = train_data,
                      method = "multinom",
                      trControl = ctrl,
                      trace = FALSE)

# Random Forest
models$rf <- train(Species ~ ., data = train_data,
                   method = "rf",
                   trControl = ctrl,
                   importance = TRUE)

# Support Vector Machine
models$svm <- train(Species ~ ., data = train_data,
                    method = "svmRadial",
                    trControl = ctrl,
                    tuneLength = 5)

# 8.3 Model Evaluation
# Compare model performance
results <- resamples(models)
summary(results)

# Visual comparison
dotplot(results, metric = "Accuracy")
bwplot(results, metric = "Accuracy")

# Test set evaluation
test_results <- lapply(models, function(model) {
  pred <- predict(model, test_data)
  cm <- confusionMatrix(pred, test_data$Species)
  return(list(predictions = pred, confusion = cm))
})

# Print test results
for(model_name in names(test_results)) {
  cat("\nModel:", model_name, "\n")
  print(test_results[[model_name]]$confusion)
}

## 9. REPORT GENERATION ##
#~~~~~~~~~~~~~~~~~~~~~~~~~

# Generate markdown report
cat("\nGenerating final report...\n")
render("iris_analysis_report.Rmd", output_file = "Iris_Analysis_Report.html")

cat("\nAnalysis complete! Report saved as Iris_Analysis_Report.html\n")