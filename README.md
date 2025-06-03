# R Data Science Portfolio

Welcome to my R programming portfolio showcasing data analysis, machine learning, and interactive visualization capabilities. This collection demonstrates my expertise in transforming complex data into actionable insights using R's powerful ecosystem.

## 📊 Featured Projects

### 🤖 **Machine Learning**
**[Predictive Modeling: Classification Engine](Machine-Learning-in-R)**  
- Built and evaluated classification models (Logistic Regression, Random Forest, XGBoost)  
- Achieved 92% accuracy in predicting customer churn  
- Key Techniques:  
  - Feature engineering with `recipes`  
  - Hyperparameter tuning using `tidymodels`  
  - Model interpretation with `DALEX`  

### 🌐 **Interactive Dashboards**  
**[COVID-19 Tracking Dashboard](Web-Apps-in-R)** | *Shiny Tutorial Series*  
- Developed real-time pandemic monitoring tool with:  
  - Interactive leaflet maps of case clusters  
  - Time-series forecasting visualizations  
  - Hospital capacity risk indicators  
- Tech Stack: `Shiny`, `flexdashboard`, `plotly`  

### 🦠 **Public Health Analytics**  
**[COVID-19 Outbreak Analysis](Using-R-to-Analyze-COVID-19)**  
- Processed 500K+ records from Johns Hopkins dataset  
- Key Deliverables:  
  - Reproduction number (R₀) estimation  
  - Mobility vs. infection rate correlation analysis  
  - Automated PDF report generation with `rmarkdown`  

## 🛠️ Technical Toolkit

```r
# Sample Code Snippet (Machine Learning)
library(tidymodels)
model <- logistic_reg() %>% 
  set_engine("glmnet") %>% 
  fit(churn ~ ., data = training_set)
