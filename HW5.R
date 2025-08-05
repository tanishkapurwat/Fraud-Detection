#Tanishka Purwat DSC HW5 

library(ggplot2)
library(dplyr)
library(readr)
library(corrplot)
library(GGally)
library(caret)
library(factoextra)
library(ggfortify)
library(dendextend)
library(cluster)
library(e1071)
library(rpart)
library(rpart.plot)
library(pROC)

df <- read_csv("synthetic_fraud_dataset.csv")

#Data Exploration
summary(df)
colSums(is.na(df))
total_missing_values <- sum(colSums(is.na(df)))
print(paste("Total missing values:", total_missing_values))

num_cols <- c("Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
              "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
              "Card_Age", "Transaction_Distance", "Risk_Score")

for (col in num_cols) {
  p <- ggplot(df, aes(x = !!sym(col))) +  
    geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
    ggtitle(paste("Distribution of", col)) +
    xlab(col) +
    theme_minimal()
  print(p)
}

for (col in num_cols) {
  print(
    ggplot(df, aes(x = as.factor(Fraud_Label), y = .data[[col]], fill = as.factor(Fraud_Label))) +
      geom_boxplot() +
      ggtitle(paste("Boxplot of", col, "by Fraud Label")) +
      xlab("Fraud Label") +
      ylab(col) +
      theme_minimal()
  )
}

num_data <- df %>% select(all_of(num_cols)) 
cor_matrix <- cor(num_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", tl.cex = 0.8)

df$Fraud_Label <- as.factor(df$Fraud_Label)
p1 <- ggplot(df, aes(x = Transaction_Amount, y = Account_Balance, color = Fraud_Label)) +
  geom_point(alpha = 0.5) +
  ggtitle("Transaction Amount vs. Account Balance") +
  theme_minimal()

p2 <- ggplot(df, aes(x = Transaction_Distance, y = Risk_Score, color = Fraud_Label)) +
  geom_point(alpha = 0.5) +
  ggtitle("Transaction Distance vs. Risk Score") +
  theme_minimal()

p3 <- ggplot(df, aes(x = as.factor(Transaction_Type), y = Transaction_Amount, fill = Fraud_Label)) +
  geom_boxplot() +
  ggtitle("Transaction Type vs. Transaction Amount") +
  xlab("Transaction Type") +
  theme_minimal()

p4 <- ggplot(df, aes(x = Risk_Score, fill = Fraud_Label)) +
  geom_density(alpha = 0.5) +
  ggtitle("Risk Score Distribution by Fraud Label") +
  theme_minimal()

print(p1)
print(p2)
print(p3)
print(p4)

#Data Cleaning

handle_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  x <- ifelse(x < lower_bound, lower_bound, x)
  x <- ifelse(x > upper_bound, upper_bound, x)
  return(x)
}

df[num_cols] <- df[num_cols] %>% mutate(across(everything(), handle_outliers))

#d. Data Preprocessing

df$Transaction_Type <- as.factor(df$Transaction_Type)
df$Device_Type <- as.factor(df$Device_Type)
df$Location <- as.factor(df$Location)
df$Merchant_Category <- as.factor(df$Merchant_Category)
df$Card_Type <- as.factor(df$Card_Type)
df$Authentication_Method <- as.factor(df$Authentication_Method)
df$Fraud_Label <- as.factor(df$Fraud_Label)

df_encoded <- dummyVars(" ~ .", data = df)
df <- as.data.frame(predict(df_encoded, df))

for (col in num_cols) {
  print(
    ggplot(df, aes(x = as.factor(Fraud_Label), y = .data[[col]], fill = as.factor(Fraud_Label))) +
      geom_boxplot() +
      ggtitle(paste("Boxplot", col, "part c")) +
      xlab("Fraud Label") +
      ylab(col) +
      theme_minimal()
  )
}

df <- df %>% select(-Transaction_ID, -User_ID, -Timestamp)
df[num_cols] <- scale(df[num_cols]) 
summary(df)

#Clustering

print(colnames(df)) 
#existing_categorical_cols <- c( "Transaction_Type.ATM Withdrawal", "Transaction_Type.Bank Transfer", 
                                #"Device_Type.Laptop" ,"Device_Type.Mobile","Device_Type.Tablet",    
                               # "Location.London","Location.Mumbai","Location.New York","Location.Sydney","Location.Tokyo",
                                #"Merchant_Category.Clothing","Merchant_Category.Electronics","Merchant_Category.Groceries","Merchant_Category.Restaurants","Merchant_Category.Travel", 
                              #  "Card_Type.Amex","Card_Type.Discover","Card_Type.Mastercard","Card_Type.Visa","Authentication_Method.Biometric","Authentication_Method.OTP","Authentication_Method.Password","Authentication_Method.PIN")

df_clustering <- df %>% select(-Fraud_Label.0, -Fraud_Label.1)

#feature selection as there are too many features in existing_categorical_cols

selected_features <- c("Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
                       "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
                       "Card_Age", "Transaction_Distance", "Risk_Score", "Is_Weekend")

df_clustering <- df_clustering %>% select(all_of(selected_features))
df_clustering <- as.data.frame(lapply(df_clustering, as.numeric))

print(dim(df_clustering))

set.seed(42)
sample_size <- min(5000, nrow(df_clustering))  # max 5000 samples (memory issues)
df_sample <- df_clustering[sample(1:nrow(df_clustering), sample_size), ]

fviz_nbclust(df_sample, kmeans, method = "wss") +
  ggtitle("Elbow Method (Optimal K)")

set.seed(42)
optimal_k <- 1
kmeans_result <- kmeans(df_clustering, centers = optimal_k, nstart = 25, algorithm = "Lloyd")

df$Cluster <- as.factor(kmeans_result$cluster)

library(ggfortify)
pca_result <- prcomp(df_clustering, scale. = TRUE)

autoplot(pca_result, data = df, colour = "Cluster") +
  ggtitle("PCA Projection") +
  theme_minimal()

print(table(df$Cluster))

#HAC
df_clustering <- df %>% select(-Fraud_Label.0, -Fraud_Label.1)

df_clustering <- df_clustering %>% select(all_of(selected_features))
df_clustering <- as.data.frame(df_clustering)
df_clustering <- scale(df_clustering)

set.seed(42)
sample_size <- min(5000, nrow(df_clustering))  # memory issue without row reduction
df_sample <- df_clustering[sample(1:nrow(df_clustering), sample_size), ]

distance_matrix <- dist(df_sample, method = "euclidean") 
hc_result <- hclust(distance_matrix, method = "ward")
plot(as.dendrogram(hc_result), main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "")

k <- 4  
df$Cluster <- cutree(as.hclust(hc_result), k = k)

pca_result <- prcomp(df_clustering)
autoplot(pca_result, data = df, colour = "Cluster") +
  ggtitle("Hierarchical Clustering - PCA Projection") +
  theme_minimal()

table(df$Cluster)

#Classification

df_classification$Fraud_Label <- as.factor(df$Fraud_Label.1)
df_classification <- df %>% select(-Fraud_Label.0, -Fraud_Label.1)
selected_features <- c("Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
                       "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d",
                       "Card_Age", "Transaction_Distance", "Risk_Score", "Is_Weekend")

df_classification <- df_classification %>% select(all_of(selected_features), Fraud_Label)

df_classification$Fraud_Label <- as.factor(df$Fraud_Label.1)

set.seed(42)
trainIndex <- createDataPartition(df_classification$Fraud_Label, p = 0.8, list = FALSE)
train_data <- df_classification[trainIndex, ]
test_data <- df_classification[-trainIndex, ]

set.seed(42)
svm_model <- svm(Fraud_Label ~ ., data = train_data, kernel = "radial", cost = 1, gamma = 0.1)
dt_model <- rpart(Fraud_Label ~ ., data = train_data, method = "class", control = rpart.control(cp = 0.01))

svm_predictions <- predict(svm_model, test_data, type = "class")
dt_predictions <- predict(dt_model, test_data, type = "class")

svm_accuracy <- mean(svm_predictions == test_data$Fraud_Label)
dt_accuracy <- mean(dt_predictions == test_data$Fraud_Label)

svm_cm <- confusionMatrix(as.factor(svm_predictions), test_data$Fraud_Label)
dt_cm <- confusionMatrix(as.factor(dt_predictions), test_data$Fraud_Label)

svm_roc <- roc(test_data$Fraud_Label, as.numeric(svm_predictions))
dt_roc <- roc(test_data$Fraud_Label, as.numeric(dt_predictions))

svm_auc <- auc(svm_roc)
dt_auc <- auc(dt_roc)

print(paste("SVM Accuracy:", round(svm_accuracy * 100, 2), "%"))
print(paste("Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%"))

print(paste("SVM AUC:", round(svm_auc, 4)))
print(paste("Decision Tree AUC:", round(dt_auc, 4)))

print("SVM Confusion Matrix:")
print(svm_cm)

print("Decision Tree Confusion Matrix:")
print(dt_cm)

rpart.plot(dt_model, main = "Decision Tree for Fraud Prediction")

#Evaluation

TP <- dt_cm$table[2,2] 
FP <- dt_cm$table[1,2] 
FN <- dt_cm$table[2,1] 
TN <- dt_cm$table[1,1]  

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)

print(paste("Precision:", round(precision, 4)))
print(paste("Recall:", round(recall, 4)))

dt_probabilities <- predict(dt_model, test_data, type = "prob")[,2]
dt_roc <- roc(test_data$Fraud_Label, dt_probabilities)

plot(dt_roc, col = "blue", main = "ROC Curve")
auc_value <- auc(dt_roc)
print(paste("AUC:", round(auc_value, 4)))

set.seed(42)
dt_model <- rpart(Fraud_Label ~ ., data = train_data, method = "class",
                  control = rpart.control(cp = 0.02, minsplit = 20, maxdepth = 5)) 

train_control <- trainControl(method = "cv", number = 5)  # 5-fold CV
dt_model_cv <- train(Fraud_Label ~ ., data = train_data, method = "rpart",
                     trControl = train_control, tuneLength = 10)

cor_matrix <- cor(df_classification %>% select(-Fraud_Label))
print("Features Correlated with Fraud_Label:")
print(sort(abs(cor_matrix[, "Fraud_Label"]), decreasing = TRUE))

print("Class Distribution for Test Set:")
print(table(test_data$Fraud_Label))

dt_predictions <- predict(dt_model, test_data, type = "class")

dt_cm <- confusionMatrix(dt_predictions, test_data$Fraud_Label)
print("Confusion Matrix Pruned Decision Tree:")
print(dt_cm)

rpart.plot(dt_model, main = "Pruned Decision Tree")

dt_probabilities <- predict(dt_model, test_data, type = "prob")[,2]
dt_roc <- roc(test_data$Fraud_Label, dt_probabilities)
plot(dt_roc, col = "blue", main = "ROC Curve Pruned Decision Tree")

auc_value <- auc(dt_roc)
print(paste("AUC After Fixing Overfitting:", round(auc_value, 4)))





