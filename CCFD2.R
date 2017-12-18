##########################################################
# Credit card Fraud Detection : Date : 11/03/2017
##########################################################
library(tensorflow)
rm(list = ls())
tf$reset_default_graph()
setwd(setwd("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\SEM3\\DataMining\\DDFDProject\\TensorFlow"))
data = read.csv("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\SEM3\\DataMining\\DDFDProject\\TensorFlow\\CCFDuncoded.csv", header = FALSE)
data <- as.data.frame(data)
names(data)[31]<-"Class"
X <- as.matrix(data[,1:30])
yl <- as.matrix(data[,31])
dim(data)
head(data)
summary(data)

#Check for missing values
#sum(is.na(data))

# Check for each class
#table((unique(data)$Class))
#prop.table(table(data$Class))
#barplot(table((unique(data)$Class)), col = 'blue', ylim = c(0,500), main = 'Class Distribution')

# Make train and test datasets
set.seed(3333)
sample1 <- sample(nrow(data), size = nrow(data)*0.70)
train <- sample1
test <- -sample1
trainData <- data[train,]
testData <- data[test,]
XTrain <- as.matrix(trainData[,1:30])
YTrain <- as.matrix(trainData[,31])
XTest <- as.matrix(testData[,1:30])
YTest <- as.matrix(testData[,31])
#dim(trainData)
#dim(testData)
#table((unique(trainData)$Class))
#table((unique(testData)$Class))

# Do one hot encoding
library(keras)
YTrain <- to_categorical(YTrain, 2)
YTest <- to_categorical(YTest, 2)

# Build a tensorflow model
library(tensorflow)
model_path <- ("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\SEM3\\DataMining\\DDFDProject\\TensorFlow\\CCFD1")
learning_rate <- 0.1
training_epochs <- 300
cost_history <- c()
n_dim <- dim(XTrain)[2]
n_class <- 2

# Define the number of hidden layers and neurons for each layer
n_hidden_1 <- 20
n_hidden_2 <- 20
n_hidden_3 <- 20
n_hidden_4 <- 20

# Define weights and Biases
x <- tf$placeholder(tf$float32, list(NULL, n_dim))
W <- tf$Variable(tf$zeros(list(n_dim, n_class)))
b <- tf$Variable(tf$zeros(n_class))
y_ <- tf$placeholder(tf$float32, list(NULL, n_class))

multi_perceptron <- function(x, weights, biases){
  # Hidden layer with sigmoid Activation
  layer_1 <- tf$add(tf$matmul(x, weights$h1), biases$b1)
  layer_1 <- tf$nn$sigmoid(layer_1)
  
  # Hidden layer with sigmoid Activation
  layer_2 <- tf$add(tf$matmul(layer_1, weights$h2), biases$b2)
  layer_2 <- tf$nn$sigmoid(layer_2)
  
  # Hidden layer with sigmoid Activation
  layer_3 <- tf$add(tf$matmul(layer_2, weights$h3), biases$b3)
  layer_3 <- tf$nn$sigmoid(layer_3)
  
  # Hidden layer with RELU Activation
  layer_4 <- tf$add(tf$matmul(layer_3, weights$h4), biases$b4)
  layer_4 <- tf$nn$relu(layer_4)
  
  # Output layer with linear activation
  out_layer <- tf$matmul(layer_4, weights$out) + biases$out
  return(out_layer)
}

# Define the weights and the biases for each layer
weights <- vector(mode = "list", length = 5)
names(weights) <- c('h1','h2','h3','h4','out')
weights[[1]] <- tf$Variable(tf$truncated_normal(shape(n_dim, n_hidden_1)))
weights[[2]] <- tf$Variable(tf$truncated_normal(shape(n_hidden_1, n_hidden_2)))
weights[[3]] <- tf$Variable(tf$truncated_normal(shape(n_hidden_2, n_hidden_3)))
weights[[4]] <- tf$Variable(tf$truncated_normal(shape(n_hidden_3, n_hidden_4)))
weights[[5]] <- tf$Variable(tf$truncated_normal(shape(n_hidden_4, n_class)))

biases <- vector(mode = "list", length = 5)
names(biases) <- c('b1','b2','b3','b4','out')
biases[[1]]<- tf$Variable(tf$truncated_normal(shape(n_hidden_1)))
biases[[2]]<- tf$Variable(tf$truncated_normal(shape(n_hidden_2)))
biases[[3]]<- tf$Variable(tf$truncated_normal(shape(n_hidden_3)))
biases[[4]]<- tf$Variable(tf$truncated_normal(shape(n_hidden_4)))
biases[[5]]<- tf$Variable(tf$truncated_normal(shape(n_class)))

# Initialize all the variables
init <- tf$global_variables_initializer()
saver <- tf$train$Saver()

# Call the model
y <- multi_perceptron(x, weights, biases)
# Define the cost function and optimizer
cost_function <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = y, labels = y_))
training_step <- tf$train$AdamOptimizer(learning_rate)$minimize(cost_function)


init <- tf$global_variables_initializer()
saver <- tf$train$Saver()
sess <- tf$Session()
sess$run(init)
saver$restore(sess, model_path)
prediction<- tf$argmax(y,1L)
correct_prediction <- tf$equal(prediction, tf$argmax(y_,1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Print (Accuracy)
print(paste('###########################################################'))
print(paste("O stands for Normal Transaction and 1 is for fraudulent transaction"))
print(paste('###########################################################'))

y_true <- c()
y_pred <- c()
library('reshape')
for (i in as.numeric(rownames(XTest))){
  k <-as.numeric((X[i,]))
  k <-t(X[i,])
  prediction_run <- sess$run(prediction, feed_dict = dict(x = k))
  print(paste("Original Class : ", yl[i], "Predicted Values : ", prediction_run, i))
  y_true <- c(y_true, yl[i])
  y_pred <- c(y_pred, prediction_run)
}

library(ggplot2)
plotConfusion <-function(confusion_matrix){
  True_Class<- factor(c(0, 0, 1, 1))
  Prediction <- factor(c(0, 1, 0, 1))
  Y <- c(confusion_matrix[1], confusion_matrix[3], confusion_matrix[2], confusion_matrix[4])
  df <- data.frame(True_Class, Prediction, Y)
  
  return(ggplot(data =  df, mapping = aes(x = True_Class, y = Prediction)) +
           geom_tile(aes(fill = Y), colour = "white") +
           geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
           scale_fill_gradient(low = "yellow", high = "dark green") +
           theme_bw() + theme(legend.position = "none"))
}

confusion_matrix <- ftable(y_true, y_pred)
library(caret)
x11()
cm1 <- confusionMatrix(y_pred, y_true)
cmm1 <- plotConfusion(confusion_matrix)
cmm1

