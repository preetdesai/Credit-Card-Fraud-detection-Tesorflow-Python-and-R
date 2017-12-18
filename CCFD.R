##########################################################
# Credit card Fraud Detection : Date : 11/03/2017
##########################################################
tf$reset_default_graph()
rm(list = ls())
setwd(setwd("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\SEM3\\DataMining\\DDFDProject\\TensforFlow"))
underSampleData = read.csv("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\SEM3\\DataMining\\DDFDProject\\TensorFlow\\CCFDuncoded.csv", header = FALSE)
underSampleData <- as.data.frame(underSampleData)
names(underSampleData)[31]<-"Class"
dim(underSampleData)
head(underSampleData)
summary(underSampleData)

#Check for missing values
sum(is.na(underSampleData))

# Check for each class
table((unique(underSampleData)$Class))
prop.table(table(underSampleData$Class))
barplot(table((unique(underSampleData)$Class)), col = 'blue', ylim = c(0,500), main = 'Class Distribution')

# Make train and test datasets
set.seed(3333)
sample1 <- sample(nrow(underSampleData), size = nrow(underSampleData)*0.70)
train <- sample1
test <- -sample1
underSampletrainData <- underSampleData[train,]
underSampletestData <- underSampleData[test,]
XUTrain <- as.matrix(underSampletrainData[,1:30])
YUTrain <- as.matrix(underSampletrainData[,31])
XUTest <- as.matrix(underSampletestData[,1:30])
YUTest <- as.matrix(underSampletestData[,31])
dim(underSampletrainData)
dim(underSampletestData)
table((unique(underSampletrainData)$Class))
table((unique(underSampletestData)$Class))

# Do one hot encoding
library(keras)
YUTrain <- to_categorical(YUTrain, 2)
YUTest <- to_categorical(YUTest, 2)

# Build a tensorflow model
library(tensorflow)
learning_rate <- 0.1
training_epochs <- 300
cost_history <- c()
n_dim <- dim(XUTrain)[2]
n_class <- 2
model_path <- ("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\SEM3\\DataMining\\DDFDProject\\TensorFlow\\CCFD1")

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
  
  # Hidden layer with sigmoid Activation
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
training_step = tf$train$AdamOptimizer(learning_rate)$minimize(cost_function)
init <- tf$global_variables_initializer()
sess <- tf$Session()
sess$run(init)

# Calculate the cost and accuracy of each epoch
mse_history <- c()
accuracy_history <- c()

for (epoch in 1:training_epochs){
  sess$run(training_step, feed_dict= dict(x = XUTrain, y_ = YUTrain))
  cost <- sess$run(cost_function, feed_dict= dict(x = XUTrain, y_ = YUTrain))
  cost_history <- c(cost_history, cost)
  correct_prediction <- tf$equal(tf$argmax(y,1L), tf$argmax(y_, 1L))
  accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
  # Print ("Accuracy: ", (sess.run(accuracy, feed_dict = {x : test_x, y_:test_y})))
  pred_y <- sess$run(y, feed_dict= dict(x = XUTest))
  mse <- tf$reduce_mean(tf$square(pred_y - YUTest))
  mse_ <- sess$run(mse)
  mse_history <- c(mse_history, mse_)
  accuracy <- (sess$run(accuracy, feed_dict= dict(x = XUTrain, y_ = YUTrain)))
  print(paste(pred_y))
  accuracy_history <- c(accuracy_history,accuracy)
  print(paste('epoch : ', epoch, 'cost : ', cost, "-MSE : ", mse_, "-Train Accuracy : ", accuracy))
}

save_path <-saver$save(sess, model_path)
print(paste('Model saved in file :', save_path))

# Plot mse and accuracy graph
x11()
plot(mse_history, type = 'l', xlab = 'epoch', ylab = 'MSE', col = 'red', main = 'MSE_history')
plot(accuracy_history, type = 'l', xlab = 'epoch', ylab = 'Recall Score', col = 'blue', main = 'Recall_history')


# print the final accuracy
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
print(paste("Test Accuracy: ", (sess$run(accuracy, feed_dict=dict(x = XUTest, y_ = YUTest)))))

# Print the final mse
pred_y <- sess$run(y, feed_dict=dict(x = XUTest))
mse <- tf$reduce_mean(tf$square(pred_y - YUTest))
print(paste("MSE: ",sess$run(mse))) 

