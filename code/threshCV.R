library(caret)

thresh_code <- getModelInfo("rf", regex = FALSE)[[1]]
thresh_code$type <- c("Classification")
## Add the threshold as another tuning parameter
thresh_code$parameters <- data.frame(parameter = c("mtry", "threshold"),
                                     class = c("numeric", "numeric"),
                                     label = c("#Randomly Selected Predictors",
                                               "Probability Cutoff"))
## The default tuning grid code:
thresh_code$grid <- function(x, y, len = NULL, search = "grid") {
    p <- ncol(x)
    if(search == "grid") {
        grid <- expand.grid(mtry = floor(sqrt(p)), 
                            threshold = seq(.01, .99, length = len))
    } else {
        grid <- expand.grid(mtry = sample(1:p, size = len),
                            threshold = runif(1, 0, size = len))
    }
    grid
}

## Here we fit a single random forest model (with a fixed mtry)
## and loop over the threshold values to get predictions from the same
## randomForest model.
thresh_code$loop = function(grid) {   
    library(plyr)
    loop <- ddply(grid, c("mtry"),
                  function(x) c(threshold = max(x$threshold)))
    submodels <- vector(mode = "list", length = nrow(loop))
    for(i in seq(along = loop$threshold)) {
        index <- which(grid$mtry == loop$mtry[i])
        cuts <- grid[index, "threshold"] 
        submodels[[i]] <- data.frame(threshold = cuts[cuts != loop$threshold[i]])
    }    
    list(loop = loop, submodels = submodels)
}

## Fit the model independent of the threshold parameter
thresh_code$fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
    if(length(levels(y)) != 2)
        stop("This works only for 2-class problems")
    randomForest(x, y, mtry = param$mtry, ...)
}

## Now get a probability prediction and use different thresholds to
## get the predicted class
thresh_code$predict = function(modelFit, newdata, submodels = NULL) {
    class1Prob <- predict(modelFit, 
                          newdata, 
                          type = "prob")[, modelFit$obsLevels[1]]
    ## Raise the threshold for class #1 and a higher level of
    ## evidence is needed to call it class 1 so it should 
    ## decrease sensitivity and increase specificity
    out <- ifelse(class1Prob >= modelFit$tuneValue$threshold,
                  modelFit$obsLevels[1], 
                  modelFit$obsLevels[2])
    if(!is.null(submodels)) {
        tmp2 <- out
        out <- vector(mode = "list", length = length(submodels$threshold))
        out[[1]] <- tmp2
        for(i in seq(along = submodels$threshold)) {
            out[[i+1]] <- ifelse(class1Prob >= submodels$threshold[[i]],
                                 modelFit$obsLevels[1], 
                                 modelFit$obsLevels[2])
        }
    } 
    out  
}

## The probabilities are always the same but we have to create
## mulitple versions of the probs to evaluate the data across
## thresholds
thresh_code$prob = function(modelFit, newdata, submodels = NULL) {
    out <- as.data.frame(predict(modelFit, newdata, type = "prob"))
    if(!is.null(submodels)) {
        probs <- out
        out <- vector(mode = "list", length = length(submodels$threshold)+1)
        out <- lapply(out, function(x) probs)
    } 
    out 
}

fourStats <- function (data, lev = levels(data$obs), model = NULL) {
    ## This code will get use the area under the ROC curve and the
    ## sensitivity and specificity values using the current candidate
    ## value of the probability threshold.
    out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
    
    ## The best possible model has sensitivity of 1 and specificity of 1. 
    ## How far are we from that value?
    coords <- matrix(c(1, 1, out["Spec"], out["Sens"]), 
                     ncol = 2, 
                     byrow = TRUE)
    colnames(coords) <- c("Spec", "Sens")
    rownames(coords) <- c("Best", "Current")
    c(out, Dist = dist(coords)[1])
}

# source("code/threshCV.R")
# 
# set.seed(789)
# threFit <- train(Class ~ ., data = training,
#                  method = thresh_code,
#                  ## Minimize the distance to the perfect model
#                  metric = "Dist",
#                  maximize = FALSE,
#                  tuneLength = 20,
#                  ntree = 1000,
#                  trControl = trainControl(method = "cv",
#                                           classProbs = TRUE,
#                                           summaryFunction = fourStats))
