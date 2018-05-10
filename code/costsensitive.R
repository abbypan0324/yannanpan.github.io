all_packages = c("caret","pROC","C50","DMwR")
sapply(all_packages, require, character.only=TRUE)

## Training-validation-test split
test_cost = function(i){
    ## Read data
    german = read.table('code/germancredit.txt')
    colnames(german)[21] = 'score'
    german$score = factor(german$score, levels=c(1,2), labels=c("good","bad"))
    table(german$score)
    
    ## Cost matrix
    cost_mat <- matrix(c(0, 3, 7, 0), nrow = 2)
    rownames(cost_mat) <- colnames(cost_mat) <- levels(german$score)
    cost_mat
    
    set.seed(2426+i)
    train_idx = createDataPartition(german$score, p=0.7)[[1]]
    training = german[train_idx,]
    other = german[-train_idx,]
    
    eval_idx = createDataPartition(other$score, p=1/3)[[1]]
    evaluation = other[eval_idx,]
    test = other[-eval_idx,]
    
    ## CV
    cvCtrl <- trainControl(method = "cv",
                           number = 5,
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE)
    
    ## Baseline
    rfFit <- train(score ~., data = training,
                   method = 'rf',
                   trControl = cvCtrl,
                   ntree = 500,
                   tuneLength = 5,
                   metric = "ROC")
    
    baseProb <- predict(rfFit, newdata = test, type = "prob")[,1]
    basePred <- predict(rfFit, newdata = test)
    baseCM <- confusionMatrix(basePred, test$score)
    
    ## Sampling
    smoteTrain <- SMOTE(score ~ ., data=training)
    table(smoteTrain$score)
    
    smoteFit <- train(score ~., data = smoteTrain,
                      method = 'rf',
                      trControl = cvCtrl,
                      ntree = 500,
                      tuneLength = 5,
                      metric = "ROC")
    
    smotePred <- predict(smoteFit, newdata = test)
    smoteCM <- confusionMatrix(smotePred, test$score)
    
    ## Thresholding
    evalResults <- data.frame(score = evaluation$score)
    evalResults$RF <- predict(rfFit,
                              newdata = evaluation,
                              type = "prob")[,1]
    
    rfROC <- roc(evalResults$score, evalResults$RF,
                 levels = rev(levels(evalResults$score)))
    rfThresh <- coords(rfROC, x = "best", best.method = "closest.topleft") # best threshold 
    
    threPred <- factor(ifelse(baseProb > rfThresh[1],
                              "good", "bad"),
                       levels = levels(evalResults$score))
    threCM <- confusionMatrix(threPred, test$score)
    
    ## Cost-sensitive
    cvCtrlNoProb <- cvCtrl
    cvCtrlNoProb$summaryFunction <- defaultSummary
    cvCtrlNoProb$classProbs <- FALSE
    
    # (When employing costs, the predict function for C5.0 does not produce probabilities.)
    c5Fit <- train(score ~., data = training,
                   method = 'C5.0',
                   trControl = cvCtrlNoProb,
                   tuneGrid = data.frame(.model = "tree",
                                         .trials = c(1:100),
                                         .winnow = FALSE),
                   metric = "Kappa",
                   cost = cost_mat)
    
    c5Pred <- predict(c5Fit, newdata = test)
    c5CM <- confusionMatrix(c5Pred, test$score)
    
    return(c(sum(baseCM$table*cost_mat),
             sum(smoteCM$table*cost_mat),
             sum(threCM$table*cost_mat),
             sum(c5CM$table*cost_mat)))
}

test_cost_repeat = function(n){

    require(foreach)
    require(doParallel)

    cl <- makeCluster(detectCores())
    registerDoParallel(cl)
    
    res = foreach(i = 1:n,
                  .combine = rbind,
                  .export = "test_cost",
                  .packages = all_packages) %dopar%
                  {test_cost(i)}
    stopCluster(cl)
    
    colnames(res) = c("Baseline","SMOTE","thresholding","C5.0")
    return(res)
}


