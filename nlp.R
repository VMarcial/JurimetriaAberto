install.packages("quanteda")
install.packages(c("readtext","spacyr"))

library(readr)
#df_train <- read_csv("relatorios_treinamento.csv")
#df_test <- read_csv("relatorios_teste.csv")
#df_test[1,]

#df_train = df_train[2:3]

#colnames(df_train) = c("julgado", "resultado")

#df_test = df_test[2:3]

#colnames(df_test) = c("julgado", "resultado")

#write_csv(df_train, 'relatorios_treinamento2.csv', col_names = T)
#write_csv(df_test, 'relatorios_teste2.csv', col_names = T)

#df_train

#library(quanteda)
#corpus_train = corpus(df_train$julgado)
#summary(corpus_train)
#docvars(corpus_train, "resultado") = df_train$resultado


#dfm_train = dfm(corpus_train, remove = stopwords("portuguese"), remove_punct = T, stem = T)

#?write_csv


#dfm_train[1:5,1:5]

#prop.table()



library(readr)
library(caret)
library(caTools)

df_dec <- read_csv("df_dec.csv")
df_dec = df_dec[2:4]

## 80% of the sample size
#smp_size <- floor(0.8* nrow(df_dec))

## set the seed to make your partition reproducible
#set.seed(123)
#train_ind <- sample(seq_len(nrow(df_dec)), size = smp_size)

#train <- df_dec[train_ind, ]
#test <- df_dec[-train_ind, ]

library(quanteda)
library(stringi)
df_dec$julgado = quanteda::char_tolower(df_dec$julgado)
df_dec$julgado = stri_trans_general(df_dec$julgado, "Latin-ASCII")
df_dec$julgado = stri_trans_char(df_dec$julgado, 'º', ' ')
df_dec$julgado = stri_trans_char(df_dec$julgado, 'ª', ' ')
df_dec$julgado = stri_trans_char(df_dec$julgado, '°', ' ')
#df_dec$julgado = stri_trans_char(df_dec$julgado, '%', 'cento')
df_dec$julgado = stri_replace_all(df_dec$julgado, regex ='%', replacement = ' por cento')

df_dec = subset(df_dec, df_dec$decisao != 4)
df_dec = subset(df_dec, df_dec$decisao != 5)


corpus_full = corpus(df_dec$julgado)
docvars(corpus_full, "resultado_y") = df_dec$resultado
docvars(corpus_full, "decisao_y") = df_dec$decisao
summary(corpus_full)

token <- tokens(corpus_full, include_docvars = T, remove_numbers = T, remove_punct = T, remove_symbols = T, split_hyphens = T)


sw <- stopwords(language = "pt")
token <- tokens_remove(token, c(sw, "é"))
token <- tokens_wordstem(token, language = "portuguese")
dfm_full = dfm(token)
dfm_full

#dfm_full = dfm(corpus_full, remove = stopwords("portuguese"), remove_punct = T, stem = T, remove_numbers = T, include_docvars = T)

colnames(dfm_full)

dfm_full = dfm_trim(dfm_full, sparsity = 0.97)



dfm_full = convert(dfm_full, "data.frame")
dfm_full$resultado_y = df_dec$resultado
dfm_full$decisao_y = df_dec$decisao

dfm_full$decisao_y

## 80% of the sample size
smp_size <- floor(0.8* nrow(dfm_full))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dfm_full)), size = smp_size)

train <- dfm_full[train_ind, ]
test <- dfm_full[-train_ind, ]

decisao_train = train$decisao_y
decisao_test = test$decisao_y

train = subset(train, select = -c(doc_id, resultado_y))
test = subset(test, select = -c(doc_id, resultado_y))

#train = convert(train, "data.frame")
#test = convert(test, "data.frame")
#train$resultado_y
#train$resultado_y
#train$decisao_y

#dfm_trim(dfm_train, 0.995, termfreq_type = 'prop')

# randomForest

#library(randomForest)
#RF_model = randomForest(decisao_y ~ ., data=train, do.trace = T)
#predictRF = predict(RF_model, newdata=test)
#table(test, predictRF)




library(caret)
library(doParallel)
c1 <- makePSOCKcluster(22)
registerDoParallel(c1)

trainctrl <- trainControl(verboseIter = TRUE)
nn_model = train(as.factor(decisao_y) ~ ., data = train, method = 'mlp', trControl = trainctrl)
rf_model = train(as.factor(decisao_y) ~ ., data = train, method = 'rf', trControl = trainctrl)
rf_model = nn_model

# SVM separando df original
## 80% of the sample size
smp_size <- floor(0.8* nrow(df_dec))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df_dec)), size = smp_size)

df_dec_train <- df_dec[train_ind, ]
df_dec_test <- df_dec[-train_ind, ]

decisao_train = train$decisao_y
decisao_test = test$decisao_y

train = subset(train, select = -c(doc_id, resultado_y))
test = subset(test, select = -c(doc_id, resultado_y))

library(kernlab)
### TODO fazer modelo com SVM. Parece o mais adequado para classificação com texto.
mdl <- train(x=cbind(julgado=df_dec_train$julgado), y = cbind(decisao=df_dec_train$decisao), method="svmBoundrangeString", metric = "Accuracy")#,trControl=trainControl(method="cv"))

mdl <- tsv <- ksvm(df_dec_train$julgado,df_dec_train$decisao,kernel="stringdot",kpar=list(length=5),cross=3,C=10)


nn_model$pred

julgado = df_dec$julgado

svm_model = train(as.factor(decisao) ~ julgado, data = df_dec, method = 'svmBoundrangeString', trControl = trainctrl)

#pred = caret::predict(nn_model, newdata = test)

pred = caret::predict.train(nn_model, newdata = test)

#pred = floor(pred)

confusionMatrix(data = as.factor(pred), reference = as.factor(test$decisao_y))

mean(pred == test$decisao_y)

svm_model = train(decisao ~ julgado, data = df_dec, method = 'svmBoundrangeString')

stopCluster(c1)
