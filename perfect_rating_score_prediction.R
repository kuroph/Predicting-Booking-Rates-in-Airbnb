###########################################################################################
################## This code trains a machine learning model based on the #################
############### two input training files and outputs the prediction results ###############
################################# based on the test file. #################################
###########################################################################################

library(tidyverse)
library(text2vec)
library(tm)
library(SnowballC)
library(glmnet)
library(vip)
library(naivebayes)
library(ranger)
library(xgboost)
library(ROCR)
library(rsample)
library(caret)
library(randomForest)
library(fastDummies)
library(Matrix)
library(text2vec)
set.seed(1)

train_x <- read_csv("airbnb_train_x_2023.csv")
train_y <- read_csv("airbnb_train_y_2023.csv")
test_x <- read_csv("airbnb_test_x_2023.csv")

###########################################################################################
#################################STEP 1: DATA PREPARATION##################################
###########################################################################################
x_data <- rbind(train_x, test_x)

train_y <- train_y %>%
  mutate(perfect_rating_score = ifelse(perfect_rating_score == "YES", 1, 0),
         high_booking_rate = ifelse(high_booking_rate == "YES", 1, 0))

###########################################################################################
##################################STEP 2: DATA CLEANING####################################
###########################################################################################

# Cleaning variables:
# For cancellation_policy: {strict, super_strict_30} --> {strict}
# Convert cleaning_fee and price into numbers
# Replace NAs in cleaning_fee and price with 0
# Replace NAs in beds, bedrooms, host_total_istings_count with their mean

# Creating New Features:
# price_per_person is the nightly price per accommodates 
# has_cleaning_fee - {YES, NO}
# bed_category - {bed, other}
# property_category - {apartment, hotel, condo, house, other}
# ppp_ind is 1 if the price_per_person is greater than the median for the property_category, and 0 otherwise

airbnb_clean <- x_data %>%
  group_by(cancellation_policy) %>%
  mutate(cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30"), 'strict', cancellation_policy)) %>%
  ungroup() %>%
  mutate(cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
         cleaning_fee = as.numeric(parse_number(cleaning_fee)),
         price = ifelse(is.na(price), 0, price),
         price = as.numeric(parse_number(price)),
         bedrooms = as.numeric(ifelse(is.na(bedrooms), mean(bedrooms, na.rm=TRUE), bedrooms)),
         beds = as.numeric(ifelse(is.na(beds), mean(beds, na.rm=TRUE), beds)),
         host_total_listings_count = as.numeric(ifelse(is.na(host_total_listings_count), mean(host_total_listings_count, na.rm=TRUE), host_total_listings_count))) %>%
  mutate(price_per_person = as.numeric(round(price/accommodates, 2)),
         has_cleaning_fee = as.factor(ifelse(cleaning_fee == 0, "NO", "YES")),
         bed_category = as.factor(ifelse(bed_type == "Real Bed", "bed", "other")),
         property_category = ifelse(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "apartment", property_type),
         property_category = ifelse(property_category %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel", property_category),
         property_category = ifelse(property_category %in% c("Townhouse", "Condominium"), "condo", property_category),
         property_category = ifelse(property_category %in% c("Bungalow", "House"), "house", property_category),
         property_category = ifelse(property_category %in% c("apartment", "hotel", "condo", "house"), property_category, "other"),
         property_category = as.factor(property_category),
         property_type = as.factor(property_type)
  ) %>%
  group_by(property_category) %>%
  mutate(ppp_ind = as.numeric(ifelse(price_per_person > median(price_per_person, na.rm=TRUE), 1, 0))) %>%
  ungroup() %>%
  mutate(bed_type = as.factor(bed_type),
         cancellation_policy = as.factor(cancellation_policy),
         room_type = as.factor(room_type),
         ppp_ind = as.factor(ppp_ind))

########################################################################################################
# Cleaning variables:
# Replace NAs in bathrooms with the median value
# Replace NAs in host_is_superhost with FALSE

# Creating New Features new features:
# "charges_for_extra" - {YES, NO}
# "host_acceptance" - {ALL, SOME, MISSING}
# "host_response"  - {ALL, SOME, MISSING}
# "has_min_nights" - {YES, NO}
# Replace market with "OTHER" if there are under 300 instances in that market. Convert market to a factor.
airbnb_clean <- airbnb_clean %>% 
  mutate(market = ifelse(is.na(market), "MISSING", market))

airbnb_clean <- airbnb_clean %>%
  mutate(bathrooms = as.numeric(ifelse(is.na(bathrooms), median(bathrooms, na.rm=TRUE), bathrooms)),
         host_is_superhost = ifelse(is.na(host_is_superhost), FALSE, host_is_superhost),
         extra_people = as.numeric(parse_number(extra_people)),
         charges_for_extra = as.factor(ifelse(extra_people > 0, "YES", "NO")),
         host_acceptance = as.factor(case_when(is.na(host_acceptance_rate) ~ "MISSING",
                                               host_acceptance_rate == 1.00 | host_acceptance_rate == "100%"~ "ALL",
                                               TRUE ~ "SOME")),
         host_response = as.factor(case_when(is.na(host_response_rate) ~ "MISSING",
                                             host_response_rate == 1.00 |  host_response_rate == "100%" ~ "ALL",
                                             TRUE ~ "SOME")),
         has_min_nights = as.factor(ifelse(minimum_nights > 1, "YES", "NO"))) %>%
  group_by(market) %>%
  mutate(market = as.factor(ifelse(n() < 300, "OTHER", market))) %>%
  ungroup()

########################################################################################################

airbnb_clean<-airbnb_clean%>%
  mutate(list_till_today=Sys.Date()-host_since,
         last_review_till_today=Sys.Date()-first_review,
         list_till_today = as.numeric(list_till_today),
         last_review_till_today = as.numeric(last_review_till_today))

airbnb_clean <- airbnb_clean %>% 
  mutate(list_till_today =  ifelse(is.na(list_till_today), mean(list_till_today, na.rm=TRUE), list_till_today),
         interaction = ifelse(is.na(interaction), "MISSING", interaction),
         access = ifelse(is.na(access), "MISSING", access),
         description = ifelse(is.na(description), "MISSING", description))
########################################################################################################


###########################################################################################
###############################STEP 3: FEATURES FROM TEXT##################################
###########################################################################################
########################################## INTERACTION ###########################################
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    word_tokenizer 
}

it_interaction = itoken(airbnb_clean$interaction, 
                  preprocessor = tolower, 
                  tokenizer = cleaning_tokenizer, 
                  progressbar = FALSE)
vocab_interaction <- create_vocabulary(it_interaction, ngram = c(1,3))
pruned_vocab_interaction <- prune_vocabulary(vocab_interaction, term_count_min = 5)
vectorizer_interaction = vocab_vectorizer(pruned_vocab_interaction)
dtm_interaction <- create_dtm(it_interaction, vectorizer_interaction)
tcm_interaction <- create_tcm(it_interaction, vectorizer_interaction,  skip_grams_window = 5L)

glove = GlobalVectors$new(rank = 50, x_max = 10)
model_interaction = glove$fit_transform(tcm_interaction, n_iter = 10, 
                                        convergence_tol = 0.01, n_threads = 8)
word_vectors_interactions = model_interaction + t(glove$components)
weighted_dtm_interaction <- dtm_interaction %*% word_vectors_interactions

kmeans_result <- kmeans(weighted_dtm_interaction, centers = 150)

airbnb_clean<-airbnb_clean%>%
  mutate(cluster_interaction <- kmeans_result$cluster)
colnames(airbnb_clean)[81] <- "cluster_interaction"




############################################## ACCESS ##############################################
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    word_tokenizer 
}

it_access = itoken(airbnb_clean$access, 
                        preprocessor = tolower, 
                        tokenizer = cleaning_tokenizer, 
                        progressbar = FALSE)
vocab_access <- create_vocabulary(it_access, ngram = c(1,3))
pruned_vocab_access <- prune_vocabulary(vocab_access, term_count_min = 5)
vectorizer_access = vocab_vectorizer(pruned_vocab_access)
dtm_access <- create_dtm(it_access, vectorizer_access)
tcm_access <- create_tcm(it_access, vectorizer_access,  skip_grams_window = 5L)

glove = GlobalVectors$new(rank = 10, x_max = 10)
model_access = glove$fit_transform(tcm_access, n_iter = 10, 
                                        convergence_tol = 0.01, n_threads = 8)
word_vectors_access = model_access + t(glove$components)
weighted_dtm_access <- dtm_access %*% word_vectors_access

kmeans_result <- kmeans(weighted_dtm_access, centers = 100)

airbnb_clean<-airbnb_clean%>%
  mutate(cluster_access <- kmeans_result$cluster)
colnames(airbnb_clean)[82] <- "cluster_access"





############################################## DESCRIPTION ##############################################
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(stopwords()) %>% #remove stopwords
    word_tokenizer 
}

it_descrip = itoken(airbnb_clean$description, 
                   preprocessor = tolower, 
                   tokenizer = cleaning_tokenizer, 
                   progressbar = FALSE)
vocab_descrip <- create_vocabulary(it_descrip, ngram = c(1,3))
pruned_vocab_descrip <- prune_vocabulary(vocab_descrip, term_count_min = 5)
vectorizer_descrip = vocab_vectorizer(pruned_vocab_descrip)
dtm_descrip <- create_dtm(it_descrip, vectorizer_descrip)
tcm_descrip <- create_tcm(it_descrip, vectorizer_descrip,  skip_grams_window = 5L)

glove = GlobalVectors$new(rank = 50, x_max = 10)
model_descrip = glove$fit_transform(tcm_descrip, n_iter = 10, 
                                   convergence_tol = 0.01, n_threads = 8)
word_vectors_descrip = model_descrip + t(glove$components)
weighted_dtm_descrip <- dtm_descrip %*% word_vectors_descrip

kmeans_result <- kmeans(weighted_dtm_descrip, centers = 75)
###
#keep_cols <- seq_len(ncol(airbnb_clean)) != ncol(airbnb_clean)
#airbnb_clean <- airbnb_clean[, keep_cols]
###
airbnb_clean<-airbnb_clean%>%
  mutate(cluster_descrip <- kmeans_result$cluster)
colnames(airbnb_clean)[83] <- "cluster_descrip"





############################################## AMENITIES ##############################################

cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>%
    stemDocument %>%
    word_tokenizer 
}

it_amenities = itoken(airbnb_clean$amenities, 
                   preprocessor = tolower, 
                   tokenizer = cleaning_tokenizer, 
                   progressbar = FALSE)
vocab_amenities <- create_vocabulary(it_amenities)
pruned_vocab_amenities <- prune_vocabulary(vocab_amenities, term_count_min = 500)
vectorizer_amenities = vocab_vectorizer(pruned_vocab_amenities)
dtm_amenities <- create_dtm(it_amenities, vectorizer_amenities)





#################CREATING A SPARSE MATRIX FROM THE DATAFRAME#################

perfect_data <- airbnb_clean %>%
  select(accommodates, bedrooms, cancellation_policy, has_cleaning_fee, host_total_listings_count, 
         price, ppp_ind, property_category, bathrooms, charges_for_extra,
         host_acceptance, has_min_nights, market, host_is_superhost, room_type,
         list_till_today, last_review_till_today, cluster_interaction, cluster_access, cluster_descrip)

dummy <- dummyVars( ~ . , data=perfect_data, fullRank = TRUE)
perfect_data<- data.frame(predict(dummy, newdata = perfect_data))

my_sparse <- sparseMatrix(i = as.integer(row(perfect_data)),
                          j = as.integer(col(perfect_data)),
                          x = as.numeric(unlist(perfect_data)),
                          dims = c(nrow(perfect_data), ncol(perfect_data)),
                          dimnames = list(NULL, names(perfect_data)))
my_sparse <- cbind(my_sparse, dtm_amenities)#, weighted_dtm_access, weighted_dtm_interaction, weighted_dtm_descrip)

#weighted_dtm_mat <- cbind(weighted_dtm_access, weighted_dtm_interaction, weighted_dtm_descrip)

###########################################################################################
##################################STEP 4: SEPARATING TEST##################################
###########################################################################################

airbnb_train <- my_sparse[1:dim(train_x)[1], ]
airbnb_test <- my_sparse[(dim(train_x)[1] + 1):(dim(train_x)[1] + dim(test_x)[1]), ]

###########################################################################################
##################################STEP 5: HYPERPARAMETER TUNING############################
###########################################################################################

valid_instn = sample(nrow(airbnb_train), 0.30*nrow(airbnb_train))
train_boost_x <- airbnb_train[-valid_instn, ]
train_boost_y <- train_y$perfect_rating_score[-valid_instn]
valid_boost_x <- airbnb_train[valid_instn, ]
valid_boost_y <- train_y$perfect_rating_score[valid_instn]

c(11, 12, 13, 14, 15, 16, 17, 18)
depth_list = c(9, 10, 11)
eta_list = c(0.01)
nrounds_choose <- c(300, 500, 700)
max_req_TPR = 0
best_depth = depth_list[1]
best_eta = eta_list[1]
best_nrounds = nrounds_choose[1]

for(i in c(1:length(depth_list))){
  print(i)
  for(j in c(1:length(best_eta))){
    for(k in c(1:length(nrounds_choose))){
      xgbst <- xgboost(data = train_boost_x, label = train_boost_y,
                       max.depth = depth_list[i], eta = eta_list[j], nrounds = nrounds_choose[k],
                       objective = "binary:logistic", verbosity = 0, verbose = FALSE)

      preds_bst <- predict(xgbst, valid_boost_x)
      pred_full <- prediction(preds_bst, valid_boost_y)
      roc_full <- performance(pred_full, "tpr", "fpr")
      plot(roc_full, col = "red", lwd = 2)
      TPR<-roc_full@y.values
      FPR<-roc_full@x.values
      cutoff<-roc_full@alpha.values
      summary(FPR)
      #revise the list to vector
      FPR<-unlist(FPR)
      TPR<-unlist(TPR)
      FPR_index<-which.min(abs(FPR-0.1))
      req_TPR <- TPR[FPR_index]
      if(req_TPR > max_req_TPR){
        max_req_TPR = req_TPR
        best_nrounds = nrounds_choose[k]
        best_eta = eta_list[j]
        best_depth = depth_list[i]
      }
    }
  }
}

###########################################################################################
################################STEP 6: MODEL FINALIZATION#################################
###########################################################################################
xgbst <- xgboost(data = train_boost_x, label = train_boost_y, 
               max.depth = 13, eta = 0.03, nrounds = 400,  
               objective = "binary:logistic")

preds_bst <- predict(xgbst, valid_boost_x)
pred_full <- prediction(preds_bst, valid_boost_y)
roc_full <- performance(pred_full, "tpr", "fpr")
plot(roc_full, col = "red", lwd = 2)
TPR<-roc_full@y.values
FPR<-roc_full@x.values
cutoff<-roc_full@alpha.values
summary(FPR)
#revise the list to vector
FPR<-unlist(FPR)
TPR<-unlist(TPR)
cutoff<-unlist(cutoff)
#find FPR close to 0.1
FPR_index<-which.min(abs(FPR-0.1))
FPR_index
FPR[FPR_index-27]
TPR[FPR_index-27]
cutoff[FPR_index-27]


###########################################################################################
################################STEP 7: PREDICTION RESULTS#################################
###########################################################################################

preds_bst <- predict(xgbst, airbnb_test)
classifications_perfect <- ifelse(preds_bst >  0.4955689, "YES", "NO")
summary(as.factor(classifications_perfect))
write.table(classifications_perfect, "perfect_rating_score.csv", row.names = FALSE)








