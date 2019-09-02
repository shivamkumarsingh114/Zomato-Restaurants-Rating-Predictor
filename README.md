# Zomato-Restaurants-Rating-Predictor
A ML Model created using XGBoost. This model predicts ratings of restaurant listed on Zomato with over 90% accuracy. It also identifies the features that people feel are the most Important. 
# Zomato Restaurants Rating Predictor

## Introduction

**Problem Statement:** Predicting the ratings of restaurants available on Zomato’s Bangalore page and find
correlation between different features and ratings.

We aim to find correlation between different features and ratings. This will help restaurants as they will
have better understanding of features that are important for customers. This will also give an insight on
geolocation benefits for a restaurant.


For customers like us, this will give an insight as to where are most of the restaurants located and which
restaurant has the features that matters the most to customers.

### Scraping Dataset from Zomato and preprocessing it.

We scraped Zomato’s Bangalore website to extract desired features that are available on a restaurant
card on Zomato

```
We got all the features for total of 6000 restaurants and created a robust dataset containing all the
information available on the card.
```
```
We processed the data by removing all the restaurants which are void of ratings as they are new with
very less votes.
```

We can see there is a strong correlation between Featured in categories and ratings, we can also see a
strong correlation between votes and ratings as expected.


**We will now use Xgboost to train on 70% dataset and test on 30%**

From Xgboost we got repeated accuracy of 90-91%

We can see that most important features are: **_Votes, Cost, No. of cuisines offered, and no. of categories
that a restaurant has been featured in._**
