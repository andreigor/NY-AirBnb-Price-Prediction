## Introduction

Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. 
This dataset describes the listing activity and metrics in NYC, NY for 2019. In this project, we would like to recommend a reasonable price for its users, 
in case those users are going to list proprieties to be rented at their platform, they already have a lot of useful data from previous and current customers.

Beyond just recomending a reasonable price, we are also interested in learning about what makes listings have a higher or lower price, 
so its possible to provide a better exprience to different customer groups.

In this project, we've performed an extensive EDA in order to learn about the characteristics that compose a listing price. Then, based on the results
of this EDA, we've performed a Data Engineering process, with custom transformers to clean the existing features, create new ones and prepare the data to be 
used by ML estimators. We've made test with 3 different models of estimatores (Linear Regression, Random Forest and Gradient Boosting Trees). We also used two
different set of features in each model. The metric of evaluation was the r² score. At the end, the Random Forest Model was able to perform better, with a r² score
of 0.62

## Exploratory Data Analysis

The dataset of AirBnB listings has over 49000 rows and 17 columns. There are a mix of categorical and numerical columns. In the [Full EDA](https://github.com/andreigor/NY-AirBnb-Price-Prediction/blob/master/Exploratory%20Data%20Analysis/Full_EDA.ipynb),
we've discussed about all of the columns used to train the ML models. In this solution overview, we will highlight the most important variables.

|   id |                                             name | host_id |   host_name | neighbourhood_group | neighbourhood | latitude | longitude |       room_type | price | minimum_nights | number_of_reviews | last_review | reviews_per_month | calculated_host_listings_count | availability_365 |
|-----:|-------------------------------------------------:|--------:|------------:|--------------------:|--------------:|---------:|----------:|----------------:|------:|---------------:|------------------:|------------:|------------------:|-------------------------------:|-----------------:|
| 2539 |               Clean & quiet apt home by the park |    2787 |        John |            Brooklyn |    Kensington | 40.64749 | -73.97237 |    Private room |   149 |              1 |                 9 |  2018-10-19 |              0.21 |                              6 |              365 |
| 2595 |                            Skylit Midtown Castle |    2845 |    Jennifer |           Manhattan |       Midtown | 40.75362 | -73.98377 | Entire home/apt |   225 |              1 |                45 |  2019-05-21 |              0.38 |                              2 |              355 |
| 3647 |              THE VILLAGE OF HARLEM....NEW YORK ! |    4632 |   Elisabeth |           Manhattan |        Harlem | 40.80902 |  -73.9419 |    Private room |   150 |              3 |                 0 |        null |              null |                              1 |              365 |
| 3831 |                  Cozy Entire Floor of Brownstone |    4869 | LisaRoxanne |            Brooklyn |  Clinton Hill | 40.68514 | -73.95976 | Entire home/apt |    89 |              1 |               270 |  2019-07-05 |              4.64 |                              1 |              194 |
| 5022 | Entire Apt: Spacious Studio/Loft by central park |    7192 |       Laura |           Manhattan |   East Harlem | 40.79851 | -73.94399 | Entire home/apt |    80 |             10 |                 9 |  2018-11-19 |              0.10 |                              1 |                0 |

### Univariate Analysis

When dealing with numerical values, it is very important to check the distribution of these values. Specially when using models such as linear regression or neural networks.
In here, we will show that there are a lot of outliers values in the numerical columns. There are also skewed distributions, indicating the need to use a logarithm
transformation.

#### Price

As we can see, the price distribution is very skewed to the left, due to the presence of very expensive rooms. 
75% of the rooms have a price below 150.95 400. We can also see that there are rooms that have an outlier price, reaching up to $ 10,000. 
The price variable also does not have a normalized shape. 
To use this feature in some future Machine Learning model it would be interesting to apply a logarithmic function or square root with the intention of normalizing the values.

![price](https://user-images.githubusercontent.com/44973832/101644619-2e91dc00-3a14-11eb-9420-695545688989.jpeg)

#### Minimum Nights

As we can see, a large number of rooms have a minimum number of nights less than 10 (85%). 
There is also an interesting peak in rooms where you can only rent for a minimum period of 30 days (equivalent to one month). 
This indicates a special type of air bnb listing. There are also outliers (1000 minimum nights) values that do not match the reality of AirBnb.

![nights](https://user-images.githubusercontent.com/44973832/101644949-91837300-3a14-11eb-9ee7-c3db49123679.jpeg)

#### Latitude and Longitude

Even though there are a few outliers in the latitude and longitude, once they're removed, there a fairly good distributions of listings around
the NY City. We can also note a concentration of listings in the Manhattan area.

![lat](https://user-images.githubusercontent.com/44973832/101645340-0787da00-3a15-11eb-8b58-3873b210c48c.jpeg)


### Bivariate Analysis

#### Price and Availability of Neighbourhood Groups

![price_avail](https://user-images.githubusercontent.com/44973832/101645852-972d8880-3a15-11eb-886c-a09a998a4b3b.jpeg)


By looking at the mean and the distribution of the price among the neighbourhood groups, we can see that Manhattan stands out. It is the neighbourhood with the most expensive rooms, with a average price of almost $ 200. Brooklyn is the neighbourhood that is most similar to Manhattan with an average price of $ 124. The other neighbourhoods present a very similar price distribution. They seem to have similar room prices among them.

When looking at the availability distributions, we again confirm that Brooklyn and Manhattan have a similar pattern, while Queens, Staten Island and the Brox have a similar behavior among them. In the first group (Manhattan and Brooklyn) we can see a large number of rooms that are available less then 80 days a year, even tough though the prices in those areas is higher. The secund group, in their turn, have a bigger availability of rooms throughout the year. which means that they are not so wanted by consumers.

Since New York is a very rich city, with a very large tourism, this leads us to believe that Manhattan and Brooklyn have high standard rooms that highly sought after by guests. The Bronx, Queens and Staten Island have more financial acessible rooms that are available for a longer portion of the year.

We can also visualize the price of the rooms by their location in a more immersive way, by looking at the map of New York.

![price_manh](https://user-images.githubusercontent.com/44973832/101645669-63eaf980-3a15-11eb-9398-4cfc9eb25249.jpeg)

![avail](https://user-images.githubusercontent.com/44973832/101645687-6a797100-3a15-11eb-83d9-2c76434dd3b9.jpeg)


## Data Engineering

### Pipeline

The Data Engineering phase has the objective of applying all the transformations necessary to the dataset in order to feed it into a ML estimator, including
the stages of feature engineering, feature selection and cleaning the dataset.
In the Figure above, we can see all the stages that were applied in this Pipeline.

![Pipeline (1)](https://user-images.githubusercontent.com/44973832/101668847-d8329680-3a2f-11eb-87e5-f23da79d71b0.jpeg)


This process is also called the pre processing of the data. The Pre Processing section is one of the most important sections of the work. 
Here we will use the techniques to treat outliers and encode the categorical variables. 
We will also create a pipeline object to handle all of the transformations applied to the dataset.

- **Cleaning the data**: As we've seen in the scatter plots in the exploratory data analysis, there are a few outliers that need to be removed from the dataset. This stage deals with these values by defining an lower and upper boundary values for each of the columns. 
The lower and upper boundaries were chosen manually, using an iteractive process.

- **Securing Categorical Distributions**: It is important to assure that the values of the categorical columns on the test set were previously seen by the train
dataset. Therefore, we added a stage to check if these categorical values of the test set were on the training set. 

- **Imputing Missing Values**: This stage performs the imputation of the columns with values provided by the user. In this dataset, there are very few values missing.
The only columns that needed imputing were "name" and "number_of_reviews".

- **Logarithm Transformer**: As we've seen on the EDA, a few numerical columns are very skewed to the left. Applying a logarithm transformer will help the estimators.

- **Cleaning Text Features**: In this dataset there is a important column containing a text description of the AirBnB listing. This feature can be used to help the ML
estimators, but text is a very unstructured type of data. Therefore, it is important to perform a cleaning stage, removing stop words, unnecessary characteres and
tokenizing.

- **Bag of Words Vectorization**

- **OHE + Vector Assembler**


### Testing Pipeline

An important step of every project is design suitable test for the application. In this case, we've defined test cases for each of the custom transformers applied
to the pipeline. The full tests can be seen in the [QA section of this repository](https://github.com/andreigor/NY-AirBnb-Price-Prediction/blob/master/Quality%20Assurance/QA.ipynb).

## Machine Learning Results

The metric of evaluation of the estimators was the r² score, which corresponds to the proportion of the variance in the dependent variable that 
is predictable from the independent variable(s). **Three different models were developed, with two different sets of features. The first set of features
does not contains the text features and is used as a baseline model. The second set of features contains the text features in a bag of words vectorization.**

In all of the models a **grid search** was performed to find the best hyperparameters. All of the models were also trained using 
a cross validation with 5 folds. The best results of the models can be seen on the table below.

|                              | Linear Regression | Random Forest | Gradient Boost |
|------------------------------|-------------------|---------------|----------------|
| Baseline Features (r² score) | 0.588             | 0.610         | 0.598          |
| Text Features (r² score)     | 0.638             | 0.620         | 0.608          |

As we can see, the model that performed the best was the linear regression, with the following hyperparameters: 10 iterations of training and an regularization parameter
of 0.01.








