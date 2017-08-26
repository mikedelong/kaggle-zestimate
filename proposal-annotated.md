
Udacity Machine Learning capstone project proposal (annotated)
==================================================
This is a proposal to solve a problem from a Kaggle competition to fulfill the capstone project requirements for the Udacity nanodegree Machine Learning Engineer.

I was working in financial services in 2007, when the 2007-2008 financial crisis began. It began systemically many years earlier, with an escalation in housing prices, which fueled an expectation that home values could only go up. Because of the dislocation the crisis caused in the economy, and because I personally suffered a cash loss that I would prefer not to characterize, I continue to be interested in rational house prices.

Domain background
-----------------
For most people who own their own house their house will be one of the largest investment decisions they will ever make. Historically when a house is put up for sale the asking price is justified by looking at a handful of "comparables," determined during the real estate appraisal process. The traditional five factors for determining a comparable are
* Conditions of sale
* Financing Conditions
* Market Conditions
* Locational comparability
* Physical comparability

These five factors are typically presented indirectly; instead the seller's agent presents MLS listings for a handful of properties and the reader is left to imagine what light they actually shed on the seller's property. When a potential buyer is looking at a package consisting of an asking price and a list of comparables it is difficult to see what impact each factor has on the asking price.

Comparables are also relatively illiquid: they do not tell us what the asking price would have been a year ago, or more importantly what a reasonable asking price would be a year from now.

One alternative to a short list of comparables is the Zillow Estimate (Zestimate), which uses a proprietary machine learning algorithm to estimate the final sale price based (mostly) on publicly available information. Of the five factors listed above, Zestimates focus mostly on locational and physical comparability. The assumption that we can depend entirely on public information will allow us to give an estimate for the sale price at any time, and the price should be fairly liquid, meaning that each time we update the model based on new information (e.g. tax assessments, sale prices) we can update the Zestimate.

It is worth noting that the Zestimate represents a big step forward in empowering buyers and sellers in what is generally an opaque and even frightening life experience, because Zestimates are available for any property, because they are fairly good predictions of the final sale price, and because they are available at no cost to the public.

Of course the other factors listed above do not really appear in our model: if we do not have explicit features for condition of sale, financing conditions, or market conditions then we are not explicitly capturing whatever they contribute to the sale price. It is open for discussion how and whether the source data for Zestimates capture these factors, or even if Zestimates do the same thing that comparables do.

Also, there are vagaries of the sales process that impact the final sale price that are very difficult to capture and will introduce noise (variance, or errors) into the model: e.g. how busy and therefore how engaged is the seller's agent; how motivated are the buyers and sellers; etc.

Zillow would like to improve Zestimates if at all possible, and for that reason they have sponsored the Kaggle Zillow Prize (https://www.kaggle.com/c/zillow-prize-1) and provided data to use as part of the prize contest (https://www.kaggle.com/c/zillow-prize-1/data).

This is a problem that is an obvious candidate for a machine learning solution. In fact the Udacity Machine Learning nanodegree includes a project that uses a linear regression to model Boston home prices. This is also a problem with academic solutions at the undergraduate or occasionally graduate level. Academic solutions to this problem include:
* Jingyi Mu, Fang Wu, and Aihua Zhang: Housing Value Forecasting Based on Machine Learning Methods, where the authors solve the problem using support vector machines and partial least squares: https://www.hindawi.com/journals/aaa/2014/648047/
* Aaron Ng: Machine Learning for a London Housing Price Prediction Mobile Application, where the author used a host of regression methods: http://www.doc.ic.ac.uk/~mpd37/theses/2015_beng_aaron-ng.pdf
* Nissan Pow, Emil Janulewicz, and Liu (Dave) Liu: Applied Machine Learning Project 4 Prediction of real estate property prices in Montreal, where the authors use Support Vector Regression, k-Nearest Neighbors (kNN), and Random Forest Regression (RFR), along with a kNN/RFR ensemble method: http://rl.cs.mcgill.ca/comp598/fall2014/comp598_submission_99.pdf

Reviewer comment:
  *This section could also be used to provide an intro to the algorithms and techniques you plan to use and why you chose them.*
Problem statement
-----------------
Given a set of inputs we are looking for the log-error (or the log-residual-error) of the Zestimate, which is the log-error is the log of the Zestimate minus the log of the actual sale price. This log-error will always be nonzero; over the lifetime of Zestimates Zillow has reduced the median margin of error from approximately 14% to approximately 5%.

Zillow is holding a contest on Kaggle to improve Zestimates; the contest consists of two rounds: a public round and a private round. Only the top 100 contestants from the public round will go through to the private round.

For the public round we are looking to minimize the mean absolute error between the predicted log error and the actual log error. For each property we predict the log error for each of up to six periods: each month in the last quarter of the calendar year (October, November, and December) for 2016 and 2017. If a property had no transaction for that month then the property is ignored for that period and does not contribute to the overall score.

At the present time we are looking just for the log-error for Fall 2016, since Fall 2017 is in the future so its transactions are not available.

This is primarily a regression problem; our dependent variables are continuous. We have 58 independent variables, but we can put them in three broad categories:
1. Location data, which tells us where the house is
2. House intrinsic data, which tells us things about the structure, its features, or the lot on which it sits
3. Tax data, which tells us the tax assessment value, and where appropriate if the property is tax delinquent and if so which year it became delinquent

Datasets and inputs
-------------------
We have two data files as input:
* A training set of properties and home features for 2016: 2985217 properties and 58 features.
* A training set of transactions for 2016: 90275 transactions, including sale prices and dates.

The properties are all from three counties in California: Los Angeles, Orange, and Ventura that sold in 2016. The training data has a full set of transactions (dates and prices) from before October 15, 2016 and some transactions after October 15, 2016.

The competition artifacts also include a data dictionary and a sample submission.

All of these files are available at the Zillow Prize data page: https://www.kaggle.com/c/zillow-prize-1/data

The raw or native features are as follows:

|Feature	|Description|
|---|---|
|airconditioningtypeid	 |Type of cooling system present in the home (if any)|
|architecturalstyletypeid	 |Architectural style of the home (i.e. ranch, colonial, split-level, etc…)|
|basementsqft	 |Finished living area below or partially below ground level|
|bathroomcnt	 |Number of bathrooms in home including fractional bathrooms|
|bedroomcnt	 |Number of bedrooms in home|
|buildingqualitytypeid	 |Overall assessment of condition of the building from best (lowest) to worst (highest)|
|buildingclasstypeid	|The building framing type (steel frame, wood frame, concrete/brick) |
|calculatedbathnbr	 |Number of bathrooms in home including fractional bathroom|
|decktypeid	|Type of deck (if any) present on parcel|
|threequarterbathnbr	 |Number of 3/4 bathrooms in house (shower + sink + toilet)|
|finishedfloor1squarefeet	 |Size of the finished living area on the first (entry) floor of the home|
|calculatedfinishedsquarefeet	 |Calculated total finished living area of the home |
|finishedsquarefeet6	|Base unfinished and finished area|
|finishedsquarefeet12	|Finished living area|
|finishedsquarefeet13	|Perimeter  living area|
|finishedsquarefeet15	|Total area|
|finishedsquarefeet50	 |Size of the finished living area on the first (entry) floor of the home|
|fips	 |Federal Information Processing Standard code |
|fireplacecnt	 |Number of fireplaces in a home (if any)|
|fireplaceflag	 |Is a fireplace present in this home? |
|fullbathcnt	 |Number of full bathrooms (sink, shower + bathtub, and toilet) present in home|
|garagecarcnt	 |Total number of garages on the lot including an attached garage|
|garagetotalsqft	 |Total number of square feet of all garages on lot including an attached garage|
|hashottuborspa	 |Does the home have a hot tub or spa|
|heatingorsystemtypeid	 |Type of home heating system|
|latitude	 |Latitude of the middle of the parcel multiplied by 1,000,000|
|longitude	 |Longitude of the middle of the parcel multiplied by 1,000,000|
|lotsizesquarefeet	 |Area of the lot in square feet|
|numberofstories	 |Number of stories or levels the home has|
|parcelid	 |Unique identifier for parcels (lots) |
|poolcnt	 |Number of pools on the lot (if any)|
|poolsizesum	 |Total square footage of all pools on property|
|pooltypeid10	 |Spa or Hot Tub|
|pooltypeid2	 |Pool with Spa/Hot Tub|
|pooltypeid7	 |Pool without hot tub|
|propertycountylandusecode	 |County land use code i.e. its zoning at the county level|
|propertylandusetypeid	 |Type of land use the property is zoned for|
|propertyzoningdesc	 |Description of the allowed land uses (zoning) for that property|
|rawcensustractandblock	 |Census tract and block ID combined - also contains blockgroup assignment by extension|
|censustractandblock	 |Census tract and block ID combined - also contains blockgroup assignment by extension|
|regionidcounty	|County in which the property is located|
|regionidcity	 |City in which the property is located (if any)|
|regionidzip	 |Zip code in which the property is located|
|regionidneighborhood	|Neighborhood in which the property is located|
|roomcnt	 |Total number of rooms in the principal residence|
|storytypeid	 |Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.).|  
|typeconstructiontypeid	 |What type of construction material was used to construct the home|
|unitcnt	 |Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)|
|yardbuildingsqft17	|Patio in  yard|
|yardbuildingsqft26	|Storage shed/building in yard|
|yearbuilt	 |The Year the principal residence was built |
|taxvaluedollarcnt	|The total tax assessed value of the parcel|
|structuretaxvaluedollarcnt	|The assessed value of the built structure on the parcel|
|landtaxvaluedollarcnt	|The assessed value of the land area of the parcel|
|taxamount	| The total property tax assessed for that assessment year|
|assessmentyear	| The year of the property tax assessment |
|taxdelinquencyflag	| Property taxes for this parcel are past due as of 2015|
|taxdelinquencyyear	| Year for which the unpaid property taxes were due |

We will be attempting to use all of these features, with some exceptions:
* We know that the parcel ID is unique to each property and is useful only for identifying properties that sold multiple times during the period of interest, so we will use it to filter out multiple-sale properties during the data cleansing process but we will not use it to train the model.
* Some features may be redundant: e.g. the census tract and block are identified in two features; the FIPS and the county ID represent the same concept; the bathroom count and the calculated bath number are nearly identical: one of them is a raw number, while the other represents Zillow's adjusted amount. We will need to pick through these on a case-by-case basis
* If our chosen model can give us a feature significance we may filter out features it considers insignificant for performance reasons.

We only have log-error data for 90275 properties, and we will be training with a subset of those properties, subject to:
* Filtering on outliers; we don't want our model to reproduce the largest errors in the training data
* Filtering on date of sale; we want to train on data that is outside the period of interest and predict the log-error for properties that sold in the period of interest.

We will be doing some cross-validation with various folds and our test set will be the properties for which we have no transaction data, subject to some filtering for bad data.

Solution statement
------------------
We will use one or more models to predict the log-error of the Zestimates.

  When you resubmit, please be sure to give a quick overview of the algorithms and techniques you want to try. For instance, be sure to mention any pre-processing, unsupervised, supervised or other machine learning techniques you want to try. No need to go into detail about how you'll implement them...this should just be a quick summary of your methodology.

Benchmark model
---------------
Because this is an open Kaggle competition we have available to us more than one baseline model. Oleg Panichev has published three different baseline models:
* A simple mean model; this model just takes the mean of the log-error in the training data and uses it as the guess for the log-error of all of the test data.
* A brute force model; this model is a single-value model just like the mean model above, but it chooses among a range of plausible guesses the one that minimizes the mean absolute error.
* A brute force model with a monthly seasonality correction; this brute force variant separates the transactions according to their month of sale and chooses among a range of plausible guesses the one that minimizes the mean absolute error over the test data for that month.

The best of these has a mean absolute error of about 6.53%.

Reviewer note:
*In scikit-learn there's also a DummyRegressor estimator that you can customize with any additional prediction strategies you want to devise yourself.*

Evaluation metrics
------------------
We will produce a result that qualifies for the leaderboard and that produces a score that is better than the best of the baseline models. For the leaderboard submissions are evaluated based on the mean absolute error of the predicted log error as compared to the actual log error, or *log(error) = log(Zestimate) - log(Actual sale price)*. It is worth noting that the best score on the public leaderboard as of this date is about 6.402%, so the margins here are pretty slim.

Also we will produce other charts and diagrams that would help the lay reader understand some aspects of what the model is doing. The details of these have yet to be determined.

Reviewer note: *make sure your final report also includes the actual equation for calculating MEA.*

Outline of project design
-------------------------
For this project we will
* Acquire the data from the Kaggle website
* Clean up the data to remove outliers and cope with missing data as appropriate
  * Inferring missing data where appropriate
  * Parsing transaction dates into month and year and possibly day of the week
  * Adding one-hot encoding for Boolean data
  * Consolidating columns where appropriate
  * Removing redundant data
* Build and train a model
  * As much as I would like to do so I don't think simple linear regression will do much here, because the data is a mix of Boolean, enumeration, and real values. I'd love to use a linear regression model because the coefficients it produces are easy to explain
  * I will probably be using some sort of boosting and bagging or ensemble model; these models seem to do well with data of mixed types and with missing data.
  * I will probably need to do grid search to explore the model hyperparameter space
  * I will also probably need to iterate based on model outputs, such as validation data error metrics and feature importance metrics
* Produce many submissions for the Zillow Prize
* Produce documentation of a leaderboard score that is better than the baseline model
* Produce a report that discusses the engineering decisions we made and the conclusions we drew
