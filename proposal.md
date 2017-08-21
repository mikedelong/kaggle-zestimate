
Udacity Machine Learning capstone project proposal
==================================================
This is a proposal to solve a problem from a Kaggle competition to fulfill the capstone project requirements for the Udacity nanodegree Machine Learning Engineer.

I was working in financial services in 2007, when the 2007-2008 financial crisis began. It began systemically many years earlier, with an escalation in housing prices, which fueled an expectation that home values could only go up. Because of the dislocation the crisis caused in the economy, and because I personally suffered a cash loss that I would prefer not to characterize, I continue to be interested in rational house prices.

“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning.

Zillow Prize, a competition with a one million dollar grand prize, is challenging the data science community to help push the accuracy of the Zestimate even further. Winning algorithms stand to impact the home values of 110M homes across the U.S.

In this million-dollar competition, participants will develop an algorithm that makes predictions about the future sale prices of homes. The contest is structured into two rounds, the qualifying round which opens May 24, 2017 and the private round for the 100 top qualifying teams that opens on Feb 1st, 2018. In the qualifying round, you’ll be building a model to improve the Zestimate residual error. In the final round, you’ll build a home valuation algorithm from the ground up, using external data sources to help engineer new features that give your model an edge over the competition.

Because real estate transaction data is public information, there will be a three-month sales tracking period after each competition round closes where your predictions will be evaluated against the actual sale prices of the homes. The final leaderboard won’t be revealed until the close of the sales tracking period.

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

Of course the other factors listed above do not really appear in our model: if we do not have explicit features for condition of sale, financing conditions, or market conditions then we are not explicitly capturing whatever they contribute to the sale price. It is open for discussion how and whether sale prices capture these factors, or even if Zestimates do the same thing that comparables do.

Also, there are vagaries of the sales process that impact the final sale price that are very difficult to capture and will introduce noise (variance, or errors) into the model: e.g. how busy and therefore how engaged is the seller's agent; how motivated are the buyers and sellers; etc.

Problem statement
-----------------
Given a set of inputs we are looking for the log-error (or the log-residual-error) of the Zestimate, which is the log-error is the log of the Zestimate minus the log of the actual sale price. This log-error will always be nonzero; over the lifetime of Zestimates Zillow has reduced the median margin of error from approximately 14% to approximately 5%.

Zillow is holding a contest on Kaggle to improve Zestimates; the contest consists of two rounds: a public round and a private round. Only the top 100 contestants from the public round will go through to the private round.

For the public round we are looking to minimize the mean absolute error between the predicted log error and the actual log error. For each property we predict the log error for each of up to six periods: each month in the last quarter of the calendar year (October, November, and December) for 2016 and 2017. If a property had no transaction for that month then the property is ignored for that period and does not contribute to the overall score.

At the present time we are looking just for the log-error for Fall 2016, since Fall 2017 is in the future so its transactions are not available.

Datasets and inputs
-------------------
We have two data files as input:
* A training set of properties and home features for 2016: 2985217 properties and 58 features.
* A training set of transactions for 2016: 90275 transactions, including sale prices and dates.

In this competition, Zillow is asking you to predict the log-error between their Zestimate and the actual sale price, given all the features of a home. The log error is defined as
logerror=log(Zestimate)−log(SalePrice)

and it is recorded in the transactions file train.csv. In this competition, you are going to predict the logerror for the months in Fall 2017. Since all the real estate transactions in the U.S. are publicly available, we will close the competition (no longer accepting submissions) before the evaluation period begins.
Train/Test split

    You are provided with a full list of real estate properties in three counties (Los Angeles, Orange and Ventura, California) data in 2016.
    The train data has all the transactions before October 15, 2016, plus some of the transactions after October 15, 2016.
    The test data in the public leaderboard has the rest of the transactions between October 15 and December 31, 2016.
    The rest of the test data, which is used for calculating the private leaderboard, is all the properties in October 15, 2017, to December 15, 2017. This period is called the "sales tracking period", during which we will not be taking any submissions.
    You are asked to predict 6 time points for all properties: October 2016 (201610), November 2016 (201611), December 2016 (201612), October 2017 (201710), November 2017 (201711), and December 2017 (201712).
    Not all the properties are sold in each time period. If a property was not sold in a certain time period, that particular row will be ignored when calculating your score.
    If a property is sold multiple times within 31 days, we take the first reasonable value as the ground truth. By "reasonable", we mean if the data seems wrong, we will take the transaction that has a value that makes more sense.

File descriptions

    properties_2016.csv - all the properties with their home features for 2016. Note: Some 2017 new properties don't have any data yet except for their parcelid's. Those data points should be populated when properties_2017.csv is available.
    properties_2017.csv - all the properties with their home features for 2017 (will be available on 10/2/2017)
    train_2016.csv - the training set with transactions from 1/1/2016 to 12/31/2016
    train_2017.csv - the training set with transactions from 1/1/2017 to 9/15/2017 (will be available on 10/2/2017)
    sample_submission.csv - a sample submission file in the correct format

Data fields

    Please refer to zillow_data_dictionary.xlsx


Solution statement
------------------
We will use one or more models to predict the log-error of the Zestimates

Benchmark model
---------------

Evaluation metrics
------------------
We will produce a result that qualifies for the leaderboard. Alternately, we will do something that we can verify directly.

Outline of project design
-------------------------


    The project's domain background — the field of research where the project is derived;
    A problem statement — a problem being investigated for which a solution will be defined;
    The datasets and inputs — data or inputs being used for the problem;
    A solution statement — a the solution proposed for the problem given;
    A benchmark model — some simple or historical model or result to compare the defined solution to;
    A set of evaluation metrics — functional representations for how the solution can be measured;
    An outline of the project design — how the solution will be developed and results obtained.
