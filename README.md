# Project-5

Udacity Project 5 completed January 2016


Project Outline
The goal of the project is to use machine learning to predict which individuals at Enron committed fraud in the early 2000’s. The dataset includes data reguarding payroll, financial and emails, totaling 21 different features. The most important feature to our project is “poi”, which stands for person of interest. These are individuals who were indicted or settled without admitting guilt during the investigation of Enron. This is going to be the output of the classifiers being tested in the project. The dataset is fairly small, just 18 observations are POIs and 127 are non-POIs and a total of 145.

During our first look at the data we found one major outlier, which was the “TOTAL” column in our dataset. This was removed as it really had no relation to the goals of the project. The other major outliers were found to be Jeffrey Skilling and Ken Ley to prominent Enron execuitives who featured heavily in the federal indictment of the company. These are valid data points and therefore remain in the dataset for the project.

Features
I used the following four features in my POI identifier: loan_advances(30.7), deferred_income(10.8), expenses(9.2) and from_this_person_to_poi(10.7). The selection process I used was SelectKBest in sklearn, the bracketed numbers next to the feature names are the feature scores from this method. The number of features(the “k” parameter in SelectKBest) were chosen purely from observing the algorithm’s results, both using the Naïve Bayes and Decision tree algorithms. I attempted to strike a balance between the highest performing accuracy, recall and precision scores while testing out algorithms with 2, 4, 6 and 8 feature training sets. Furthermore, rescaling was not required using these two algorithms as there is no tradeoff between the features.

I tried to create two features, poi_ratio and total_messages. POI ratio was the percentage of emails to or from a person of interest sent/received out of the total number of emails sent or received. I thought the people with the most dealings with POI’s would be the most likely to be a POI themselves. Total messages was simply the total number of emails sent and received by an individual. I also hypothesized that keeping the fraud under wraps might require a large number of emails and greater communication from POI in general. It turned out that SelectKBest selected neither of these added features. Their feature scores were 0.61 for total messages and 0.64  for POI ratio.

Classifier
 I used a Decision Tree classifier over the Naïve Bayes classifier. All evaluation scores in tester.py were significantly lower in the Naïve Bayes besides the recall score, which was 78%, however the Precision score was a mere 15%. This was a tradeoff I was uncomfortable with as the algorithm classified so many non-POIs as POIs (false positives).

Parameters were very important in the case of using decision trees. In general, parameter tuning is important because it allows the user to manipulate the chosen type of algorithm. Often parameters put limits on how specifically the data is trained to the trends of the sample data. If the classifier follows the every fluctuation of the sample, overfitting can be an issue and the models performance will be negatively affected. 

The main parameter tuned in the project was min_samples_split, which is the minimum samples required to split an internal node. If this number is too high the model gets to be too generalized and does not pick up subtleties in the dataset. While if too large, the parameter causes overfitting and is too sensitive to outlier data points. Other parameters that were adjusted were the criterion and splitter parameters. Criterion dictates the way the classifier measures the quality of the split, it turned out that the non-default setting of entropy increased the recall and precision scores by around 4% holding all other parameters constant. Splitter adjusted the strategy used to split at each node. I found that the default setting of best was much more effective than a random split decision by an average of 15% across the two metrics.

Results

I used recall and precision score as two major metrics in evaluating performance, my average recall was 0.29 and   precision was 0.32, my optimized classifier ended up being 0.37 and 0.44. Recall score represents our effectiveness at identifying POIs, not taking into account people that are wrongfully identified as POIs. Precision score represents how precise the algorithm is at identifying POIs, where false positives negatively affect the precision score.
It makes sense to aim to maximize your recall score more than your precision score as it would theoretically identify the most POIs, allowing for the authorities to weed out innocent people wrongly accused, while minimizing the number of POIs not identified(false negatives).



