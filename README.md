# Covid Sentiment Analysis using Twitter API

# Introduction
This project does a sentiment analysis of covid vaccination data using twitter's api. The project is written in Python and uses a couple of python packages like Tweepy, NLTK, SciKit Learn, WordCloud, etc to query the Twitter API and do a sentiment analyis on the tweets.

Some of the keywords that have been used are: Covid Vaccine, Covid Booster.

# Implementation Details
Following are the steps that are done to conduct a sentiment analyis:

1. Choosing the keywords for live search: (Covid booster).
2. Running the live search using Twitter API by leveraging tweepy.
3. Using Sentiment Analyzer to see if the tweets are positive, negative or neutral
4. Creating visualizations to categorize the tweets as positive, negative or neutral.
5. Creating a word cloud for different categories of the tweets.
6. Cleaning up the tweets to remove punctuation, stopwords and stemming.
7. Showing the most user words in the search.
8. Creating bigrams & trigrams.

# Running the code

To run the code, simply clone the repo and open the main.py file in Visual Studio Code. Make sure to restore all the packages that are mentioned in the header section of the file and resolve any errors that are logged on the terminal. Once that is done, make sure to request a Twitter API key and replace it in the code where it's being referred. After doing that, simply running the code will ask for keywords in the terminal and the number of tweets that you want to analyze to do a sentiment analysis.

# Data Visualizations

![Screen Shot 2021-12-20 at 9 08 12 AM](https://user-images.githubusercontent.com/60200344/146779850-4ab67536-bf16-4d14-9f18-75a34f198768.png)

![Screen Shot 2021-12-20 at 9 08 18 AM](https://user-images.githubusercontent.com/60200344/146779859-56e2d600-e23d-4b33-9e7e-e562f3f6b6a5.png)

![Screen Shot 2021-12-20 at 9 08 25 AM](https://user-images.githubusercontent.com/60200344/146779869-1c31b5df-434a-4e2d-af24-92824d019d02.png)

# Most common used words

booster  11
vaccin    4
shot      3
jab       3
got       2
front     2
omicron   2
die       2
death     2
data      2


# Bigrams

[('booster jabs', 2), ('blood clots', 1), ('clots vaccine', 1), ('vaccine realize', 1), ('won save', 1), ('save nhs', 1), ('nhs omicron', 1), ('omicron booster', 1), ('jabs prime', 1), ('prime minister', 1), ('minister really', 1), ('really ought', 1), ('ought telling', 1), ('telling people', 1), ('people cut', 1), ('think tank', 1), ('tank boomer', 1), ('boomer maybe', 1), ('maybe 5th', 1), ('5th booster', 1)]

# Trigrams

[('blood clots vaccine', 1), ('clots vaccine realize', 1), ('won save nhs', 1), ('save nhs omicron', 1), ('nhs omicron booster', 1), ('omicron booster jabs', 1), ('booster jabs prime', 1), ('jabs prime minister', 1), ('prime minister really', 1), ('minister really ought', 1), ('really ought telling', 1), ('ought telling people', 1), ('telling people cut', 1), ('think tank boomer', 1), ('tank boomer maybe', 1), ('boomer maybe 5th', 1), ('maybe 5th booster', 1), ('5th booster ll', 1), ('booster ll start', 1), ('ll start question', 1)]

# Further Improvements

1. Deploy to the web and give the user run the query using a form based input {predetermined keywords} rather than running this via terminal.
2. Use multiple Twitter API accounts to resolve the tweepy limitations.
3. Connect the Twitter feed to Firebase Database to save the information & use it again to do a more accurate sentiment analysis for subsequent runs. 

