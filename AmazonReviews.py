#!/usr/bin/env python
# coding: utf-8

# In[2]:


from bs4 import BeautifulSoup
import requests
import re
import math
from datetime import datetime
import statistics
import os
import scipy.stats as st
import random
import string
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import nltk
nltk.download(["stopwords","punkt","names"])
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.patches as mpatches


# In[3]:


#Create the cookies and headers
cookie = {'session-id':'261-0624820-8189244',
        'ubid-acbuk':'257-5015255-6933932',
        'x-acbuk':"SHANeonOJaheMLxqVZ4msU4f4R@EmL3x",
        'at-acbuk':'Atza|IwEBIEnpF7WXlwtQsRyGZ7j2bTSRbzTBwDlAnOTyxuvo7Hvrhrn2dyp63ewuYjG4ia_InUZLsBocBwWU_fMQcC175R50rYypI0cd1zwygQibh3YcTNoC4w4rzORUj9CjWEky33VlVWjR03SMBLThsL1ThlCNlAbG2SSWBqYbpuYenZPGnGzsyRyykCwK-lWg_WvRnh8-ZbWAbCMCZnj4tQLVOdZD',
        'sess-at-acbuk':'"aCh5jaU0vQL4yngxb40ZUs3evR9B3Thcl34Fyde8oG8="',
        'sst-acbuk':'Sst1|PQFNcV_AOW-PykmKjih9sdAzCYa21f7Wv9FmpIdzxnMwOp3RMKf9c5iGeatKfYULnwIwkDfn_otu-l7g3xLyxdFLuJWBpuIo23QnEpM21f00qFJD1SB3ocTbbSgowpR3HTSLghqfkiGttlxXKUjjI3jgntN-q2sJuWdSkHYdQTYLL_4f2UAf6oENeuPJ9eXQfhi1TTmSBJx4GEvELHlMhuNCz7hGEB9k_JU8fP1qs6bAQnQIQ5v8m7UTGFawlDa7JwoSbr-aojXr8KbJA8K981Kxpk8mSy3tsDg77AiBzOsMY6M',
        'i18n-prefs':'GBP',
        'lc-acbuk':'en_GB',
        'av-timezone':'Europe/London',
        'session-token':'"+VfkDfO9fgKFPdKHAQnlKJz/jM7u+awmBduH5/PFl5ztXolz282+s/gJig/ma6wa199MlP38l+UB34qpiitev5XFOb5Tn7dRH5cnTCHuVrWu5NwEOQY9Z9PjAgXHSrTMNYRsNhf/5ehxk6RvDP/k62m3wvdxLv3FMQUkEy2kkqxkUhpJHuloo7YsSgiApoNh2M3qQykGoL1pnMX29K252A=="',
        'session-id-time':'2082758401l'
        }

header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    }


# In[4]:


#Find all URLs of pages
def getURLs(page1):
    urls = []
    page_stem = page1[:-1]
    starting_url = page1
    page = requests.get(starting_url,headers = header)
    print(page)
    soup = BeautifulSoup(page.content)
    num_reviews = soup.findAll("div",{'data-hook':'cr-filter-info-review-rating-count'})
    num_pages = 0
    for result in num_reviews:
        result_string = str(result)
        result_string_clean = re.sub('<[^>]+>', '', result_string)
        result_stripped = result_string_clean.strip()
        result_substring = result_stripped.split("| ")[1].split(" ")[0]
        num_pages = math.ceil(int(result_substring)/10)

    for i in range(num_pages):
        url = page_stem + str(i)
        urls.append(url)
    return urls


# In[5]:


def getReviewsFromURL(url):
    page = requests.get(str(url),cookies=cookie,headers = header)
    soup=BeautifulSoup(page.content)
    reviews = soup.findAll("span", {'data-hook':"review-body"})
    dates = soup.findAll("span", {'data-hook':"review-date"})
    return reviews,dates


# In[6]:


def getAllReviews(urls):
    all_reviews = []
    for url in urls:
        reviews,dates = getReviewsFromURL(url)
        for review,date in zip(reviews,dates):
            review_string = str(review)
            date_string = str(date).split("on ",1)[1][:-7]
            review_string_clean = re.sub('<[^>]+>', '', review_string)
            review_string_cleaner = re.sub(r'http\S+', '', review_string_clean)
            date_string_clean = re.sub('<[^>]+>', '', date_string)
            review_stripped = review_string_cleaner.strip()
            date_stripped = date_string_clean.strip()
            date_formatted = datetime.strptime(date_stripped, "%d %B %Y")
            all_reviews.append((review_stripped,date_formatted))
    return all_reviews


# In[7]:


def tokenizeReviews(product_reviews):
    tokenized_reviews = []
    stopwords = nltk.corpus.stopwords.words("english")

    for review,date in tqdm(product_reviews):
        review = review.replace("n't","not")
        tokenized_review  = [word.lower() for word in word_tokenize(review) if word.lower() not in stopwords and word.lower() not in string.punctuation and word.lower() not in ["’","``","...","....","''","amp"]]
        if len(tokenized_review) != 0:
            tokenized_reviews.append((tokenized_review,date))
    return tokenized_reviews


# In[8]:


def tokenizeWords(tokenized_reviews):
    tokenized_words = []

    for review,date in tokenized_reviews:
        for word in review:
            tokenized_words.append(word)
    
    return tokenized_words


# In[9]:


def createTargetDirectory(product):
    newpath = "Figures/" + product
    if not os.path.exists(newpath):
        os.makedirs(newpath)


# In[10]:


#Overall frequency analysis
def singleWordCount(tokenized_reviews):
    tokenized_words = tokenizeWords(tokenized_reviews)
    
    fd = nltk.FreqDist(tokenized_words)

    single_words = []
    single_counts = []

    for word,count in fd.most_common(25):
        single_words.append(word)
        single_counts.append(count)
    single_words.append("different")
    single_counts.append([fd["different"]])
    return single_words,single_counts


# In[11]:


#Bigrams
def pairWordCount(tokenized_reviews):
    tokenized_words = tokenizeWords(tokenized_reviews)

    bi_finder = nltk.collocations.BigramCollocationFinder.from_words(tokenized_words)
    
    double_words = []
    double_counts = []
    
    for word,count in bi_finder.ngram_fd.most_common(5):
        double_words.append(word[0] + ", " + word[1])
        double_counts.append(count)
    return double_words,double_counts


# In[12]:


#Trigrams
def tripletWordCount(tokenized_reviews):
    tokenized_words = tokenizeWords(tokenized_reviews)

    tri_finder = nltk.collocations.TrigramCollocationFinder.from_words(tokenized_words)
    
    triple_words = []
    triple_counts = []

    for word,count in tri_finder.ngram_fd.most_common(5):
        triple_words.append(word[0] + ", " + word[1] + ", " + word[2])
        triple_counts.append(count)
    return triple_words,triple_counts


# In[13]:


#Frequency graphs
def generateFrequencyGraph(tokenized_reviews,product):
    
    single_words,single_counts = singleWordCount(tokenized_reviews)
    pair_words,pair_counts = pairWordCount(tokenized_reviews)
    triplet_words,triplet_counts = tripletWordCount(tokenized_reviews)
    
    colors1 = []
    colors2 = ['#00876c','#4c9c85','#78b19f','#a0c6b9','#c8dbd5']
    colors3 = ['#d43d51','#df676e','#e88b8d','#eeadad','#f1cfce']

    for i in range(25):
        dark = (0, 76, 109)
        dark_r = dark[0]
        dark_g = dark[1]
        dark_b = dark[2]
        light = (192, 228, 255)
        light_r = light[0]
        light_g = light[1]
        light_b = light[2]

        diff_r = light_r - dark_r
        diff_g = light_g - dark_g
        diff_b = light_b - dark_b

        modifier = i/25
        new_color = ((dark_r + (diff_r*modifier))/255,(dark_g + (diff_g*modifier))/255,(dark_b + (diff_b*modifier))/255)
        colors1.append(new_color)
        
    colors1.append((1,0.9,0.7))

    fig1 = plt.figure(figsize=(20,12))
    fig1.add_subplot(2,1,1)
    plt.bar(range(len(single_counts)),single_counts,tick_label=single_words,color = colors1)
    plt.xticks(rotation=20)
    plt.title(f"Single Word Frequency in {product} Amazon Reviews")
    fig1.add_subplot(2,2,3)
    plt.bar(range(len(pair_counts)),pair_counts,tick_label=pair_words,color = colors2)
    plt.xticks(rotation=20)
    plt.title(f"Paired Word Frequency in {product} Amazon Reviews")
    fig1.add_subplot(2,2,4)
    plt.bar(range(len(triplet_counts)),triplet_counts,tick_label=triplet_words,color = colors3)
    plt.xticks(rotation=20)
    plt.title(f"Triplet Word Frequency in {product} Amazon Reviews")
    fig1.savefig(f"Figures/{product}/{product}_frequency")
    #plt.show()


# In[14]:


def generateSentimentAnalysis(tokenized_reviews,product):
    sia = SentimentIntensityAnalyzer()

    sentiments = []
    grouped_sentiments = {
        "strong negative":0,
        "negative":0,
        "slight negative":0,
        "neutral":0,
        "slight positive":0,
        "positive":0,
        "strong positive":0
    }

    for review,date in tokenized_reviews:
        sentiments.append(sia.polarity_scores(" ".join(review))["compound"])

    for sentiment in sentiments:
        if sentiment < -0.65:
            grouped_sentiments["strong negative"] += 1
        elif sentiment < -0.35:
            grouped_sentiments["negative"] += 1
        elif sentiment < -0.1:
            grouped_sentiments["slight negative"] += 1
        elif sentiment < 0.1:
            grouped_sentiments["neutral"] += 1
        elif sentiment < 0.35:
            grouped_sentiments["slight positive"] += 1
        elif sentiment < 0.65:
            grouped_sentiments["positive"] += 1
        else:
            grouped_sentiments["strong positive"] += 1

    sentiment_labels = []
    category_definitions = ["-1 to -0.66","-0.65 to -0.36","-0.35 to -0.11","-0.1 to 0.09","0.1 to 0.34","0.35 to 0.64","0.65 to 1"]
    sentiment_counts = []

    for category,count in grouped_sentiments.items():
            sentiment_labels.append(category)
            sentiment_counts.append(count)

    diverging_colors = ['#00876c',
    '#6aaa96',
    '#aecdc2',
    '#f1f1f1',
    '#f0b8b8',
    '#e67f83',
    '#d43d51']

    reversed_colors = []
    for color in diverging_colors:
        reversed_colors.insert(0,color)

    fig2 = plt.figure(figsize = (10,10))
    plt.bar(range(len(sentiment_counts)),sentiment_counts,tick_label = sentiment_labels,color = reversed_colors)

    patches = []

    for i in range(len(sentiment_labels)):
        patches.append(mpatches.Patch(color=reversed_colors[i], label=category_definitions[i]))

    plt.legend(handles=patches)
    plt.title(f"Sentiment Analysis of {product} Amazon reviews - {len(tokenized_reviews)} reviews")
    fig2.savefig(f"Figures/{product}/{product}_sentiment")
    #plt.show()


# In[19]:


def generateSentimentOverTime(tokenized_reviews,product):
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for review,date in tokenized_reviews:
        sentiments.append(sia.polarity_scores(" ".join(review))["compound"])
    
    today = datetime.now()
    time_grouped_sentiments = {}

    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    for i in range(len(tokenized_reviews)):
        age_months = diff_month(today,tokenized_reviews[i][1])
        if age_months < 16:
            if age_months in time_grouped_sentiments:
                time_grouped_sentiments[age_months].append(sentiments[i])
            else:
                time_grouped_sentiments[age_months] = [sentiments[i]]

    months_old = []
    avg_sentiments = []
    errors_high = []
    errors_low = []

    for key in time_grouped_sentiments:
        values = time_grouped_sentiments[key]
        months_old.append(int(key))
        mean = statistics.mean(values)
        avg_sentiments.append(mean)
        standard_error = st.sem(values)
        degrees_freedom = len(values)-1
        confidence_interval = st.t.interval(0.95, degrees_freedom, mean, standard_error)
        if math.isnan(confidence_interval[0]) or math.isnan(confidence_interval[1]):
            confidence_interval = (mean,mean)
        errors_low.append(confidence_interval[0])
        errors_high.append(confidence_interval[1])
        #print(f"Key: {key} Mean: {mean} Interval: {confidence_interval}")

    order = np.argsort(months_old)
    months_old_sorted = np.array(months_old)[order]
    avg_sentiments_sorted = np.array(avg_sentiments)[order]
    errors_high_sorted = np.array(errors_high)[order]
    errors_low_sorted = np.array(errors_low)[order]

    fig4 = plt.figure(figsize=(10,10))
    plt.plot(months_old_sorted,avg_sentiments_sorted,color='#d43d51')
    ax = plt.gca()
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim([-1,1])
    plt.xlabel("Age of Review (Months)")
    plt.ylabel("Average Sentiment Score")
    plt.title(f"Change in Sentiment of {product} Amazon Reviews Over Time")
    ax.fill_between(months_old_sorted,errors_low_sorted,errors_high_sorted,color='#d43d51',alpha=0.2)
    fig4.savefig(f"Figures/{product}/{product}_sentiment_over_time")
    #plt.show()


# In[28]:


#Wordcloud of the amazon reviews
def generateWordCloud(tokenized_reviews,product):
    colorList = [(205, 33, 56),
                 (206, 48, 72),
                 (357, 65, 64),
                 (359, 67, 73)]
    
    def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
        seed = random.randint(0,len(colorList)-1)
        color = colorList[seed]
        h = color[0]
        s = color[1]
        l = color[2]
        return "hsl({}, {}%, {}%)".format(h, s, l)
    
    tokenized_words = tokenizeWords(tokenized_reviews)
    
    fig3 = plt.figure(figsize=(16,8))
    wc = wordcloud = WordCloud(width=2000, height=1000,font_path=r'C:\Windows\Fonts\Calibri.ttf',max_words = 100,background_color = "white",color_func=random_color_func).generate(" ".join(tokenized_words))
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    fig3.savefig(f"Figures/{product}/{product}_word_cloud")
    #plt.show()


# In[26]:


def performAnalysis(product_url,product_name):
    all_product_url = getURLs(product_url)
    product_reviews = getAllReviews(all_product_url)
    product_tokenized_reviews = tokenizeReviews(product_reviews)
    createTargetDirectory(product_name)
    generateFrequencyGraph(product_tokenized_reviews,product_name)
    generateSentimentAnalysis(product_tokenized_reviews,product_name)
    generateSentimentOverTime(product_tokenized_reviews,product_name)
    generateWordCloud(product_tokenized_reviews,product_name)


# In[31]:


#flarin_url = "https://www.amazon.co.uk/FLARIN-Joint-Muscular-Relief-200mg/product-reviews/B00FOJF76E/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=1"
#nurofen_url = "https://www.amazon.co.uk/Nurofen-Tablets-200-mg-16/product-reviews/B001DXNRZ8/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"
#voltarol_url = "https://www.amazon.co.uk/Voltarol-Joint-Pain-2-32-Relief/product-reviews/B07HLR7PPZ/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=1"

#performAnalysis(flarin_url,"flarin")
#performAnalysis(nurofen_url,"nurofen")
#performAnalysis(voltarol_url,"voltarol")

running = True
while running == True:
    input_name = input("Enter the product name:")
    input_url = input("Paste the url of page 1 of the reviews here:")
    performAnalysis(input_url,input_name)
    no_accepted_input = True
    while no_accepted_input:
        user_continue = input("Would you like to perform another analysis? (yes/no)")
        if user_continue == "no":
            no_accepted_input = False
            running = False            
        elif user_continue == "yes":
            no_accepted_input = False


# In[ ]:




