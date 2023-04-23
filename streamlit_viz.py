import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
from random import sample
import textstat
import pickle
#import sklearn.linear_model
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob

#word grouping part

@st.cache_data
def nlp_model(): #function to load in the nlp model once
    from spacy import load
    nlp = load('en_core_web_sm', disable = ["tok2vec", 'ner', 'parser'])
    return nlp

def get_top_words(): #function to get top words
    words = ['free', 'imax', 'area', 'come', 'drink', 'use', 'seat', 'customer', 'happy', 'recommend', 'got', 'bread', 'thing', \
             'came', 'lobby', 'movies', 'waitress', 'sweet', 'rooms', 'early', 'breakfast', 'want', 'price', 'theaters', 'morning', \
            'location', 'long', 'order', 'tickets', 'better', 'reclining', 'rice', 'line', 'manager', 'venue', 'worth', 'street', \
            'films', 'beer', 'took', 'chairs', 'try', 'big', 'lot', 'wait', 'stand', 'tour', 'shows', 'money', 'taste', 'atmosphere',\
            'beautiful', 'staying', 'good', 'meal', 'years', 'home', 'tasty', 'hotel', 'concession', 'great', 'room', 'hour', 'close',\
            'check', 'airport', 'movie', 'eat', 'went', 'nice', 'popcorn', 'times', 'minutes', 'fresh', 'time', 'need', 'night', 'fun',\
            'drinks', 'buy', 'comfortable', 'server', 'floor', 'awesome', 'saw', 'table', 'day', 'helpful', 'visit', 'coffee', 'kids', \
            'new', 'amc', 'place', 'hours', 'fries', 'way', 'cheese', 'shower', 'small', 'food', 'fried', 'lunch', 'work', 'love', \
            'walk', 'told', 'bed', 'find', 'trip', 'feel', 'experience', 'parking', 'salad', 'right', 'ordered', 'film', 'theater', 'bit',\
            'going', 'pizza', 'burger', 'sandwich', 'enjoy', 'sure', 'super', 'flavor', 'large', 'service', 'know', 'pretty', 'delicious',\
            'screen', 'watch', 'pool', 'asked', 'dish', 'concessions', 'spot', 'said', 'sauce', 'chicken', 'clean', 'friendly', 'like', 'shrimp',\
            'seats', 'think', 'old', 'excellent', 'people', 'best', 'booked', 'prices', 'desk', 'bad', 'meat', 'water', 'called', 'door', 'showing',\
            'perfect', 'bar', 'ticket', 'away', 'tried', 'definitely', 'menu', 'amazing', 'stay', 'staff', 'favorite', 'dinner', 'bathroom', 'left',\
            'seating', 'usually', 'car', 'hot', 'city', 'stayed', 'little', 'theatre', 'things', 'restaurant', 'sound']
    return words

#takes in review text, and nlp model loaded from nlp_model(), and top words from get_top_words()
#  and returns list containing the frequency of the top words
def word_frequencies(review_text, nlp, words): 
    word_dict = dict((word, 0) for word in words)

    doc = nlp(review_text)
    doc_set = [token.lemma_ for token in doc]

    for word in doc_set:
        if word in words:
            word_dict[word] += 1

    return word_dict

def to_groups(word_frequency_dict): # returns a dictionary of each grouping and quanitity of words in each group from review text
    import pandas as pd
    activity_words = ['bar','came','car','come','eat','experience','film','films','going','got','imax','lunch','movie','movies',
                      'order','ordered','pool','reclining','saw','shower','showing','shows','stay','stayed','staying','theater','theaters',
                      'theatre','ticket','tour','tried','trip','use','visit','walk','watch','went']
    ease_words = ['line','long','stand','try','wait']
    feeling_words = ['comfortable','happy','love','want']
    food_words = ['beer','bit','bread','breakfast','burger','cheese','chicken','coffee','concession','concessions','delicious',
                  'dinner','dish','drink','drinks','flavor','food','fresh','fried','fries','meal','meat','menu','pizza','popcorn',
                  'restaurant','rice','salad','sandwich','sauce','shrimp','sweet','taste','tasty','water']
    judgement_words = ['amazing','awesome','bad','beautiful','best','better','clean','definitely','enjoy','excellent','favorite','feel',
                       'fun','good','great','helpful','know','large','like','little','lot','need','new','nice','perfect','pretty',
                       'recommend','right','small','super','sure','think','usually','work','worth']
    location_words = ['airport','area','away','city','location','place','street']
    misc_words = ['asked','find','left','said','thing','told','took','way']
    price_words = ['buy','check','free','money','price','prices','tickets']
    service_words = ['called','customer','friendly','manager','people','server','service','staff','waitress']
    time_words = ['close','day','early','hour','hours','minutes','morning','night','time','times','years']
    venue_words = ['amc','atmosphere','bathroom','bed','big','booked','chairs','desk','door','floor','home','hot','hotel','kids',
                   'lobby','old','parking','room','rooms','screen','seat','seating','seats','sound','spot','table','things','venue']
    df = pd.DataFrame([word_frequency_dict])
    df_grouped = pd.DataFrame()
    df_grouped['activity_words'] = df[activity_words].sum(axis = 1)
    df_grouped['ease_words'] = df[ease_words].sum(axis = 1)
    df_grouped['feeling_words'] = df[feeling_words].sum(axis = 1)
    df_grouped['food_words'] = df[food_words].sum(axis = 1)
    df_grouped['judgement_words'] = df[judgement_words].sum(axis = 1)
    df_grouped['location_words'] = df[location_words].sum(axis = 1)
    df_grouped['misc_words'] = df[misc_words].sum(axis = 1)
    df_grouped['price_words'] = df[price_words].sum(axis = 1)
    df_grouped['service_words'] = df[service_words].sum(axis = 1)
    df_grouped['time_words'] = df[time_words].sum(axis = 1)
    df_grouped['venue_words'] = df[venue_words].sum(axis = 1)
    return df_grouped.to_dict('records')[0]

def get_frequencies(review_text):
    #nltk.download()
    nlp = nlp_model()
    top_words = get_top_words()
    word_freq_dict = word_frequencies(review_text, nlp, top_words)
    grouped_freq_dict = to_groups(word_freq_dict)   
    return grouped_freq_dict

#sentiment part


# Define a function to tokenize the text
def tokenize_text(text, stop_words, lemmatizer):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# Define a function for sentiment polarity
def get_polarity_score(tokens, sia):
    text = ' '.join(tokens)
    score = sia.polarity_scores(text)
    return score['compound']

# Define a function for subjectivity
def get_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity

def run_sentiment(txt):
    sia = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = tokenize_text(txt, stop_words, lemmatizer)
    polarity = get_polarity_score(tokens, sia)
    subjectivity = get_subjectivity(txt)
    return polarity, subjectivity

#data loading part
@st.cache_data
def load_data(path):
    samples = []
    models = []
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('punkt') 
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    samples.append(pd.read_csv(path+'tripadvisor_sample.csv', sep = ',',  header = 0))
    samples.append(pd.read_csv(path+'yelp_cinema_sample.csv', sep = ',',  header = 0))
    samples.append(pd.read_csv(path+'yelp_hotel_sample.csv', sep = ',',  header = 0))
    samples.append(pd.read_csv(path+'yelp_restaurant_sample.csv', sep = ',',  header = 0))
    
    models.append(pickle.load(open(path+'ta_final_model_small.pickle', 'rb')))
    models.append(pickle.load(open(path+'yelp_cinema_model_smaller.PICKLE', 'rb')))
    models.append(pickle.load(open(path+'yelp_hotel_model_smaller.PICKLE', 'rb')))
    models.append(pickle.load(open(path+'yelp_restaurant_final_model_small.pickle', 'rb')))

    return samples, models
    

#st.text(os.getcwd())
samples, models = load_data('data/')


st.title('Review helpfulness predictor')
st.write('This interactive tool lets you enter review text and see how our model evaluates its chances to be rated helpful by users on Tripadvisor and Yelp.')
#st.write(max(ta_gunning))
txt = st.text_area('Input review text', '')

platform_option = st.selectbox(
    'Which platform should we estimate against?',
    ('Yelp', 'Tripadvisor'))

if platform_option == 'Yelp':
    industry_option = st.selectbox(
        'What is the industry of the business you are reviewing?',
        ('Hotel', 'Cinema', 'Restaurant'))

if txt > "":
    #st.write("Length of text entered", len(txt), "Industry:",industry_option)
    #if platform_option == 'Tripadvisor':
    if platform_option == 'Tripadvisor':
        i = 0
    elif industry_option == 'Cinema':
        i = 1
    elif industry_option == 'hotel':
        i = 2
    else: 
        i = 3
    ds = samples[i]
    model = models[i]
    
    #get sampled values
    gunning = ds['GunningFog']
    helpfulness_prob = ds['helpfulness_prob']
    subjectivity = ds['subjectivity']
    polarity = ds['polarity']
    reading_time = ds['ReadingTime']
    
    #predict current values 
    cur_gunning = textstat.gunning_fog(txt)
    cur_reading_time = textstat.reading_time(txt, ms_per_char=14.69)
    
    freq_list = get_frequencies(txt)
    
    cur_polarity, cur_subjectivity = run_sentiment(txt)
    
    #st.write(freq_list  )
    
    #assemble current datapoint
    cur = pd.DataFrame()
    cur_dict_stars = {
    'first_page_exposure':np.random.rand()*100,
    'stars': 3,
    'polarity':cur_polarity,
    'subjectivity':cur_subjectivity,
    'GunningFog':cur_gunning,
    'ReadingTime':cur_reading_time,
    'activity_group': freq_list["activity_words"],
    'ease_group':freq_list["ease_words"],
    'feeling_group':freq_list["feeling_words"],
    'food_group':freq_list["food_words"],
    'judgement_group':freq_list["judgement_words"],
    'location_group':freq_list["location_words"],
    'misc_group':freq_list["misc_words"],
    'price_group':freq_list["price_words"],
    'service_group':freq_list["service_words"],
    'time_group':freq_list["time_words"],
    'venue_group':freq_list["venue_words"]}
    cur_dict_no_stars = {
    'first_page_exposure':np.random.rand()*100,
    'polarity':cur_polarity,
    'subjectivity':cur_subjectivity,
    'GunningFog':cur_gunning,
    'ReadingTime':cur_reading_time,
    'activity_group': freq_list["activity_words"],
    'ease_group':freq_list["ease_words"],
    'feeling_group':freq_list["feeling_words"],
    'food_group':freq_list["food_words"],
    'judgement_group':freq_list["judgement_words"],
    'location_group':freq_list["location_words"],
    'misc_group':freq_list["misc_words"],
    'price_group':freq_list["price_words"],
    'service_group':freq_list["service_words"],
    'time_group':freq_list["time_words"],
    'venue_group':freq_list["venue_words"]}

    #st.write(cur_dict)
    if i == 0: 
        cur = cur.append(cur_dict_no_stars, ignore_index=True)
    else: cur = cur.append(cur_dict_stars, ignore_index=True)
    #st.write(cur)
    predicted_val = model.predict(cur)[0]
    #st.write()
    
    #OVERALL SCORE
    st.header("Overall likelihood score")
    st.write("Likelihood that your review is found helpful:", round(predicted_val*100,1),"%")
    st.header("Category scores")
    st.write("""The sections below show the scores that your review text gets on different 
    categories such as readability, sentiment and frequencies of particular words, and visualizes how those scores influence
    the likelihood of the review being considered helpful according to our model""")
    st.subheader("Readability stats")
    
    #READABILITY
    
    readability_help = "Higher means more complicated. The metric measures the number of words per sentence, and the number of long words per word."
    st.markdown(r"Your score for readability (Gunning-Fog metric) is **:blue[{0:.1f}]** against a normal range of 2 to 20".format(cur_gunning),
            help = readability_help)
    st.write("""The chart below shows the Gunning-Fog score of your text and the associated helpfulness score (their position is
    indicated by the crossing of the red dotted lines) against the background of
    the same scores for a random sample from our training set, giving you an idea of how this property correlates with helpfulness.""")
    #readability graphs

    x = gunning
    y = helpfulness_prob

    fig = px.scatter(x=x, y=y)#,  size = size)
    fig.add_vline(x=cur_gunning, line_dash = 'dash', line_color = 'firebrick')    
    fig.add_hline(y=predicted_val, line_dash = 'dash', line_color = 'firebrick')
    fig.update_layout(
    title="Gunning-Fog readability metric vs helpfulness probability",
    xaxis_title="Gunning-Fog score",
    yaxis_title="Predicted helpfulness probability")
    st.plotly_chart(fig, use_container_width=True)
    
    
    x = reading_time
    reading_time_help = "Reading time measures how long (in seconds) it takes an average adult to read"
    st.markdown(r"Your score for reading time is is **:blue[{0:.1f}]**".format(cur_reading_time),
            help = reading_time_help)
    fig = px.scatter(x=x, y=y)#,  size = size)
    fig.add_vline(x=cur_reading_time, line_dash = 'dash', line_color = 'firebrick')    
    fig.add_hline(y=predicted_val, line_dash = 'dash', line_color = 'firebrick')
    fig.update_layout(
    title="Reading time metric vs helpfulness probability",
    xaxis_title="Reading time score",
    yaxis_title="Predicted helpfulness probability")
    st.plotly_chart(fig, use_container_width=True)
    
    #SENTIMENT
    st.subheader("Sentiment stats")
    subjectivity_help = 'Measures how much the review expresses the opinion of the writer vs "objective truth"'
    st.markdown(r"Your score for subjectivity is **:blue[{0:.1f}]** (ranges from 0 to 1, least to most subjective)".format(cur_subjectivity),
            help = subjectivity_help)
    
    #cur_subjectivity = subjectivity
    x = subjectivity
    y = helpfulness_prob
    fig = px.scatter(x=x, y=y)#,  size = size)
    fig.add_vline(x=cur_subjectivity, line_dash = 'dash', line_color = 'firebrick')    
    fig.add_hline(y=predicted_val, line_dash = 'dash', line_color = 'firebrick')
    fig.update_layout(
    title="Subjectivity vs helpfulness probability",
    xaxis_title="Subjectivity",
    yaxis_title="Predicted helpfulness probability")
    st.plotly_chart(fig, use_container_width=True)
    
    polarity_help = 'Measures the overall sentiment of the text'
    st.markdown(r"Your score for polarity is **:blue[{0:.1f}]** (ranges from -1 to 1, negative to positive)".format(cur_polarity),
            help = polarity_help)
    x = polarity
    y = helpfulness_prob
    fig = px.scatter(x=x, y=y)#,  size = size)
    fig.add_vline(x=cur_polarity, line_dash = 'dash', line_color = 'firebrick')    
    fig.add_hline(y=predicted_val, line_dash = 'dash', line_color = 'firebrick')
    fig.update_layout(
    title="Polarity vs helpfulness probability",
    xaxis_title="Polarity",
    yaxis_title="Predicted helpfulness probability")
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    
    st.subheader("Word frequencies")
    st.write("According to our model, using specific words in your review can make it more likely to be rated helpful by other users")
    st.write("""We have grouped these words into several categories for convenience. Below you can see how your review 
    scored in each category (if a category has a score of 5, it means your review uses 5 different words from the category; 
    using a word more than once doesn't lead to a higher score), and also the percentage point increase in the probability
    of the review being considered helpful per each score in this category""")
    #st.table(df)
    #st.write(model.params)
    word_probs = np.array(model.params)[-11:].T
    word_probs = np.exp(word_probs)
    word_probs = word_probs-1
    #st.write(np.array(model.coef_)[0,8:])
    #st.write(cur.columns.values[-11:])
    labels = cur.columns.values[-11:]
    #st.write(np.array(cur.loc[0][-11:]))
    your_coeffs = np.array(cur.loc[0][-11:])
    
    fig = px.bar(y=your_coeffs, x = labels)
    fig.update_layout(
    title="Your text's score in each word category",
    xaxis_title="Word category",
    yaxis_title="Number of words in category")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(y=word_probs, x = labels)
    fig.update_layout(
    title="Estimated increase in probability of helpfulness",
    xaxis_title="Word category",
    yaxis_title="Increse in predicted helpfulness chance")
    st.plotly_chart(fig, use_container_width=True)
    
    #st.bar_chart(word_probs)
    #st.bar_chart(your_coeffs)
