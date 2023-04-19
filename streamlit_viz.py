import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
from random import sample
import textstat
import pickle
import sklearn.linear_model

#@st.cache_data
def load_data(path):
    ta_sample = pd.read_csv(path, sep = ',',  header = 0)
    #ta_sampled = ta.sample(1000)
    ta_model = pickle.load(open('ta_model.sav', 'rb'))
    return ta_sample, ta_model
    

#st.text(os.getcwd())
ta_sample, ta_model = load_data('ta_sample_dump.csv')




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
    ds = ta_sample
    model = ta_model
    
    #get sampled values
    gunning = ds['GunningFog']
    helpfulness_prob = ds['helpfulness_prob']
    subjectivity = ds['subjectivity']
    reading_time = ds['ReadingTime']
    
    #predict current values 
    cur_gunning = textstat.gunning_fog(txt)
    cur_reading_time = textstat.reading_time(txt, ms_per_char=14.69)
    
    #assemble current datapoint
    cur = pd.DataFrame()
    cur = cur.append({
    'first_page_exposure':np.random.rand()*100,
    'polarity':0,
    'subjectivity':np.random.rand(),
    'polarity_sign':0,
    'polarity_magnitude':np.random.rand(),
    'GunningFog':cur_gunning,
    'ReadingTime':cur_reading_time,
    'activity_group':np.random.rand(),
    'ease_group':np.random.rand(),
    'feeling_group':np.random.rand(),
    'food_group':np.random.rand(),
    'judgement_group':np.random.rand(),
    'location_group':np.random.rand(),
    'misc_group':np.random.rand(),
    'price_group':np.random.rand(),
    'service_group':np.random.rand(),
    'time_group':np.random.rand(),
    'venue_group':np.random.rand()},
    ignore_index=True)

    predicted_val = model.predict_proba(cur)[0,1]
    
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
    st.markdown(r"Your score for subjectivity is **:blue[{0:.1f}]** (ranges from 0 to 1, least to most subjective)".format(np.random.rand()),
            help = readability_help)
    
    cur_subjectivity = 0.5
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
    
    
    st.subheader("Word frequencies")
    st.write("According to our model, using specific words in your review can make it more likely to be rated helpful by other users")
    st.write("""We have grouped these words into several categories for convenience. Below you can see how your review 
    scored in each category (if a category has a score of 5, it means your review uses 5 different words from the category; 
    using a word more than once doesn't lead to a higher score), and also the percentage point increase in the probability
    of the review being considered helpful per each score in this category""")
    #st.table(df)
    
    word_probs = np.array(model.coef_)[0,-11:].T
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
