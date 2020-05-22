import streamlit as st
import pickle
import json,os
import numpy as np
import pandas as pd
import joblib
### DEFINE CONSTANTS
MODEL_PATH = 'Housing_model/Goodmodel.pkl'
SCALER = 'Housing_model/scaler.pkl'
COLUMNS = 'Housing_model/GoodColumns.json'
bedrooms = range(1,7)
bathrooms = range(1,6)
districts = range(1,29)
types = {'HDB':0,'CONDO':1,'LANDED':2}
mrt = {'Near MRT':1,'Not very near':0}
IMAGE1 = 'Housing_Model/image1.jpg'
IMAGE2 = 'Housing_Model/district.jpg'
IMAGE3 = 'Housing_Model/news-2.jpg'

def load_scaler():
    with open(os.path.join(SCALER),'rb') as f:
        model = pickle.load(f)
    return model

def load_models(PATH):
    with open(os.path.join(PATH),'rb') as f:
        model = pickle.load(f)

    return model

def load_columns():
    with open(os.path.join(COLUMNS)) as json_file:
        columns = json.load(json_file)
    return columns

def joblib_load(model_file):
    open_file = open(os.path.join(model_file),'rb')
    loaded_models = joblib.load(open_file)
    return loaded_models

def format_input(n_rooms,n_baths,floor_size,subtype,district,mrt):
    columns = load_columns()['data_columns']
    scaler = load_scaler()
    data = np.zeros((1,32))
    data[0][0:4] = [n_rooms,n_baths,floor_size,mrt]
    if subtype != 0:
        data[0][3+subtype] = 1
    if district > 1:
        data[0][4+district] = 1
    if district > 23 and district != 24:
        data[0][3+district] = 1
    final = pd.DataFrame(data=data,columns=columns)
    ans = scaler.transform(final)
    return ans



def predictor():
    st.title('Singapore Rental Price Prediction')
    # st.subheader()
    st.markdown("The output of this application only acts as a general reference to Singapore's house rental markets")
    # st.image(IMAGE1, use_column_width=True)
    st.image(IMAGE2, use_column_width=True)

    # st.text('Select the no. of Bedrooms')
    dist = st.selectbox('Select the District', districts)
    sqft = st.text_input('Area in Squarefeet', value=800)
    beds = st.selectbox('Select the no. of Bedrooms', bedrooms)
    baths = st.selectbox('Select the no. of Baths', bathrooms)

    # substype = st("Types of Housing",list(types.keys()))
    subtype = st.selectbox("Types of Housing", list(types.keys()))
    MRT = st.selectbox("Proximity to MRT", list(mrt.keys()))
    model = load_models(MODEL_PATH)
    if st.button("Estimate Price"):
        if sqft != 0 and beds != 0 and baths != 0 and dist != 0 and len(subtype) and len(MRT):
            inp = format_input(beds, baths, sqft, types[subtype], dist, mrt[MRT])
            print(inp)
            results = model.predict(inp)[0]
            st.success("Estimated Price for your input is: {} SGD".format(int(results)))
            st.balloons()
        else:
            st.text("Input not Complete")

def news():
    news_vectorizer = open('models/final_news_cv_vectorizer.pkl', 'rb')
    news_cv = joblib.load(news_vectorizer)
    st.title("News Classifier")
    st.markdown("Using Parts of the or entire News Article to predict categories")
    st.markdown("Categories included: Business, Tech, Sports, Health, Politics, Entertainment")
    st.image(IMAGE3, use_column_width=True)
    news_text = st.text_area('Enter Text', 'Type Here')
    model = joblib_load('models/Logit.pkl')
    labels = {0:'Business', 1:'Tech', 2:'Sports', 3:'Health', 4:'Politics',  5:'Entertainment'}
    if st.button('Classify'):
        st.text("Original text ::\n{}".format(news_text))
        vect_text = news_cv.transform([news_text]).toarray()
        prediction = model.predict(vect_text)[0]
        st.success("This is a piece of {} news".format(labels[prediction]))

def about():
    st.subheader("")
    st.subheader("By Zhao Zhijie")
    st.subheader("Contact: mshs.zhao.zhijie@gmail.com")


def main():
    activities = ['Rental Price Predictor','News Classification','About']
    choice = st.sidebar.selectbox('Menu',activities)
    if choice == 'Rental Price Predictor':
        predictor()
    elif choice == 'News Classification':
        news()

    elif choice == 'About':
        about()







if __name__ == "__main__":
    main()
