import streamlit as st
import joblib,os
### NLP packages
# import spacy


## WordCloud

news_vectorizer = open('models/final_news_cv_vectorizer.pkl','rb')
news_cv = joblib.load(news_vectorizer)


def load_models(model_file):
    open_file = open(os.path.join(model_file),'rb')
    loaded_models = joblib.load(open_file)
    return loaded_models

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key


def main():
    ### News Classifier with Streamlit
    st.title("News Classifier")
    st.subheader("NLP and ML App with Streamlit")

    activities = ['Prediction','NLP']
    choice = st.sidebar.selectbox('Choose Activity',activities)

    if choice == 'Prediction':
        st.info('Prediction with ML')

        news_text = st.text_area('Enter Text','Type Here')
        ml_models = ['LR','NB','RFOREST']
        model_choices = st.selectbox('Choose Models Used',ml_models)
        prediction_labels = {'Business':0,'Tech':1,'Sports':2,'Health':3,'Politics':4,'Entertainment':5}
        if st.button('Classify'):
            st.text("Original text ::\n{}".format(news_text))
            vect_text = news_cv.transform([news_text]).toarray()

            if model_choices == 'LR':
                predictor = load_models("models/newsclassifier_Logit_model.pkl")
                prediction = predictor.predict(vect_text)


            elif model_choices == 'NB':
                predictor = load_models("models/newsclassifier_NB_model.pkl")
                prediction = predictor.predict(vect_text)
                # results = get_keys(prediction, prediction_labels)
                # st.success(results)
            elif model_choices == 'RFOREST':
                predictor = load_models("models/newsclassifier_RFOREST_model.pkl")
                prediction = predictor.predict(vect_text)
                # results = get_keys(prediction, prediction_labels)
                # st.success(results)
            results = get_keys(prediction, prediction_labels)
            st.success("News is classified as : {}".format(results))

    elif choice == 'NLP':
        st.info('Natural Language Processing')






if __name__ == "__main__":
    main()
