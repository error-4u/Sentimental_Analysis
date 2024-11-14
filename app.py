# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import nltk
# from nltk.corpus import stopwords
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score
# import seaborn as sns
# import pickle

# # Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# # Set page configuration
# st.set_page_config(
#     page_title="Amazon Review Sentiment Analyzer",
#     layout="wide"
# )

# # Initialize session state
# if 'model' not in st.session_state:
#     st.session_state.model = None
# if 'vectorizer' not in st.session_state:
#     st.session_state.vectorizer = None

# def clean_review(review):
#     """Clean review text by removing stopwords"""
#     stp_words = stopwords.words('english')
#     cleanreview = " ".join(word for word in str(review).split() if word not in stp_words)
#     return cleanreview

# def create_wordcloud(data, sentiment):
#     """Create and return a wordcloud for given sentiment"""
#     consolidated = ' '.join(word for word in data['Review'][data['Sentiment']==sentiment].astype(str))
#     wordCloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110)
#     return wordCloud.generate(consolidated)

# def main():
#     st.title("Amazon Review Sentiment Analysis")
    
#     # Sidebar
#     st.sidebar.header("Controls")
#     page = st.sidebar.radio("Navigate", ["Upload Data", "Train Model", "Analyze Reviews"])

#     if page == "Upload Data":
#         st.header("Upload Your Dataset")
#         uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
#         if uploaded_file is not None:
#             data = pd.read_csv(uploaded_file)
#             st.session_state.data = data
            
#             st.write("Dataset Preview:")
#             st.dataframe(data.head())
            
#             st.write("Dataset Info:")
#             st.write(f"Total Reviews: {len(data)}")
#             st.write(f"Columns: {', '.join(data.columns)}")

#     elif page == "Train Model":
#         if 'data' not in st.session_state:
#             st.error("Please upload data first!")
#             return
        
#         st.header("Train Sentiment Analysis Model")
        
#         if st.button("Process and Train Model"):
#             with st.spinner("Processing data and training model..."):
#                 # Data preprocessing
#                 data = st.session_state.data.copy()
#                 data.dropna(inplace=True)
                
#                 # Convert ratings to binary sentiment
#                 data.loc[data['Sentiment']<=3, 'Sentiment'] = 0
#                 data.loc[data['Sentiment']>3, 'Sentiment'] = 1
                
#                 # Clean reviews
#                 data['Review'] = data['Review'].apply(clean_review)
                
#                 # Create visualizations
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Negative Reviews WordCloud")
#                     plt.figure(figsize=(10,5))
#                     plt.imshow(create_wordcloud(data, 0), interpolation='bilinear')
#                     plt.axis('off')
#                     st.pyplot(plt)
                
#                 with col2:
#                     st.subheader("Positive Reviews WordCloud")
#                     plt.figure(figsize=(10,5))
#                     plt.imshow(create_wordcloud(data, 1), interpolation='bilinear')
#                     plt.axis('off')
#                     st.pyplot(plt)
                
#                 # Vectorize text
#                 vectorizer = TfidfVectorizer(max_features=2500)
#                 X = vectorizer.fit_transform(data['Review']).toarray()
#                 y = data['Sentiment']
                
#                 # Train model
#                 model = LogisticRegression(random_state=42)
#                 model.fit(X, y)
                
#                 # Save model and vectorizer in session state
#                 st.session_state.model = model
#                 st.session_state.vectorizer = vectorizer
                
#                 st.success("Model trained successfully!")

#     elif page == "Analyze Reviews":
#         if st.session_state.model is None:
#             st.error("Please train the model first!")
#             return
        
#         st.header("Analyze New Reviews")
        
#         # Single review analysis
#         st.subheader("Analyze Single Review")
#         review_text = st.text_area("Enter a review:", height=100)
        
#         if st.button("Analyze Review"):
#             if review_text:
#                 # Preprocess and predict
#                 cleaned_review = clean_review(review_text)
#                 vectorized_review = st.session_state.vectorizer.transform([cleaned_review]).toarray()
#                 prediction = st.session_state.model.predict(vectorized_review)[0]
#                 probability = st.session_state.model.predict_proba(vectorized_review)[0]
                
#                 # Display results
#                 sentiment = "Positive" if prediction == 1 else "Negative"
#                 st.write(f"Sentiment: **{sentiment}**")
                
#                 # Create probability gauge
#                 prob_positive = probability[1]
#                 fig, ax = plt.subplots(figsize=(8, 3))
#                 ax.barh(['Sentiment'], [prob_positive], color='green', alpha=0.3)
#                 ax.barh(['Sentiment'], [1-prob_positive], left=[prob_positive], color='red', alpha=0.3)
#                 ax.set_xlim(0, 1)
#                 ax.set_xlabel('Probability')
#                 st.pyplot(fig)
        
#         # Batch analysis
#         st.subheader("Batch Analysis")
#         uploaded_file = st.file_uploader("Upload CSV file with reviews", type="csv", key="batch_upload")
        
#         if uploaded_file is not None:
#             batch_data = pd.read_csv(uploaded_file)
#             if 'Review' in batch_data.columns:
#                 with st.spinner("Analyzing reviews..."):
#                     # Process and predict
#                     batch_data['cleaned_review'] = batch_data['Review'].apply(clean_review)
#                     vectorized_reviews = st.session_state.vectorizer.transform(batch_data['cleaned_review']).toarray()
#                     predictions = st.session_state.model.predict(vectorized_reviews)
                    
#                     # Add predictions to dataframe
#                     batch_data['Predicted_Sentiment'] = predictions
                    
#                     # Display results
#                     st.write("Analysis Results:")
#                     st.dataframe(batch_data)
                    
#                     # Show distribution
#                     fig, ax = plt.subplots()
#                     sentiment_counts = batch_data['Predicted_Sentiment'].value_counts()
#                     ax.pie(sentiment_counts, labels=['Negative', 'Positive'], autopct='%1.1f%%')
#                     st.pyplot(fig)
#             else:
#                 st.error("Uploaded file must contain a 'Review' column!")

# if __name__ == "__main__":
#     main()


















import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None


st.sidebar.header("Sample Data Sets")
with open("amazon.csv", "rb") as amazon_file:
    st.sidebar.download_button(
        label="Download amazon.csv",
        data=amazon_file,
        file_name="amazon.csv",
        mime="text/csv"
    )

with open("hotel_reviews.csv", "rb") as sentiment_file:
    st.sidebar.download_button(
        label="Download hoel_reviews.csv",
        data=sentiment_file,
        file_name="hotel_reviews.csv",
        mime="text/csv"
    )


# def clean_review(review):
#     """Clean review text by removing stopwords"""
#     stp_words = stopwords.words('english')
#     cleanreview = " ".join(word for word in str(review).split() if word not in stp_words)
#     return cleanreview
def clean_review(review):
    """Clean review text by removing stopwords"""
    stp_words = stopwords.words('english')
    cleaned_review = " ".join(word for word in str(review).split() if word not in stp_words)
    if len(cleaned_review.split()) >= 1:
        return cleaned_review
    else:
        return ""
        

# def create_wordcloud(data, sentiment_column, sentiment_value):
#     """Create and return a wordcloud for given sentiment"""
#     consolidated = ' '.join(word for word in data['Review'][data[sentiment_column]==sentiment_value].astype(str))
#     wordCloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110)
#     return wordCloud.generate(consolidated)

def create_wordcloud(data, sentiment_column, sentiment_value):
    """Create and return a wordcloud for given sentiment"""
    consolidated = ' '.join(word for word in data['Review'][data[sentiment_column]==sentiment_value].astype(str))
    if consolidated:
        wordCloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110)
        return wordCloud.generate(consolidated)
    else:
        return None
    

def preprocess_and_train(data, sentiment_column):
    """Preprocess data and train the sentiment analysis model"""
    # Data preprocessing
    data.dropna(inplace=True)
    
    # Convert ratings to binary sentiment
    data.loc[data[sentiment_column]<=3, 'Sentiment'] = 0
    data.loc[data[sentiment_column]>3, 'Sentiment'] = 1
    
    # Clean reviews
    data['Review'] = data['Review'].apply(clean_review)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=2500)
    X = vectorizer.fit_transform(data['Review']).toarray()
    y = data['Sentiment']
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Save model and vectorizer in session state
    st.session_state.model = model
    st.session_state.vectorizer = vectorizer

    return model, vectorizer

def main():
    st.title("Amazon Review Sentiment Analysis")
    
    # Sidebar
    st.sidebar.header("Controls")
    page = st.sidebar.radio("Navigate", ["Upload Data", "Train Model", "Analyze Reviews"])

    if page == "Upload Data":
        st.header("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.write("Dataset Preview:")
            st.dataframe(data.head())
            
            st.write("Dataset Info:")
            st.write(f"Total Reviews: {len(data)}")
            st.write(f"Columns: {', '.join(data.columns)}")

              

    elif page == "Train Model":
        if 'data' not in st.session_state:
            st.error("Please upload data first!")
            return
        
        st.header("Train Sentiment Analysis Model")
        
        # Get sentiment column name
        sentiment_column = st.text_input("Enter the name of the sentiment/rating column", "Sentiment")
        
        if st.button("Process and Train Model"):
            with st.spinner("Processing data and training model..."):
                data = st.session_state.data.copy()
                model, vectorizer = preprocess_and_train(data, sentiment_column)
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
            # Update the code in the 'Train Model' section
            with col1:
             st.subheader("Negative Reviews WordCloud")
             neg_wordcloud = create_wordcloud(data, sentiment_column, 0)
             if neg_wordcloud is not None:
                 plt.figure(figsize=(10,5))
                 plt.imshow(neg_wordcloud, interpolation='bilinear')
                 plt.axis('off')
                 st.pyplot(plt)
             else:
                 st.write("No negative reviews found to create a wordcloud.")

            with col2:
             st.subheader("Positive Reviews WordCloud")
             pos_wordcloud = create_wordcloud(data, sentiment_column, 1)
             if pos_wordcloud is not None:
                plt.figure(figsize=(10,5))
                plt.imshow(pos_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
             else:
                st.write("No positive reviews found to create a wordcloud.")

    # ... rest of the code remains the same ...

    # elif page == "Train Model":
    #     if 'data' not in st.session_state:
    #      st.error("Please upload data first!")
    #     return
    
    # st.header("Train Sentiment Analysis Model")
    
    # # Get sentiment column name
    # sentiment_column = st.text_input("Enter the name of the sentiment/rating column", "Sentiment")
    
    # if st.button("Process and Train Model"):
    #     with st.spinner("Processing data and training model..."):
    #         data = st.session_state.data.copy()
    #         model, vectorizer = preprocess_and_train(data, sentiment_column)
            
    #         # Create visualizations
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             st.subheader("Negative Reviews WordCloud")
    #             neg_wordcloud = create_wordcloud(data, sentiment_column, 0)
    #             if neg_wordcloud is not None:
    #                 plt.figure(figsize=(10,5))
    #                 plt.imshow(neg_wordcloud, interpolation='bilinear')
    #                 plt.axis('off')
    #                 st.pyplot(plt)
    #             else:
    #                 st.write("No negative reviews found to create a wordcloud.")
            
    #         with col2:
    #             st.subheader("Positive Reviews WordCloud")
    #             pos_wordcloud = create_wordcloud(data, sentiment_column, 1)
    #             if pos_wordcloud is not None:
    #                 plt.figure(figsize=(10,5))
    #                 plt.imshow(pos_wordcloud, interpolation='bilinear')
    #                 plt.axis('off')
    #                 st.pyplot(plt)
    #             else:
    #                 st.write("No positive reviews found to create a wordcloud.")
            
    #         st.success("Model trained successfully!")
            
# ... rest of the code remains the same ...

    elif page == "Analyze Reviews":
        if st.session_state.model is None:
            st.error("Please train the model first!")
            return
        
        st.header("Analyze New Reviews")
        
        # Single review analysis
        st.subheader("Analyze Single Review")
        review_text = st.text_area("Enter a review:", height=100)
        
        if st.button("Analyze Review"):
            if review_text:
                # Preprocess and predict
                cleaned_review = clean_review(review_text)
                vectorized_review = st.session_state.vectorizer.transform([cleaned_review]).toarray()
                prediction = st.session_state.model.predict(vectorized_review)[0]
                probability = st.session_state.model.predict_proba(vectorized_review)[0]
                
                # Display results
                sentiment = "Positive" if prediction == 1 else "Negative"
                st.write(f"Sentiment: **{sentiment}**")
                
                # Create probability gauge
                prob_positive = probability[1]
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(['Sentiment'], [prob_positive], color='green', alpha=0.3)
                ax.barh(['Sentiment'], [1-prob_positive], left=[prob_positive], color='red', alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                st.pyplot(fig)
        
        # Batch analysis
        st.subheader("Batch Analysis")
        uploaded_file = st.file_uploader("Upload CSV file with reviews", type="csv", key="batch_upload")
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            if 'Review' in batch_data.columns:
                with st.spinner("Analyzing reviews..."):
                    # Process and predict
                    batch_data['cleaned_review'] = batch_data['Review'].apply(clean_review)
                    vectorized_reviews = st.session_state.vectorizer.transform(batch_data['cleaned_review']).toarray()
                    predictions = st.session_state.model.predict(vectorized_reviews)
                    
                    # Add predictions to dataframe
                    batch_data['Predicted_Sentiment'] = predictions
                    
                    # Display results
                    st.write("Analysis Results:")
                    st.dataframe(batch_data)
                    
                    # Show distribution
                    fig, ax = plt.subplots()
                    sentiment_counts = batch_data['Predicted_Sentiment'].value_counts()
                    ax.pie(sentiment_counts, labels=['Negative', 'Positive'], autopct='%1.1f%%')
                    st.pyplot(fig)
            else:
                st.error("Uploaded file must contain a 'Review' column!")

if __name__ == "__main__":
    main()
















































