import streamlit as st
import src.utils as utils
from src.pipelines.prediction_pipeline import PredictPipeline
import time

predict = PredictPipeline()

def main():

    st.title('*Sentiment Analysis System*')
    st.markdown('Check the sentiment of your text. ')
    st.markdown('**Note**: This project is in initial stage. So there are high chances you may encounter some error!')

    st.markdown(" ")
    user_input = st.text_area(label = 'Enter you text.', placeholder='Text')

    if st.button("Predict"):

        if user_input[:] == "":
            st.warning("Please enter a message.")

        else:
            # Preprocess user input
            result = predict.predict(features=user_input)


            with st.spinner('Processing!'):
                time.sleep(1)

                # Display prediction
                if result == 1:
                    st.success("Positive_sentiment")
                    
                else:
                    st.error("Negative_sentiment")
                    

                
if __name__ == "__main__":
    main()