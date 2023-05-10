import streamlit as st
import json
from keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, input_text, max_length=20):
    # initialize the generated output text with the input text
    generated_text = input_text
    # set the stop condition to False
    stop_condition = False
    while not stop_condition:
        # tokenize the input text
        input_sequence = tokenizer.texts_to_sequences([generated_text])[0]
        # pad the input sequence
        input_sequence = pad_sequences([input_sequence], maxlen=max_length-1, padding='pre')
        # make a prediction
        prediction = model.predict(input_sequence)[0]
        # get the index of the predicted word
        predicted_index = np.argmax(prediction)
        # get the predicted word
        predicted_word = tokenizer.index_word.get(predicted_index, '')
        # check if we've generated the maximum length or found the end token
        if len(generated_text.split()) == max_length or predicted_word == 'end':
            stop_condition = True
        else:
            # append the predicted word to the generated text
            generated_text += ' ' + predicted_word
    return generated_text.strip()

with open('model/lstm_tokenizer.json', 'r') as f:
    data = json.load(f)
    tokenizer_json = json.dumps(data)  # Convert dictionary to JSON-formatted string
    tokenizer = tokenizer_from_json(tokenizer_json)

    # load the mode
model = tf.keras.models.load_model('model/lstm_model_4.h5')


    # create the Streamlit app
st.title('LSTM Text Generation')
input_text = st.text_input('Enter the seed text:')
max_length = st.slider('Select the maximum length of the generated text:', 5, 50, 20)
generate_button = st.button('Generate Text')

    # generate text when the button is clicked
if generate_button:
    generated_text = generate_text(model, tokenizer, input_text, max_length=max_length)
    st.write('Generated Text:', generated_text)

