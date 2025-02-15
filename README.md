# Chatbot using NLTK & TensorFlow

This is a simple AI-powered chatbot built using Natural Language Processing (NLP) with NLTK and TensorFlow. The chatbot is trained to recognize user inputs and respond with predefined responses based on detected intents.

---

## Features
- Uses **NLTK** for tokenization and text preprocessing.
- Implements a **Neural Network** using **TensorFlow/Keras** for intent classification.
- Supports **multiple intents** with predefined responses.
- Implements a **Bag-of-Words model** for input processing.
- Stores trained data using **Pickle** for fast response times.

---

## Installation
### **Prerequisites**
Make sure you have **Python 3.x** installed. Then, install the required libraries:

```bash
pip install tensorflow numpy nltk pickle5
```

---

## Project Structure
```
├── chatbot.py               # Main chatbot script
├── train.py                 # Script to train the chatbot model
├── intents.json             # Contains predefined intents & responses
├── words.pkl                # Stores preprocessed words for classification
├── classes.pkl              # Stores unique intent classes
├── chatbot_model.h5         # Trained chatbot model
├── README.md                # Project documentation
```

---

## Training the Chatbot
To train the chatbot, run:
```bash
python train.py
```
This will:
1. **Preprocess the intents** from `intents.json`.
2. **Train the model** using a neural network.
3. **Save the trained model** and word/class data.

---

## Running the Chatbot
To start interacting with the chatbot, run:
```bash
python chatbot.py
```
This will continuously accept user inputs and provide responses.

**Example Interaction:**
```
You: Hello
Bot: Hi! How can I assist you?
```

To exit the chatbot, type `exit` or `bye`.

---

## How It Works
1. **User input** is tokenized and lemmatized using NLTK.
2. The input is converted into a **Bag-of-Words vector**.
3. The trained **Neural Network** predicts the intent.
4. A **random response** from the detected intent is returned.

---

## Future Enhancements
- Add **context handling** for better conversations.
- Improve **response accuracy** using deep learning models.
- Implement **speech recognition** for voice-based interaction.

---

## License
This project is open-source and free to use.

---

### **Contributions**
Feel free to fork this repository and contribute improvements! 🚀

