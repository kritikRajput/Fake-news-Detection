import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import asyncio
import platform
from typing import Dict, List, Tuple
import logging
import tweepy
from datetime import datetime
from sklearn.model_selection import train_test_split
import os
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsDetector:
    def __init__(self, model_weights_path: str = 'fake_news_text_classifier_weights.h5'):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.text_classifier = self._build_text_classifier()
        self.image_classifier = self._build_image_classifier()
        self.api = self._initialize_x_api()
        self.model_weights_path = model_weights_path
        if os.path.exists(model_weights_path):
            self.text_classifier.load_weights(model_weights_path)
            logger.info(f"Loaded model weights from {model_weights_path}")
        else:
            logger.warning(f"Model weights file {model_weights_path} not found. Train the model to generate weights.")

    def _initialize_x_api(self):
        """Initialize X API connection with your credentials"""
        auth = tweepy.OAuth1UserHandler(
            consumer_key="iEBkYu8YmJbconFjgTI1eLaNM",
            consumer_secret="NF6zMxPn6wOCdO8csAT6YeIWUw89FKSJn5r5c6PFYVdMoe00gD",
            access_token="1733849955536982016-QEqBvDSYqQ8QbVK6re2Y4QljIzfvHj",
            access_token_secret="CQ7YK5BxCCR3dz6DrEcAke5JBOIp5A3GCjHCz6ymVmYzl"
        )
        return tweepy.API(auth, wait_on_rate_limit=True)

    def _build_text_classifier(self) -> tf.keras.Model:
        """Build BERT-based text classification model"""
        input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
        
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask)[0]
        cls_token = bert_outputs[:, 0, :]
        
        x = tf.keras.layers.Dropout(0.1)(cls_token)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _build_image_classifier(self) -> tf.keras.Model:
        """Build CNN-based image classification model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def preprocess_text(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess text for BERT"""
        encoded = self.bert_tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for CNN"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    def load_and_preprocess_data(self, true_path: str, fake_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess True.csv and Fake.csv datasets"""
        try:
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)
            
            true_df['label'] = 1
            fake_df['label'] = 0
            
            combined_df = pd.concat([true_df, fake_df], ignore_index=True)
            combined_df['text'] = combined_df['text'].fillna('')
            
            input_ids = []
            attention_masks = []
            labels = combined_df['label'].values
            
            for text in combined_df['text']:
                ids, mask = self.preprocess_text(text)
                input_ids.append(ids.numpy().flatten())
                attention_masks.append(mask.numpy().flatten())
            
            input_ids = np.array(input_ids)
            attention_masks = np.array(attention_masks)
            
            return input_ids, attention_masks, labels
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {str(e)}")
            raise

    def train_model(self, input_ids: np.ndarray, attention_masks: np.ndarray, labels: np.ndarray, 
                    epochs: int = 3, batch_size: int = 16, validation_split: float = 0.2):
        """Train the text classifier and save weights"""
        try:
            X_train_ids, X_val_ids, X_train_mask, X_val_mask, y_train, y_val = train_test_split(
                input_ids, attention_masks, labels, test_size=validation_split, random_state=42
            )
            
            history = self.text_classifier.fit(
                [X_train_ids, X_train_mask], y_train,
                validation_data=([X_val_ids, X_val_mask], y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            self.text_classifier.save_weights(self.model_weights_path)
            logger.info(f"Model weights saved to {self.model_weights_path}")
            
            return history
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    async def analyze_post(self, post: Dict) -> Dict:
        """Analyze a single post for fake news"""
        try:
            text_score = self.text_classifier.predict(
                self.preprocess_text(post.get('text', '')), verbose=0
            )[0][0]
            
            image_score = 0.0
            if 'image_path' in post:
                image_score = self.image_classifier.predict(
                    self.preprocess_image(post['image_path']), verbose=0
                )[0][0]

            combined_score = 0.7 * text_score + 0.3 * image_score
            
            return {
                'post_id': post.get('id'),
                'fake_probability': float(combined_score),
                'is_fake': combined_score > 0.5,
                'timestamp': post.get('created_at'),
                'text': post.get('text', '')
            }
        except Exception as e:
            logger.error(f"Error analyzing post {post.get('id')}: {str(e)}")
            return {}

    def stream_posts(self):
        """Stream posts from X in real-time using Tweepy v2"""
        try:
            class MyStreamingClient(tweepy.StreamingClient):
                def __init__(self, detector, **kwargs):
                    super().__init__(**kwargs)
                    self.detector = detector

                def on_tweet(self, tweet):
                    post = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                    }
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.detector.analyze_post(post))
                    loop.close()
                    if result.get('is_fake'):
                        self.detector.alert(result)
            stream = MyStreamingClient(
                detector=self,
                bearer_token="AAAAAAAAAAAAAAAAAAAAALx61AEAAAAAMVR4IoYtk6BumgCv9Ohh8vsyEo4%3DSAC1q9PoBmVj7jQUfUfzwhWTiQgyLTORjlHizs86VuUW9Rs1QP"
            )
            stream.add_rules(tweepy.StreamRule(value="news"))
            stream.filter(tweet_fields=["created_at"], threaded=True)
        except Exception as e:
            logger.error(f"Error in streaming posts: {str(e)}")
            raise

    def alert(self, result: Dict):
        """Generate alert for fake news"""
        logger.warning(f"FAKE NEWS DETECTED: Post {result['post_id']} "
                      f"(Probability: {result['fake_probability']:.2f})")
        if 'detections' not in st.session_state:
            st.session_state.detections = []
        st.session_state.detections.append(result)

def create_dashboard(detector: FakeNewsDetector):
    """Create Streamlit dashboard"""
    st.title("Fake News Detection Dashboard")
    
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    
    if st.button("Analyze Sample Post"):
        post = {'id': 'test_123', 'text': 'Sample news article', 'created_at': datetime.now()}
        result = asyncio.run(detector.analyze_post(post))
        st.session_state.detections.append(result)
    
    if st.button("Start X Streaming"):
        threading.Thread(target=detector.stream_posts, daemon=True).start()
        st.write("Streaming started in the background...")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts Analyzed", len(st.session_state.detections))
    col2.metric("Fake News Detected", sum(1 for d in st.session_state.detections if d.get('is_fake')))
    col3.metric("Detection Accuracy", "N/A")
    
    st.subheader("Recent Detections")
    if st.session_state.detections:
        st.dataframe(pd.DataFrame(st.session_state.detections))

def main():
    detector = FakeNewsDetector()
    
    retrain = input("Do you want to retrain the model? (yes/no): ").strip().lower() == 'yes'
    if retrain:
        true_news_path = "True.csv"
        fake_news_path = "Fake.csv"
        if not (os.path.exists(true_news_path) and os.path.exists(fake_news_path)):
            logger.error("Dataset files not found. Please ensure True.csv and Fake.csv are in the directory.")
            return
        logger.info("Loading and preprocessing datasets...")
        input_ids, attention_masks, labels = detector.load_and_preprocess_data(true_news_path, fake_news_path)
        logger.info("Training the model...")
        detector.train_model(input_ids, attention_masks, labels, epochs=3, batch_size=16)
    
    logger.info("Starting X post streaming...")
    detector.stream_posts()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        mode = input("Run mode (stream/dashboard): ").strip().lower()
        if mode == 'stream':
            detector = FakeNewsDetector()  # Instantiate only when needed
            main()
        elif mode == 'dashboard':
            detector = FakeNewsDetector()  # Instantiate only when needed
            create_dashboard(detector)
        else:
            logger.error("Invalid mode. Choose 'stream' or 'dashboard'.")