import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
from dateutil import parser


nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')


# Define the allowed locations
ALLOWED_LOCATIONS = {
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona"
}


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_reviews(self, location=None, start_date=None, end_date=None):
        # Filter reviews based on location
        filtered_reviews = reviews
        if location:
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

        # Filter reviews based on timestamp
        if start_date or end_date:
            start_date = parser.parse(start_date) if start_date else None
            end_date = parser.parse(end_date) if end_date else None
            filtered_reviews = [
                review for review in filtered_reviews
                if (not start_date or datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date) and
                    (not end_date or datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date)
            ]

        # Add sentiment analysis and sort by compound score
        for review in filtered_reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
        filtered_reviews.sort(key=lambda r: r['sentiment']['compound'], reverse=True)

        return filtered_reviews

    
    
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query = parse_qs(environ['QUERY_STRING'])
            location = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]
            
            #print(query)
            #print(location)
            #print(start_date)
            #print(end_date)

            
            filtered_reviews = self.filter_reviews(location, start_date, end_date)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
                params = parse_qs(request_body)

                location = params.get('Location', [None])[0]
                review_body = params.get('ReviewBody', [None])[0]

                
                if not location or not review_body:
                    start_response("400 Bad Request", [
                        ("Content-Type", "text/plain")
                    ])
                    return [b"Missing required parameters: 'Location' and 'ReviewBody'"]

                # Validate location
                if location not in ALLOWED_LOCATIONS:
                    response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]

                    

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                review_id = str(uuid.uuid4())
                sentiment = self.analyze_sentiment(review_body)

                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            except Exception as e:
                start_response("500 Internal Server Error", [
                    ("Content-Type", "text/plain")
                ])
                return [b"Internal Server Error"]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()