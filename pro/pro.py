import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
import pandas as pd

nltk.download('vader_lexicon')  # Download the necessary data for sentiment analysis

def extract_text_from_url(url):
    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the article content
        article_tag = soup.find('article')  # Adjust this based on the HTML structure of the webpage

        if article_tag:
            # Extract the text of the article
            article_text = '\n'.join([p.get_text() for p in article_tag.find_all('p')])
            return article_text
        else:
            print(f"Article not found on the page: {url}")
            return None
    else:
        print(f"Failed to retrieve the page {url}. Status code: {response.status_code}")
        return None

def calculate_sentiment_scores(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['compound']

def calculate_subjectivity_score(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound'], sentiment_scores['neu']

def calculate_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    return sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / len(sentences)

def calculate_percentage_complex_words(text):
    words = nltk.word_tokenize(text)
    complex_words = [word for word in words if syllable_count(word) > 2]  # Assuming words with more than 2 syllables are complex
    return (len(complex_words) / len(words)) * 100

def calculate_fog_index(text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    average_sentence_length = len(words) / len(sentences)
    percentage_complex_words = calculate_percentage_complex_words(text)
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)
    return fog_index

def calculate_average_word_length(text):
    words = nltk.word_tokenize(text)
    return sum(len(word) for word in words) / len(words)

def syllable_count(word):
    # A simple function to count syllables in a word
    # This is a basic estimation and may not be accurate for all words
    return max(1, len(re.findall(r'[aeiouAEIOU]+', word)))

def calculate_syllables_per_word(text):
    words = nltk.word_tokenize(text)
    total_syllables = sum(syllable_count(word) for word in words)
    return total_syllables / len(words)

def count_personal_pronouns(text):
    personal_pronouns = ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourselves', 'they', 'them', 'their', 'theirs', 'themselves']
    words = nltk.word_tokenize(text)
    personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
    return personal_pronoun_count

# List of URLs and corresponding names or labels
url_data = [
('https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-3-2/',	'123'),
('https://insights.blackcoffer.com/rise-of-e-health-and-its-impact-on-humans-by-the-year-2030/',	'321'),
('https://insights.blackcoffer.com/rise-of-e-health-and-its-imapct-on-humans-by-the-year-2030-2/',	'2345'),
('https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-2/',	'4321'),
('https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-2-2/',	'432'),
('https://insights.blackcoffer.com/rise-of-chatbots-and-its-impact-on-customer-support-by-the-year-2040/',	'2893.8'),
('https://insights.blackcoffer.com/rise-of-e-health-and-its-imapct-on-humans-by-the-year-2030/',	'3355.6'),
('https://insights.blackcoffer.com/how-does-marketing-influence-businesses-and-consumers/',	'3817.4'),
('https://insights.blackcoffer.com/how-advertisement-increase-your-market-value/',	'4279.2'),
('https://insights.blackcoffer.com/negative-effects-of-marketing-on-society/',	'4741'),
('https://insights.blackcoffer.com/how-advertisement-marketing-affects-business/',	'5202.8'),
('https://insights.blackcoffer.com/rising-it-cities-will-impact-the-economy-environment-infrastructure-and-city-life-by-the-year-2035/',	'5664.6'),
('https://insights.blackcoffer.com/rise-of-ott-platform-and-its-impact-on-entertainment-industry-by-the-year-2030/',	'6126.4'),
('https://insights.blackcoffer.com/rise-of-electric-vehicles-and-its-impact-on-livelihood-by-2040/',	'6588.2'),
('https://insights.blackcoffer.com/rise-of-electric-vehicle-and-its-impact-on-livelihood-by-the-year-2040/',	'7050'),
('https://insights.blackcoffer.com/oil-prices-by-the-year-2040-and-how-it-will-impact-the-world-economy/',	'7511.8'),
('https://insights.blackcoffer.com/an-outlook-of-healthcare-by-the-year-2040-and-how-it-will-impact-human-lives/',	'7973.6'),
('https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/',	'8435.4'),
('https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/',	'8897.2'),
('https://insights.blackcoffer.com/what-jobs-will-robots-take-from-humans-in-the-future/',	'9359'),
('https://insights.blackcoffer.com/will-machine-replace-the-human-in-the-future-of-work/',	'9820.8'),
('https://insights.blackcoffer.com/will-ai-replace-us-or-work-with-us/',	'10282.6'),
('https://insights.blackcoffer.com/man-and-machines-together-machines-are-more-diligent-than-humans-blackcoffe/',	'10744.4'),
('https://insights.blackcoffer.com/in-future-or-in-upcoming-years-humans-and-machines-are-going-to-work-together-in-every-field-of-work/',	'11206.2'),
('https://insights.blackcoffer.com/how-neural-networks-can-be-applied-in-various-areas-in-the-future/',	'11668'),
('https://insights.blackcoffer.com/how-machine-learning-will-affect-your-business/',	'12129.8'),
('https://insights.blackcoffer.com/deep-learning-impact-on-areas-of-e-learning/',	'12591.6'),
('https://insights.blackcoffer.com/how-to-protect-future-data-and-its-privacy-blackcoffer/',	'13053.4'),
('https://insights.blackcoffer.com/how-machines-ai-automations-and-robo-human-are-effective-in-finance-and-banking/',	'13515.2'),
('https://insights.blackcoffer.com/ai-human-robotics-machine-future-planet-blackcoffer-thinking-jobs-workplace/',	'13977'),
('https://insights.blackcoffer.com/how-ai-will-change-the-world-blackcoffer/',	'14438.8'),
('https://insights.blackcoffer.com/future-of-work-how-ai-has-entered-the-workplace/',	'14900.6'),
('https://insights.blackcoffer.com/ai-tool-alexa-google-assistant-finance-banking-tool-future/',	'15362.4'),
('https://insights.blackcoffer.com/ai-healthcare-revolution-ml-technology-algorithm-google-analytics-industrialrevolution/',	'15824.2'),



]

# Create an empty DataFrame to store the results
result_df = pd.DataFrame()

# Loop through each URL and extract data
for url, name in url_data:
    extracted_text = extract_text_from_url(url)

    if extracted_text:
        # Sentiment Analysis
        positive_score, negative_score, compound_score = calculate_sentiment_scores(extracted_text)

        # Subjectivity Analysis
        polarity_score, subjective_score = calculate_subjectivity_score(extracted_text)

        # Sentence Length Analysis
        avg_sentence_length = calculate_sentence_length(extracted_text)

        # Percentage of Complex Words Analysis
        percentage_complex_words = calculate_percentage_complex_words(extracted_text)

        # Fog Index Analysis
        fog_index = calculate_fog_index(extracted_text)

        # Average Word Length Analysis
        avg_word_length = calculate_average_word_length(extracted_text)

        # Syllables per Word Analysis
        syllables_per_word = calculate_syllables_per_word(extracted_text)

        # Personal Pronoun Count
        personal_pronoun_count = count_personal_pronouns(extracted_text)

        # Word Count
        word_count = len(nltk.word_tokenize(extracted_text))

        # Create a DataFrame with the results for the current URL
        data = {
            'URL': [url],
            'Name': [name],
            'Positive Score': [positive_score],
            'Negative Score': [negative_score],
            'Compound Score': [compound_score],
            'Polarity Score': [polarity_score],
            'Subjective Score': [subjective_score],
            'Average Sentence Length': [avg_sentence_length],
            'Percentage of Complex Words': [percentage_complex_words],
            'Fog Index': [fog_index],
            'Average Word Length': [avg_word_length],
            'Syllables per Word': [syllables_per_word],
            'Personal Pronoun Count': [personal_pronoun_count],
            'Word Count': [word_count]
        }

        url_result_df = pd.DataFrame(data)

        # Append the results for the current URL to the overall DataFrame
        result_df = pd.concat([result_df, url_result_df], ignore_index=True)

# ... (Previous code remains unchanged)

# Save the DataFrame to the Excel file
excel_filename = 'output_data_structure.xlxs'
result_df.to_excel(excel_filename, index=False)
print(f"Results saved to {excel_filename}")

