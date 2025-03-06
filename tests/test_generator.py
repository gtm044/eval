import unittest
from src.data.generator import SyntheticDataGenerator
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

class TestSyntheticQueryPrompt(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.openai = OpenAI(api_key=self.api_key)
        self.similarity_threshold = 0.8

    # Helper function to get embeddings
    def get_embedding(self, text):
        response = self.openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    # Helper function to calculate cosine similarity
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Helper function to check if a text is similar to any in a list
    def is_similar_to_any(self, text, possible_texts):
        text_embedding = self.get_embedding(text)
        for possible_text in possible_texts:
            possible_embedding = self.get_embedding(possible_text)
            similarity = self.cosine_similarity(text_embedding, possible_embedding)
            if similarity >= self.similarity_threshold:
                return True
        return False

    # Helper function to call generate_questions and return a single question
    def get_single_question(self, doc, metadata=None):
        docs = [doc]  # Wrap in a list
        generator = SyntheticDataGenerator()
        questions = generator.generate_questions(docs, metadata)  # Call function
        return questions[0]  # Unwrap first result

    # 1. TRUE INFORMATION CASES
    def test_true_statement(self):
        doc = "Albert Einstein was born in 1879."
        expected = "When was Albert Einstein born?"
        question = self.get_single_question(doc)
        question_embedding = self.get_embedding(question)
        expected_embedding = self.get_embedding(expected)
        similarity = self.cosine_similarity(question_embedding, expected_embedding)
        self.assertGreaterEqual(similarity, self.similarity_threshold)

    def test_multiple_true_statements(self):
        doc = "The Eiffel Tower is in Paris. The moon orbits the Earth."
        possible_expected = ["Where is the Eiffel Tower located?", "What does the moon orbit?"]
        question = self.get_single_question(doc)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    def test_json_true_information(self):
        doc = '{"scientist": "Einstein", "discovery": "Theory of Relativity"}'
        metadata = '{"dataset": "scientific_discoveries"}'
        possible_expected = ["What did Einstein discover?", "Who discovered the theory of relativity?"]
        question = self.get_single_question(doc, metadata)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    def test_json_with_metadata(self):
        doc = '{"city": "Paris", "country": "France"}'
        metadata = '{"dataset": "geography"}'
        possible_expected = ["Which country is Paris in?", "Which country is Paris located in?"]
        question = self.get_single_question(doc, metadata)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    # 2. FALSE INFORMATION CASES
    def test_false_statement(self):
        doc = "The Great Wall of China is in Brazil."
        expected = "Where is the Great Wall of China located?"
        question = self.get_single_question(doc)
        question_embedding = self.get_embedding(question)
        expected_embedding = self.get_embedding(expected)
        similarity = self.cosine_similarity(question_embedding, expected_embedding)
        self.assertGreaterEqual(similarity, self.similarity_threshold)

    def test_multiple_false_statements(self):
        doc = "Water is dry. The sun is cold."
        possible_expected = ["What is the state of water?", "What is the temperature of the sun?"]
        question = self.get_single_question(doc)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    def test_json_false_information(self):
        doc = '{"planet": "Sun", "gravity": "None"}'
        metadata = '{"dataset": "astronomy", "Planet": "Name of a planet", "Gravity": "Whether a planet has gravity or not"}'
        expected = "Does the Sun have gravity?"
        question = self.get_single_question(doc, metadata)
        question_embedding = self.get_embedding(question)
        expected_embedding = self.get_embedding(expected)
        similarity = self.cosine_similarity(question_embedding, expected_embedding)
        self.assertGreaterEqual(similarity, self.similarity_threshold)

    def test_json_false_with_metadata(self):
        doc = '{"country": "Atlantis", "capital": "Poseidon City"}'
        metadata = '{"dataset": "world_countries"}'
        expected = "What is the capital of Atlantis?"
        question = self.get_single_question(doc, metadata)
        question_embedding = self.get_embedding(question)
        expected_embedding = self.get_embedding(expected)
        similarity = self.cosine_similarity(question_embedding, expected_embedding)
        self.assertGreaterEqual(similarity, self.similarity_threshold)

    # 3. MIXED TRUE & FALSE INFORMATION CASES
    def test_mixed_information(self):
        doc = "The Earth is round. The sun revolves around the Earth."
        possible_expected = ["What is the shape of the Earth?", "What does the sun revolve around?"]
        question = self.get_single_question(doc)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    def test_json_mixed_information(self):
        doc = '{"animal": "Elephant", "wings": "Yes"}'
        metadata = '{"dataset": "animal_characteristics"}'
        possible_expected = ["Does an elephant have wings?", "Do elephants have wings?"]
        question = self.get_single_question(doc, metadata)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    # 4. INSUFFICIENT INFORMATION CASES
    def test_vague_text(self):
        doc = "It is what it is."
        expected = "NO_QUESTION_POSSIBLE"
        self.assertEqual(self.get_single_question(doc), expected)

    def test_json_insufficient_data(self):
        doc = '{"object": "Unknown", "value": "None"}'
        metadata = '{"dataset": "generic_objects"}'
        expected = "NO_QUESTION_POSSIBLE"
        self.assertEqual(self.get_single_question(doc, metadata), expected)

    # 5. EDGE CASES
    def test_empty_document(self):
        doc = ""
        expected = "NO_QUESTION_POSSIBLE"
        self.assertEqual(self.get_single_question(doc), expected)

    def test_gibberish_text(self):
        doc = "asjdhakjsd hsjh!@# 12312"
        expected = "NO_QUESTION_POSSIBLE"
        self.assertEqual(self.get_single_question(doc), expected)

    def test_long_convoluted_text(self):
        doc = "Many years ago, a man named George Washington led a country. Meanwhile, some say that the moon is made of cheese. Some argue that the color blue is actually green if seen under red light."
        possible_expected = [
            "Who led a country many years ago?",
            "What is the moon made of?",
            "What happens to blue under red light?"
        ]
        question = self.get_single_question(doc)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    def test_multiple_questions_possible(self):
        doc = "The Pacific Ocean is the largest ocean. Mount Everest is the highest mountain."
        possible_expected = [
            "Which is the largest ocean?",
            "Which is the highest mountain?"
        ]
        question = self.get_single_question(doc)
        self.assertTrue(self.is_similar_to_any(question, possible_expected))

    # 6. MULTIPLE DOCUMENTS TEST
    def test_multiple_documents(self):
        docs = [
            "The Amazon rainforest is in South America.",
            "The capital of Japan is Tokyo.",
            "The moon is a satellite of Earth."
        ]
        possible_questions = [
            ["Where is the Amazon rainforest?", "Where is the Amazon rainforest located?"],
            ["What is the capital of Japan?"],
            ["What is the moon a satellite of?", "What is the moon in relation to Earth?"]
        ]
        generator = SyntheticDataGenerator()
        questions = generator.generate_questions(docs)
        for i, question in enumerate(questions):
            self.assertTrue(self.is_similar_to_any(question, possible_questions[i]))

    # 7. TEST FOR BANNED PHRASES
    def test_no_banned_phrases(self):
        doc = "The Statue of Liberty is located in New York."
        question = self.get_single_question(doc)
        
        # List of banned phrases from prompts.py
        banned_phrases = [
            "according to the document",
            "in the document",
            "the document states",
            "what does the document say",
            "based on the document",
            "the text mentions",
            "as mentioned in"
        ]
        
        # Check that none of the banned phrases appear in the question
        for phrase in banned_phrases:
            self.assertNotIn(phrase.lower(), question.lower())
            
        # Also check for common variations
        variations = [
            "the document mentions",
            "as stated in the document",
            "according to the text",
            "the passage states",
            "as per the document"
        ]
        
        for variation in variations:
            self.assertNotIn(variation.lower(), question.lower())
            
    def test_no_banned_phrases_json(self):
        doc = '{"landmark": "Taj Mahal", "location": "India"}'
        metadata = '{"dataset": "world_landmarks"}'
        question = self.get_single_question(doc, metadata)
        
        # List of banned phrases from prompts.py
        banned_phrases = [
            "according to the document",
            "in the document",
            "the document states",
            "what does the document say",
            "based on the document",
            "the text mentions",
            "as mentioned in",
            "according to the json",
            "in the json",
            "the json states"
        ]
        
        # Check that none of the banned phrases appear in the question
        for phrase in banned_phrases:
            self.assertNotIn(phrase.lower(), question.lower())

if __name__ == "__main__":
    unittest.main()