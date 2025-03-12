import unittest
import os
import json
import tempfile
import uuid
from src.data.dataset import EvalDataset

class TestEvalDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data that will be used across multiple tests"""
        self.valid_data = {
            "questions": [
                "What is the capital of France?", 
                "Who is the president of the USA?", 
                "What is the largest planet in our solar system?"
            ],
            "answers": [
                ["Paris"], 
                ["Joe Biden"], 
                ["Jupiter"]
            ],
            "responses": [
                "Paris is the capital of France", 
                "Joe Biden is the president", 
                "Jupiter is the largest planet in our solar system"
            ],
            "reference_contexts": [
                "Paris is the capital of France", 
                "Joe Biden is the 46th president of the USA", 
                "Jupiter is the largest planet in our solar system, with a mass of 1.9 × 10^27 kg"
            ],
            "retrieved_contexts": [
                ["Paris is in France"], 
                ["Joe Biden was elected in 2020"], 
                ["Jupiter has a diameter of approximately 139,820 kilometers"]
            ]
        }
        
    def test_init_with_valid_data(self):
        """Test initialization with valid data"""
        dataset = EvalDataset(**self.valid_data)
        self.assertEqual(len(dataset.questions), 3)
        self.assertEqual(len(dataset.answers), 3)
        self.assertEqual(len(dataset.responses), 3)
        self.assertEqual(len(dataset.reference_contexts), 3)
        self.assertEqual(len(dataset.retrieved_contexts), 3)
        self.assertIsNotNone(dataset.dataset_id)
        
    def test_init_with_empty_data(self):
        """Test initialization with empty data"""
        dataset = EvalDataset()
        self.assertIsNone(dataset.questions)
        self.assertIsNone(dataset.answers)
        self.assertIsNone(dataset.responses)
        self.assertIsNone(dataset.reference_contexts)
        self.assertIsNone(dataset.retrieved_contexts)
        self.assertIsNotNone(dataset.dataset_id)
        
    def test_init_with_partial_data(self):
        """Test initialization with partial data"""
        partial_data = {
            "questions": ["What is the capital of France?", "Who is the president of the USA?"],
            "answers": [["Paris"], ["Joe Biden"]]
        }
        dataset = EvalDataset(**partial_data)
        self.assertEqual(len(dataset.questions), 2)
        self.assertEqual(len(dataset.answers), 2)
        self.assertIsNone(dataset.responses)
        self.assertIsNone(dataset.reference_contexts)
        self.assertIsNone(dataset.retrieved_contexts)
        
    def test_length_validation(self):
        """Test validation that all lists must be of the same length"""
        invalid_data = {
            "questions": ["What is the capital of France?", "Who is the president of the USA?"],
            "answers": [["Paris"]],  # Only one answer
            "responses": ["Paris is the capital of France", "Joe Biden is the president"]
        }
        with self.assertRaises(ValueError):
            EvalDataset(**invalid_data)
            
    def test_to_json_with_complete_data(self):
        """Test to_json method with complete data"""
        dataset = EvalDataset(**self.valid_data)
        json_data = dataset.to_json()
        
        self.assertEqual(len(json_data), 3)
        self.assertEqual(json_data[0]["question"], "What is the capital of France?")
        self.assertEqual(json_data[0]["answers"], ["Paris"])
        self.assertEqual(json_data[1]["response"], "Joe Biden is the president")
        
    def test_to_json_with_file_output(self):
        """Test to_json method with file output"""
        dataset = EvalDataset(**self.valid_data)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        try:
            dataset.to_json(filename=temp_filename)
            
            # Verify file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filename))
            
            with open(temp_filename, 'r') as f:
                loaded_data = json.load(f)
                
            self.assertEqual(len(loaded_data), 3)
            self.assertEqual(loaded_data[0]["question"], "What is the capital of France?")
            self.assertEqual(loaded_data[1]["question"], "Who is the president of the USA?")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
    def test_to_json_with_partial_data(self):
        """Test to_json method with partial data"""
        partial_data = {
            "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        }
        dataset = EvalDataset(**partial_data)
        json_data = dataset.to_json()
        
        self.assertEqual(len(json_data), 2)
        self.assertEqual(json_data[0]["question"], "What is the capital of France?")
        self.assertIsNone(json_data[0]["answers"])
        self.assertIsNone(json_data[0]["response"])
        
    def test_from_json_with_complete_data(self):
        """Test from_json class method with complete data"""
        dataset = EvalDataset(**self.valid_data)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        try:
            dataset.to_json(filename=temp_filename)
            
            # Load the dataset from the JSON file
            loaded_dataset = EvalDataset.from_json(temp_filename)
            
            # Verify the loaded dataset
            self.assertEqual(len(loaded_dataset.questions), 3)
            self.assertEqual(loaded_dataset.questions[0], "What is the capital of France?")
            self.assertEqual(loaded_dataset.answers[0], ["Paris"])
            self.assertEqual(loaded_dataset.responses[1], "Joe Biden is the president")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
    def test_from_json_with_partial_data(self):
        """Test from_json class method with partial data"""
        # Create a JSON file with partial data
        partial_data = [
            {
                "question": "What is the capital of France?",
                "answers": ["Paris"],
                "response": None,
                "reference_context": None,
                "retrieved_contexts": None
            },
            {
                "question": "Who is the president of the USA?",
                "answers": ["Joe Biden"],
                "response": None,
                "reference_context": None,
                "retrieved_contexts": None
            }
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_filename = temp_file.name
            with open(temp_filename, 'w') as f:
                json.dump(partial_data, f)
            
        try:
            # Load the dataset from the JSON file
            loaded_dataset = EvalDataset.from_json(temp_filename)
            
            # Verify the loaded dataset
            self.assertEqual(len(loaded_dataset.questions), 2)
            self.assertEqual(loaded_dataset.questions[0], "What is the capital of France?")
            self.assertEqual(loaded_dataset.answers[0], ["Paris"])
            self.assertIsNone(loaded_dataset.responses)
            self.assertIsNone(loaded_dataset.reference_contexts)
            self.assertIsNone(loaded_dataset.retrieved_contexts)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
    def test_dataset_id_generation(self):
        """Test that dataset_id is generated as a UUID string"""
        dataset = EvalDataset()
        # Verify that dataset_id is a string and can be parsed as a UUID
        self.assertIsInstance(dataset.dataset_id, str)
        try:
            uuid_obj = uuid.UUID(dataset.dataset_id)
            self.assertIsInstance(uuid_obj, uuid.UUID)
        except ValueError:
            self.fail("dataset_id is not a valid UUID string")

if __name__ == '__main__':
    unittest.main()