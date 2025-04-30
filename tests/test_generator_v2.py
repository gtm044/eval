import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import pandas as pd
import tempfile
from src.data.generator_v2 import DataGenerator
from src.data.cluster import SemanticCluster

# Sample test data
SAMPLE_CLUSTER = [
    {"text": "Document 1 about AI and machine learning"},
    {"text": "Document 2 about neural networks and deep learning"}
]

SAMPLE_CLUSTER_JSON = json.dumps(SAMPLE_CLUSTER)

SAMPLE_QUESTION = "What are the main topics in machine learning?"

SAMPLE_ANSWERS = [
    "Machine learning encompasses various topics including neural networks and deep learning.",
    "The main topics in machine learning include supervised learning, unsupervised learning, and reinforcement learning.",
    "Machine learning is a field of AI that includes topics like neural networks, decision trees, and clustering algorithms."
]

SAMPLE_CSV_DATA = """text
Document 1 about AI and machine learning
Document 2 about neural networks and deep learning
"""

SAMPLE_JSON_DATA = """[
    {"text": "Document 1 about AI and machine learning"},
    {"text": "Document 2 about neural networks and deep learning"}
]"""

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, if any."""
        self.generator = DataGenerator()
        
        # Mock OpenAI completion response for generate_query
        self.mock_query_completion = MagicMock()
        self.mock_query_completion.choices = [MagicMock()]
        self.mock_query_completion.choices[0].message.content = SAMPLE_QUESTION
        
        # Mock OpenAI completion response for generate_answer
        self.mock_answer_completion = MagicMock()
        self.mock_answer_completion.choices = [MagicMock()]
        self.mock_answer_completion.choices[0].message.content = json.dumps({"answers": SAMPLE_ANSWERS})
        
    @patch('src.data.generator_v2.OpenAI')
    def test_initialization(self, mock_openai_class):
        """Test DataGenerator initialization."""
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        
        mock_openai_class.assert_called_once_with(api_key="test_key")
        self.assertEqual(generator.api_key, "test_key")
        self.assertEqual(generator.client, mock_openai_instance)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_query(self, mock_openai_class):
        """Test generate_query method."""
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = self.mock_query_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_query(SAMPLE_CLUSTER_JSON)
        
        mock_openai_instance.chat.completions.create.assert_called_once()
        self.assertEqual(result, SAMPLE_QUESTION)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_query_json_response(self, mock_openai_class):
        """Test generate_query method with JSON response."""
        mock_openai_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({"question": SAMPLE_QUESTION})
        mock_openai_instance.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_query(SAMPLE_CLUSTER_JSON)
        
        self.assertEqual(result, SAMPLE_QUESTION)

    @patch('src.data.generator_v2.OpenAI')
    def test_generate_query_with_json_array(self, mock_openai_class):
        """Test generate_query method with JSON array response."""
        mock_openai_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({"questions": [SAMPLE_QUESTION]})
        mock_openai_instance.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_query(SAMPLE_CLUSTER_JSON)
        
        self.assertEqual(result, SAMPLE_QUESTION)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_query_with_code_block(self, mock_openai_class):
        """Test generate_query method with code block response."""
        mock_openai_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = f"```json\n{json.dumps({'question': SAMPLE_QUESTION})}\n```"
        mock_openai_instance.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_query(SAMPLE_CLUSTER_JSON)
        
        self.assertEqual(result, SAMPLE_QUESTION)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_query_with_retry(self, mock_openai_class):
        """Test generate_query method with retry for invalid responses."""
        mock_openai_instance = MagicMock()
        mock_completion1 = MagicMock()
        mock_completion1.choices = [MagicMock()]
        # This response will cause a JSONDecodeError, forcing a retry
        mock_completion1.choices[0].message.content = "{invalid json" 
        
        mock_completion2 = MagicMock()
        mock_completion2.choices = [MagicMock()]
        # The second response is valid
        mock_completion2.choices[0].message.content = SAMPLE_QUESTION
        
        mock_openai_instance.chat.completions.create.side_effect = [mock_completion1, mock_completion2]
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_query(SAMPLE_CLUSTER_JSON)
        
        self.assertEqual(result, SAMPLE_QUESTION)
        self.assertEqual(mock_openai_instance.chat.completions.create.call_count, 2)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_answer(self, mock_openai_class):
        """Test generate_answer method."""
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = self.mock_answer_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_answer(SAMPLE_CLUSTER_JSON, SAMPLE_QUESTION)
        
        mock_openai_instance.chat.completions.create.assert_called_once()
        self.assertEqual(result, SAMPLE_ANSWERS)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_answer_with_kwargs(self, mock_openai_class):
        """Test generate_answer method with custom kwargs."""
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = self.mock_answer_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_answer(
            SAMPLE_CLUSTER_JSON, 
            SAMPLE_QUESTION,
            answer_style="detailed",
            answer_format="json",
            tone="formal",
            max_length=100,
            include_citations=True,
            additional_instructions="Be concise",
            custom_instructions="Provide 3 answers"
        )
        
        mock_openai_instance.chat.completions.create.assert_called_once()
        self.assertEqual(result, SAMPLE_ANSWERS)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_answer_with_code_block(self, mock_openai_class):
        """Test generate_answer method with code block response."""
        mock_openai_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = f"```json\n{json.dumps({'answers': SAMPLE_ANSWERS})}\n```"
        mock_openai_instance.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_answer(SAMPLE_CLUSTER_JSON, SAMPLE_QUESTION)
        
        self.assertEqual(result, SAMPLE_ANSWERS)
    
    @patch('src.data.generator_v2.OpenAI')
    def test_generate_answer_with_retry(self, mock_openai_class):
        """Test generate_answer method with retry for invalid responses."""
        mock_openai_instance = MagicMock()
        
        # First response has wrong format
        mock_completion1 = MagicMock()
        mock_completion1.choices = [MagicMock()]
        mock_completion1.choices[0].message.content = json.dumps({"wrong_key": SAMPLE_ANSWERS})
        
        # Second response is valid
        mock_completion2 = MagicMock()
        mock_completion2.choices = [MagicMock()]
        mock_completion2.choices[0].message.content = json.dumps({"answers": SAMPLE_ANSWERS})
        
        mock_openai_instance.chat.completions.create.side_effect = [mock_completion1, mock_completion2]
        mock_openai_class.return_value = mock_openai_instance
        
        generator = DataGenerator(api_key="test_key")
        result = generator.generate_answer(SAMPLE_CLUSTER_JSON, SAMPLE_QUESTION)
        
        self.assertEqual(result, SAMPLE_ANSWERS)
        self.assertEqual(mock_openai_instance.chat.completions.create.call_count, 2)
    
    @patch.object(DataGenerator, 'generate_query')
    @patch.object(DataGenerator, 'generate_answer')
    def test_process_clusters(self, mock_generate_answer, mock_generate_query):
        """Test process_clusters method."""
        mock_generate_query.return_value = SAMPLE_QUESTION
        mock_generate_answer.return_value = SAMPLE_ANSWERS
        
        generator = DataGenerator()
        result = generator.process_clusters([SAMPLE_CLUSTER])
        
        mock_generate_query.assert_called_once()
        mock_generate_answer.assert_called_once()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], SAMPLE_QUESTION)
        self.assertEqual(result[0]["answer"], SAMPLE_ANSWERS)
        self.assertTrue("reference" in result[0])
    
    @patch.object(DataGenerator, 'process_clusters')
    @patch.object(DataGenerator, 'save_results')
    @patch('src.data.generator_v2.get_default_save_directory')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.rmdir')
    def test_process_from_clusters(self, mock_rmdir, mock_remove, mock_listdir, mock_get_default_save_directory, mock_save_results, mock_process_clusters):
        """Test process_from_clusters method."""
        mock_get_default_save_directory.return_value = "/temp/cluster_path"
        mock_listdir.return_value = ["cluster1.json"]
        mock_open_obj = mock_open(read_data=json.dumps(SAMPLE_CLUSTER))
        mock_process_clusters.return_value = [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}]
        
        with patch('builtins.open', mock_open_obj):
            generator = DataGenerator()
            result = generator.process_from_clusters()
            
            mock_process_clusters.assert_called_once()
            mock_remove.assert_called_once()
            mock_rmdir.assert_called_once()
            self.assertEqual(result, [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}])
    
    @patch.object(DataGenerator, 'process_from_clusters')
    @patch.object(DataGenerator, 'save_results')
    @patch.object(SemanticCluster, 'process_json')
    @patch.object(SemanticCluster, 'build_clusters')
    def test_synthesize_from_json(self, mock_build_clusters, mock_process_json, mock_save_results, mock_process_from_clusters):
        """Test synthesize_from_json method."""
        mock_process_from_clusters.return_value = [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}]
        
        generator = DataGenerator()
        result = generator.synthesize_from_json(json_path="test.json", field="text", limit=10)
        
        mock_process_json.assert_called_once_with("test.json", field="text", limit=10)
        mock_build_clusters.assert_called_once()
        mock_process_from_clusters.assert_called_once()
        self.assertEqual(result, [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}])
    
    @patch.object(DataGenerator, 'process_from_clusters')
    @patch.object(DataGenerator, 'save_results')
    @patch.object(SemanticCluster, 'process_csv')
    @patch.object(SemanticCluster, 'build_clusters')
    def test_synthesize_from_csv(self, mock_build_clusters, mock_process_csv, mock_save_results, mock_process_from_clusters):
        """Test synthesize_from_csv method."""
        mock_process_from_clusters.return_value = [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}]
        
        generator = DataGenerator()
        result = generator.synthesize_from_csv(csv_path="test.csv", field="text", limit=10)
        
        mock_process_csv.assert_called_once_with("test.csv", field="text", limit=10)
        mock_build_clusters.assert_called_once()
        mock_process_from_clusters.assert_called_once()
        self.assertEqual(result, [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}])
    
    @patch.object(DataGenerator, 'process_from_clusters')
    @patch.object(DataGenerator, 'save_results')
    @patch.object(SemanticCluster, 'build_clusters')
    def test_synthesize_from_text(self, mock_build_clusters, mock_save_results, mock_process_from_clusters):
        """Test synthesize_from_text method."""
        mock_process_from_clusters.return_value = [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}]
        texts = ["Document 1", "Document 2"]
        
        generator = DataGenerator()
        result = generator.synthesize_from_text(texts=texts)
        
        mock_build_clusters.assert_called_once()
        mock_process_from_clusters.assert_called_once()
        self.assertEqual(result, [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}])

    def test_save_results(self):
        """Test save_results method."""
        test_data = [{"question": SAMPLE_QUESTION, "answer": SAMPLE_ANSWERS, "reference": ["ref1"]}]
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
            generator = DataGenerator()
            result_path = generator.save_results(test_data, output_path=temp_path)
            
            with open(result_path, 'r') as f:
                saved_data = json.load(f)
                
            self.assertEqual(saved_data, test_data)
            
            # Clean up
            os.remove(temp_path)

if __name__ == '__main__':
    unittest.main() 