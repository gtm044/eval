import streamlit as st
import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
from src.data.dataset import EvalDataset
from src.data.generator import SyntheticDataGenerator, init_generator
from src.controller.options import ExperimentOptions
from src.controller.manager import Experiment
from src.evaluator.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    context_precision, 
    answer_correctness, 
    avg_chunk_size, 
    context_similarity, 
    context_score,
    named_entity_score,
    llm_grading
)
from src.evaluator.validation import ValidationEngine
from src.data.load import LoadOperator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

STREAMLIT_ENV_FILE = ".env.streamlit"

# Load environment variables
load_dotenv()

# Initialize session state for button click
if "saved" not in st.session_state:
    st.session_state.saved = False

# Set page config
st.set_page_config(
    page_title="RAG Evaluation Framework",
    page_icon="🧪",
    layout="wide"
)

# Sidebar 
with st.sidebar:
    st.title("RAG Evaluation Framework")
    st.markdown("A comprehensive framework for evaluating RAG systems")
        
    option = st.radio(
        "Choose functionality:",
        ["Generate Data", "Evaluate", "Experiment", "Result Analysis", "Compare Experiments", "Docs"]
    )

    # Expander for Couchbase Credentials
    with st.expander("Provide Couchbase Credentials (optional)", expanded=False):
        couchbase_url = st.text_input("Cluster URL", type="default")
        couchbase_username = st.text_input("Username", type="default")
        couchbase_password = st.text_input("Password", type="password")
        couchbase_bucket = st.text_input("Bucket", type="default")
        couchbase_scope = st.text_input("Scope", type="default")
        couchbase_collection = st.text_input("Collection", type="default")

        # Save button with session state tracking
        if st.button("Save"):
            # Save to .env.streamlit file
            with open(STREAMLIT_ENV_FILE, "w") as f:
                f.write(f"cluster_url={couchbase_url}\n")
                f.write(f"cb_username={couchbase_username}\n")
                f.write(f"cb_password={couchbase_password}\n")
                f.write(f"bucket={couchbase_bucket}\n")
                f.write(f"scope={couchbase_scope}\n")
                f.write(f"collection={couchbase_collection}\n")    
                os.environ["cluster_url"] = couchbase_url
                os.environ["cb_username"] = couchbase_username
                os.environ["cb_password"] = couchbase_password
                os.environ["bucket"] = couchbase_bucket
                os.environ["scope"] = couchbase_scope
                os.environ["collection"] = couchbase_collection

            # Set session state to True to prevent multiple clicks
            st.session_state.saved = True
            st.rerun()
            
    if st.session_state.saved:
        st.success("Couchbase credentials saved")

# Main content

            
if option == "Generate Data":
    st.title("Synthetic Data Generation")
    
    # Initialize session state variables for Data Generation tab
    if 'data_generation_completed' not in st.session_state:
        st.session_state.data_generation_completed = False
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'generated_dataset' not in st.session_state:
        st.session_state.generated_dataset = None
    
    # Add generation type selection
    generation_type = st.radio(
        "Select generation method:",
        ["Single-hop (faster, simpler)", "Multi-hop (complex, multi-document reasoning)"],
        help="Single-hop generates QA pairs from individual documents. Multi-hop generates QA pairs that require reasoning across multiple documents."
    )
    
    # Set multi_hop flag based on selection
    multi_hop = generation_type == "Multi-hop (complex, multi-document reasoning)"
    
    # Add information about the generation types
    st.info("""
    **About Generation Methods:**
    
    - **Single-hop generation** creates questions that can be answered from a single document. This is faster and works well for most use cases.
    
    - **Multi-hop generation** creates more complex questions that require reasoning across multiple documents. The system will automatically cluster related documents to create questions that need information from multiple sources.
    """)
    
    # Input tabs
    tab1, tab2 = st.tabs(["From CSV/JSON", "From Raw Text"])
    
    with tab1:
        st.header("Generate from CSV/JSON")
        file_format = st.selectbox("File format", ["csv", "json"])
        
        uploaded_file = st.file_uploader(f"Upload {file_format} file", type=[file_format], key="csv_json_file_uploader")
        
        if file_format == "json" and uploaded_file is not None:
            field = st.text_input("Field name in JSON to use (optional)")
        
        metadata = st.text_area("Metadata description (helps in generation)", 
                               placeholder="E.g., 'Document contains product descriptions with fields: name, description, price, and category.'")
        
        limit = st.number_input("Limit number of rows to process (optional)", min_value=1, max_value=100, value=10)
        
        # Add customization options in expanders
        with st.expander("Question Generation Options", expanded=False):
            q_custom_instructions = st.text_area(
                "Custom instructions for question generation",
                placeholder="Enter custom instructions to override the default question generation behavior"
            )
            
            example_questions_text = st.text_area(
                "Example document-question pairs",
                placeholder="What is the capital of France?\nHow tall is the Eiffel Tower?\nWhen was the Declaration of Independence signed?"
            )
            
            example_questions = [q.strip() for q in example_questions_text.split('\n') if q.strip()] if example_questions_text else None
        
        with st.expander("Answer Generation Options", expanded=False):
            answer_style = st.text_area(
                "Answer style instructions",
                placeholder="E.g., 'Use a conversational tone with simple language'"
            )
            
            answer_format = st.text_area(
                "Answer format instructions",
                placeholder="E.g., 'Structure answers as bullet points'"
            )
            
            tone = st.text_input(
                "Tone for answers",
                placeholder="E.g., 'professional', 'casual', 'academic'"
            )
            
            max_length = st.number_input(
                "Maximum answer length (words)", 
                min_value=10, 
                value=None,
                help="Maximum word count for each answer"
            )
            
            additional_instructions = st.text_area(
                "Additional instructions",
                placeholder="Any additional instructions for answer generation"
            )
            
        if uploaded_file is not None:
            if st.button("Generate Data", key="generate_csv_json"):
                try:
                    # Save the uploaded file
                    with open(f"temp_upload.{file_format}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Write metadata to file
                    with open("temp_metadata.txt", "w") as f:
                        f.write(metadata)
                    
                    # Initialize the generator based on user selection
                    generator = init_generator(multi_hop=multi_hop)
                    
                    with st.spinner("Generating synthetic data..."):
                        start_time = time.time()
                        if file_format == "csv":
                            df = pd.read_csv("temp_upload.csv")
                            df = df[:limit]
                            temp_path = "temp_cleaned_upload.csv"
                            df.to_csv(temp_path, index=False)   
                            
                            if multi_hop:
                                # Use multi-hop generator
                                generated_data = generator.synthesize_from_csv(
                                    csv_path=temp_path,
                                    metadata=metadata,
                                    limit=limit,
                                    # Answer generation parameters
                                    answer_style=answer_style if answer_style else None,
                                    answer_format=answer_format if answer_format else None,
                                    tone=tone if tone else None,
                                    max_length=max_length,
                                    additional_instructions=additional_instructions if additional_instructions else None,
                                    custom_instructions=None,
                                )
                                # Convert to format expected by the app
                                questions = []
                                answers = []
                                reference_contexts = []
                                for item in generated_data:
                                    questions.append(item["question"])
                                    answers.append(item["answer"])
                                    reference_contexts.append(item["reference"])
                                
                                generated_data = {
                                    "questions": questions,
                                    "answers": answers,
                                    "reference_contexts": reference_contexts
                                }
                            else:
                                # Use single-hop generator
                                generated_data = generator.synthesize_from_csv(
                                    path=temp_path,
                                    metadata=metadata,
                                    # Question generation parameters
                                    question_custom_instructions=q_custom_instructions if q_custom_instructions else None,
                                    example_questions=example_questions,
                                    # Answer generation parameters
                                    answer_style=answer_style if answer_style else None,
                                    answer_format=answer_format if answer_format else None,
                                    tone=tone if tone else None,
                                    max_length=max_length,
                                    additional_instructions=additional_instructions if additional_instructions else None,
                                )
                            os.remove(temp_path)
                        else:  # json
                            field_param = field if field else None
                            
                            if multi_hop:
                                # Use multi-hop generator
                                generated_data = generator.synthesize_from_json(
                                    json_path="temp_upload.json",
                                    field=field_param,
                                    limit=limit,
                                    metadata=metadata,
                                    # Answer generation parameters
                                    answer_style=answer_style if answer_style else None,
                                    answer_format=answer_format if answer_format else None,
                                    tone=tone if tone else None,
                                    max_length=max_length,
                                    additional_instructions=additional_instructions if additional_instructions else None,
                                    custom_instructions=None,
                                )
                                # Convert to format expected by the app
                                questions = []
                                answers = []
                                reference_contexts = []
                                for item in generated_data:
                                    questions.append(item["question"])
                                    answers.append(item["answer"])
                                    reference_contexts.append(item["reference"])
                                
                                generated_data = {
                                    "questions": questions,
                                    "answers": answers,
                                    "reference_contexts": reference_contexts
                                }
                            else:
                                # Single-hop: Load documents and use the single-hop generator
                                documents = generator.load_from_json(
                                    path="temp_upload.json",
                                    field=field_param
                                )
                                # Limit the number of documents
                                documents = documents[:limit]
                                generated_data = generator.synthesize_from_text(
                                    documents=documents,
                                    metadata=metadata,
                                    # Question generation parameters
                                    question_custom_instructions=q_custom_instructions if q_custom_instructions else None,
                                    example_questions=example_questions,
                                    # Answer generation parameters
                                    answer_style=answer_style if answer_style else None,
                                    answer_format=answer_format if answer_format else None,
                                    tone=tone if tone else None,
                                    max_length=max_length,
                                    additional_instructions=additional_instructions if additional_instructions else None,
                                )
                        generation_time = time.time() - start_time
                    
                    # Store generated data in session state
                    st.session_state.data_generation_completed = True
                    st.session_state.generated_data = generated_data
                    
                    dataset = {
                        "questions": st.session_state.generated_data["questions"],
                        "answers": st.session_state.generated_data["answers"],
                        "reference_contexts": st.session_state.generated_data["reference_contexts"]
                    }
                    
                    st.session_state.generated_dataset = dataset
                    
                    # Save to JSON for download
                    with open("generated_dataset.json", "w") as f:
                        json.dump(st.session_state.generated_dataset, f, indent=4)
                    
                    # Clean up
                    os.remove(f"temp_upload.{file_format}")
                    os.remove("temp_metadata.txt")
                    
                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
        
        # Display generated data if available
        if st.session_state.data_generation_completed and st.session_state.generated_data:
            generated_data = st.session_state.generated_data 
            
            # Display success message
            st.success(f"Successfully generated {len(generated_data['questions'])} question-answer pairs!")
            
            # Display generation time if available
            if 'generation_time' in locals():
                st.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Create DataFrame for display
            data_for_display = []
            for i, (q, a, ctx) in enumerate(zip(
                    generated_data["questions"], 
                    generated_data["answers"], 
                    generated_data["reference_contexts"])):
                data_for_display.append({
                    "Question": q,
                    "Answers": str(a),
                    "Context": ctx[:100] + "..." if len(ctx) > 100 else ctx
                })
            
            st.dataframe(pd.DataFrame(data_for_display))
            
            # Download button
            with open("generated_dataset.json", "r") as f:
                st.download_button(
                    label="Download Dataset",
                    data=f,
                    file_name="rag_eval_dataset.json",
                    mime="application/json"
                )
    
    with tab2:
        st.header("Generate from Raw Text")
        
        raw_text = st.text_area("Enter raw text documents (one per line)", height=200)
        metadata = st.text_area("Metadata description (helps in generation)", 
                                placeholder="E.g., 'Documents are technical articles about machine learning.'",
                                key="raw_text_metadata")
        
        # Add customization options in expanders
        with st.expander("Question Generation Options", expanded=False):
            q_custom_instructions_raw = st.text_area(
                "Custom instructions for question generation",
                placeholder="Enter custom instructions to override the default question generation behavior",
                key="q_custom_instructions_raw"
            )
            
            example_questions_text_raw = st.text_area(
                "Example document-question pairs",
                placeholder="What is the capital of France?\nHow tall is the Eiffel Tower?\nWhen was the Declaration of Independence signed?",
                key="example_questions_text_raw"
            )
            
            example_questions_raw = [q.strip() for q in example_questions_text_raw.split('\n') if q.strip()] if example_questions_text_raw else None
        
        with st.expander("Answer Generation Options", expanded=False):
            answer_style_raw = st.text_area(
                "Answer style instructions",
                placeholder="E.g., 'Use a conversational tone with simple language'",
                key="answer_style_raw"
            )
            
            answer_format_raw = st.text_area(
                "Answer format instructions",
                placeholder="E.g., 'Structure answers as bullet points'",
                key="answer_format_raw"
            )
            
            tone_raw = st.text_input(
                "Tone for answers",
                placeholder="E.g., 'professional', 'casual', 'academic'",
                key="tone_raw"
            )
            
            max_length_raw = st.number_input(
                "Maximum answer length (words)", 
                min_value=10, 
                value=None,
                help="Maximum word count for each answer",
                key="max_length_raw"
            )
            
            include_citations_raw = st.checkbox(
                "Include citations",
                help="Whether to include citations to specific parts of the document",
                key="include_citations_raw"
            )
            
            additional_instructions_raw = st.text_area(
                "Additional instructions",
                placeholder="Any additional instructions for answer generation",
                key="additional_instructions_raw"
            )
            
            a_custom_instructions_raw = st.text_area(
                "Custom instructions for answer generation",
                placeholder="Enter custom instructions to override the default answer generation behavior",
                key="a_custom_instructions_raw"
            )
        
        if st.button("Generate Data", key="generate_raw_text") and raw_text:
            try:
                # Parse input into documents
                documents = [doc.strip() for doc in raw_text.split("\n") if doc.strip()]
                
                # Initialize the generator based on user selection
                generator = init_generator(multi_hop=multi_hop)
                
                with st.spinner("Generating synthetic data..."):
                    start_time = time.time()
                    
                    if multi_hop:
                        # Use multi-hop generator
                        generated_data = generator.synthesize_from_text(
                            texts=documents,
                            metadata=metadata,
                            # Answer generation parameters
                            answer_style=answer_style_raw if answer_style_raw else None,
                            answer_format=answer_format_raw if answer_format_raw else None,
                            tone=tone_raw if tone_raw else None,
                            max_length=max_length_raw,
                            include_citations=include_citations_raw,
                            additional_instructions=additional_instructions_raw if additional_instructions_raw else None,
                            custom_instructions=a_custom_instructions_raw if a_custom_instructions_raw else None
                        )
                        # Convert to format expected by the app
                        questions = []
                        answers = []
                        reference_contexts = []
                        for item in generated_data:
                            questions.append(item["question"])
                            answers.append(item["answer"])
                            reference_contexts.append(item["reference"])
                        
                        generated_data = {
                            "questions": questions,
                            "answers": answers,
                            "reference_contexts": reference_contexts
                        }
                    else:
                        # Use single-hop generator
                        generated_data = generator.synthesize_from_text(
                            documents=documents,
                            metadata=metadata,
                            # Question generation parameters
                            question_custom_instructions=q_custom_instructions_raw if q_custom_instructions_raw else None,
                            example_questions=example_questions_raw,
                            # Answer generation parameters
                            answer_style=answer_style_raw if answer_style_raw else None,
                            answer_format=answer_format_raw if answer_format_raw else None,
                            tone=tone_raw if tone_raw else None,
                            max_length=max_length_raw,
                            include_citations=include_citations_raw,
                            additional_instructions=additional_instructions_raw if additional_instructions_raw else None,
                            answer_custom_instructions=a_custom_instructions_raw if a_custom_instructions_raw else None
                        )
                    generation_time = time.time() - start_time
                
                # Store generated data in session state
                st.session_state.data_generation_completed = True
                st.session_state.generated_data = generated_data
                
                dataset = {
                    "questions": st.session_state.generated_data["questions"],
                    "answers": st.session_state.generated_data["answers"],
                    "reference_contexts": st.session_state.generated_data["reference_contexts"]
                }
                st.session_state.generated_dataset = dataset
                                    
                # Save to JSON for download
                with open("generated_dataset.json", "w") as f:
                    json.dump(st.session_state.generated_dataset, f, indent=4)
                
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
        
        # Display generated data if available
        if st.session_state.data_generation_completed and st.session_state.generated_data:
            generated_data = st.session_state.generated_data
            
            # Display success message
            st.success(f"Successfully generated {len(generated_data['questions'])} question-answer pairs!")
            
            # Display generation time if available
            if 'generation_time' in locals():
                st.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Create DataFrame for display
            data_for_display = []
            for i, (q, a, ctx) in enumerate(zip(
                    generated_data["questions"], 
                    generated_data["answers"], 
                    generated_data["reference_contexts"])):
                data_for_display.append({
                    "Question": q,
                    "Answers": str(a),
                    "Context": ctx[:100] + "..." if len(ctx) > 100 else ctx
                })
            
            st.dataframe(pd.DataFrame(data_for_display))
            
            # Offer download button
            with open("generated_dataset.json", "r") as f:
                st.download_button(
                    key = "download_generated_dataset",
                    label="Download Dataset",
                    data=f,
                    file_name="rag_eval_dataset.json",
                    mime="application/json"
                )
    
    # Add button to clear generated data
    if st.session_state.data_generation_completed:
        if st.button("Generate New Data", key="clear_generated_data"):
            st.session_state.data_generation_completed = False
            st.session_state.generated_data = None
            st.session_state.generated_dataset = None
            # Remove temporary files if they exist
            if os.path.exists("generated_dataset.json"):
                os.remove("generated_dataset.json")

elif option == "Evaluate":
    st.title("RAG System Evaluation")
    
    # Initialize session state
    if 'evaluation_dataset_loaded' not in st.session_state:
        st.session_state.evaluation_dataset_loaded = False
    if 'evaluation_dataset' not in st.session_state:
        st.session_state.evaluation_dataset = None
    if 'evaluation_results_loaded' not in st.session_state:
        st.session_state.evaluation_results_loaded = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'evaluation_avg_metrics' not in st.session_state:
        st.session_state.evaluation_avg_metrics = None
    if 'evaluation_metrics_objs' not in st.session_state:
        st.session_state.evaluation_metrics_objs = None
    
    # Clear evaluation state 
    if st.button("Clear Evaluation", key="clear_evaluation"):
        st.session_state.evaluation_dataset_loaded = False
        st.session_state.evaluation_dataset = None
        st.session_state.evaluation_results_loaded = False
        st.session_state.evaluation_results = None
        st.session_state.evaluation_avg_metrics = None
        st.session_state.evaluation_metrics_objs = None
    
    if not st.session_state.evaluation_dataset_loaded:
        # Create tabs for dataset source selection
        dataset_source_tab1, dataset_source_tab2 = st.tabs(["Upload Dataset", "Load from Couchbase"])
        
        with dataset_source_tab1:
            uploaded_file = st.file_uploader("Upload evaluation dataset (JSON)", type=["json"], key="evaluation_dataset_uploader")
            
            if uploaded_file is not None:
                try:
                    # Save the uploaded file
                    with open("temp_dataset.json", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load the JSON data first
                    with open("temp_dataset.json", "r") as f:
                        json_data = json.load(f)
                    
                    # Load dataset
                    dataset = EvalDataset(**json_data)
                    
                    # Store in session state
                    st.session_state.evaluation_dataset_loaded = True
                    st.session_state.evaluation_dataset = dataset
                    
                    # Clean up
                    os.remove("temp_dataset.json")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
        
        with dataset_source_tab2:
            st.write("Load dataset from Couchbase: Try sample dataset id: 3aa57d61-c2b4-4a3a-8939-f7bdef078eb2")
            
            # Check if Couchbase credentials are available
            if (os.environ.get("cluster_url") and 
                os.environ.get("cb_username") and 
                os.environ.get("cb_password") and 
                os.environ.get("bucket") and 
                os.environ.get("scope") and 
                os.environ.get("collection")):
                
                # Provide dataset ID input
                dataset_id = st.text_input("Dataset ID", placeholder="Enter the dataset ID to load")
                
                if st.button("Load Dataset", key="load_dataset_from_cb"):
                    try:
                        with st.spinner("Loading dataset from Couchbase..."):
                            # Initialize dataset with load operator
                            start_time = time.time()
                            dataset = LoadOperator().retrieve_docs(dataset_id=dataset_id)
                            load_time = time.time() - start_time
                            
                            # Store in session state
                            st.session_state.evaluation_dataset_loaded = True
                            st.session_state.evaluation_dataset = dataset
                            
                            # Display loading time
                            st.info(f"Dataset loaded in {load_time:.2f} seconds")
                            
                    except Exception as e:
                        st.error(f"Error loading dataset from Couchbase: {str(e)}")
            else:
                st.warning("Couchbase credentials not configured. Please provide them in the sidebar.")
    
    # Display dataset info and evaluation options
    if st.session_state.evaluation_dataset_loaded:
        # Show preview
        st.subheader("Dataset Preview")
        
        dataset = st.session_state.evaluation_dataset
        
        # Create DataFrame for display
        data_preview = []
        for i in range(min(5, len(dataset.questions) if dataset.questions else 0)):
            preview = {"Question": dataset.questions[i] if dataset.questions else "N/A"}
            
            if dataset.answers:
                preview["Ground Truth"] = str(dataset.answers[i])
            
            if dataset.responses:
                preview["Response"] = dataset.responses[i]
            
            if dataset.reference_contexts:
                preview["Reference Context"] = dataset.reference_contexts[i][:100] + "..." if len(dataset.reference_contexts[i]) > 100 else dataset.reference_contexts[i]
            
            if dataset.retrieved_contexts:
                preview["Retrieved Context"] = str(dataset.retrieved_contexts[i])
            
            data_preview.append(preview)
        
        st.dataframe(pd.DataFrame(data_preview))
        
        # Show evaluation options
        if not st.session_state.evaluation_results_loaded:
            # Metrics selection
            st.subheader("Select Evaluation Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_faithfulness = st.checkbox("Faithfulness", value=False)
                use_answer_relevancy = st.checkbox("Answer Relevancy", value=False)
                use_context_recall = st.checkbox("Context Recall", value=False)
                use_context_precision = st.checkbox("Context Precision", value=False)
                use_llm_grading = st.checkbox("LLM Grading", value=False)
            
            with col2:
                use_answer_correctness = st.checkbox("Answer Correctness", value=False)
                use_avg_chunk_size = st.checkbox("Average Chunk Size", value=False)
                use_context_similarity = st.checkbox("Context Similarity", value=False)
                use_context_score = st.checkbox("Context Score", value=False)
            
            if st.button("Run Evaluation", key="run_evaluation"):
                # Prepare metrics list
                metrics = []
                if use_faithfulness: metrics.append(faithfulness)
                if use_answer_relevancy: metrics.append(answer_relevancy)
                if use_context_recall: metrics.append(context_recall)
                if use_context_precision: metrics.append(context_precision)
                if use_answer_correctness: metrics.append(answer_correctness)
                if use_avg_chunk_size: metrics.append(avg_chunk_size)
                if use_context_similarity: metrics.append(context_similarity)
                if use_context_score: metrics.append(context_score)
                if use_llm_grading: metrics.append(llm_grading)
                
                with st.spinner("Running evaluation..."):
                    # Run evaluation
                    start_time = time.time()
                    engine = ValidationEngine(dataset=dataset, metrics=metrics)
                    results, metrics_objs, schema, avg_metrics = engine.evaluate()
                    evaluation_time = time.time() - start_time
                
                # Store results in session state
                st.session_state.evaluation_results_loaded = True
                st.session_state.evaluation_results = results
                st.session_state.evaluation_avg_metrics = avg_metrics
                st.session_state.evaluation_metrics_objs = metrics_objs
                
                # Display evaluation time
                st.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Display results if available
        if st.session_state.evaluation_results_loaded:
            # Display results
            st.subheader("Evaluation Results")
            
            # Display average metrics
            st.write("Average Metrics:")
            metrics_df = pd.DataFrame([st.session_state.evaluation_avg_metrics])
            st.dataframe(metrics_df)
            
            # Display detailed results
            st.write("Detailed Results:")
            results_df = pd.DataFrame(st.session_state.evaluation_results)
            st.dataframe(results_df)
            
            # Offer download of results
            with open(".results/results.json", "r") as f:
                st.download_button(
                    label="Download Results (JSON)",
                    data=f,
                    file_name="evaluation_results.json",
                    mime="application/json"
                )
            
            with open(".results/results.csv", "r") as f:
                st.download_button(
                    label="Download Results (CSV)",
                    data=f,
                    file_name="evaluation_results.csv",
                    mime="text/csv"
                )

elif option == "Experiment":
    st.title("Experiment based evaluation")
    
    # Initialize session state variables for Experiment tab if they don't exist
    if 'experiment_dataset' not in st.session_state:
        st.session_state.experiment_dataset = None
    if 'experiment_results_loaded' not in st.session_state:
        st.session_state.experiment_results_loaded = False
    if 'experiment_id' not in st.session_state:
        st.session_state.experiment_id = f"exp_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    if 'experiment_config' not in st.session_state:
        st.session_state.experiment_config = None
    if 'experiment_results_df' not in st.session_state:
        st.session_state.experiment_results_df = None
    
    tab1, tab2 = st.tabs(["Create Experiment", "Retrieve Experiment"])
    
    with tab1:
        st.header("Create New Experiment")
        
        # Dataset source selection
        dataset_source = st.radio("Select dataset source:", ["Upload JSON", "Load from Couchbase"])
        
        if dataset_source == "Upload JSON":
            # Dataset upload
            uploaded_file = st.file_uploader("Upload evaluation dataset (JSON)", type=["json"], key="experiment_dataset_uploader")
            dataset_loaded = uploaded_file is not None
        else:  # Load from Couchbase
            # Check if environment variables are set
            env_vars_set = all([os.getenv("bucket"), os.getenv("scope"), 
                              os.getenv("cluster_url"), os.getenv("cb_username"), 
                              os.getenv("cb_password")])
            
            if not env_vars_set:
                st.warning("Couchbase environment variables not set. Please configure them in the sidebar.")
                dataset_loaded = False
            else:
                # Allow user to specify collection and dataset ID
                dataset_id = st.text_input("Dataset ID")
                
                if st.button("Load Dataset from Couchbase"):
                    try:
                        with st.spinner("Loading dataset from Couchbase..."):
                            # Initialize LoadOperator
                            load_op = LoadOperator()
                            
                            # Load dataset from Couchbase
                            start_time = time.time()
                            dataset = load_op.retrieve_docs(dataset_id=dataset_id)
                            load_time = time.time() - start_time
                            
                            if dataset:
                                st.session_state.experiment_dataset = dataset
                                st.success(f"Successfully loaded dataset with ID: {dataset_id}")
                                st.info(f"Dataset loaded in {load_time:.2f} seconds")
                                dataset_loaded = True
                            else:
                                st.error(f"No data found for dataset ID: {dataset_id}")
                                dataset_loaded = False
                    except Exception as e:
                        st.error(f"Error loading dataset from Couchbase: {str(e)}")
                        dataset_loaded = False
                else:
                    dataset_loaded = False
        
        # Experiment configuration
        st.subheader("Experiment Metadata")
        
        experiment_id = st.text_input("Experiment ID", value=st.session_state.experiment_id)
        st.session_state.experiment_id = experiment_id
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=1, value=None)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=None)
            embedding_model = st.text_input("Embedding Model", value="")
        
        with col2:
            embedding_dimension = st.number_input("Embedding Dimension", min_value=1, value=None)
            llm_model = st.text_input("LLM Model", value="")
        
        # Metrics selection
        st.subheader("Select Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_faithfulness = st.checkbox("Faithfulness", value=False, key="exp_faith")
            use_answer_relevancy = st.checkbox("Answer Relevancy", value=False, key="exp_ans_rel")
            use_context_recall = st.checkbox("Context Recall", value=False, key="exp_ctx_recall")
            use_context_precision = st.checkbox("Context Precision", value=False, key="exp_ctx_prec")
            use_llm_grading = st.checkbox("LLM Grading", value=False, key="exp_llm_grading")
        
        with col2:
            use_answer_correctness = st.checkbox("Answer Correctness", value=False, key="exp_ans_corr")
            use_avg_chunk_size = st.checkbox("Average Chunk Size", value=False, key="exp_chunk_size")
            use_context_similarity = st.checkbox("Context Similarity", value=False, key="exp_ctx_sim")
            use_context_score = st.checkbox("Context Score", value=False, key="exp_ctx_score")
        
        create_experiment_button = st.button("Create Experiment", key="create_experiment", disabled=not dataset_loaded)
        
        if create_experiment_button:
            try:
                # If using uploaded file, save it and load dataset
                if dataset_source == "Upload JSON":
                    with open("temp_dataset.json", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # et the json dict
                    with open("temp_dataset.json", "r") as f:
                        json_data = json.load(f)
                    
                    # Load dataset
                    dataset = EvalDataset(**json_data)
                    st.session_state.experiment_dataset = dataset
                else:
                    # Dataset already loaded from Couchbase
                    dataset = st.session_state.experiment_dataset
                
                # Prepare metrics list
                metrics = []
                if use_faithfulness: metrics.append(faithfulness)
                if use_answer_relevancy: metrics.append(answer_relevancy)
                if use_context_recall: metrics.append(context_recall)
                if use_context_precision: metrics.append(context_precision)
                if use_answer_correctness: metrics.append(answer_correctness)
                if use_avg_chunk_size: metrics.append(avg_chunk_size)
                if use_context_similarity: metrics.append(context_similarity)
                if use_context_score: metrics.append(context_score)
                if use_llm_grading: metrics.append(llm_grading)
                
                # Create experiment options
                options = ExperimentOptions(
                    experiment_id=experiment_id,
                    dataset_id=dataset.dataset_id,
                    metrics=metrics,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_model=embedding_model,
                    embedding_dimension=embedding_dimension,
                    llm_model=llm_model
                )
                
                st.session_state.experiment_config = options
                
                with st.spinner("Running experiment..."):
                    # Initialize experiment
                    start_time = time.time()
                    experiment = Experiment(dataset=dataset, options=options)
                    experiment_time = time.time() - start_time
                
                st.session_state.experiment_results_loaded = True
                
                # Display experiment time
                st.info(f"Experiment completed in {experiment_time:.2f} seconds")
                
                # Load and store results in session state
                results_file = f".results-{experiment_id}/results.json"
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        results = json.load(f)
                    st.session_state.experiment_results_df = pd.DataFrame(results)
                
                # Clean up if using uploaded file
                if dataset_source == "Upload JSON" and os.path.exists("temp_dataset.json"):
                    os.remove("temp_dataset.json")
            except Exception as e:
                st.error(f"Error creating experiment: {str(e)}")
        
        # Display results if available
        if st.session_state.experiment_results_loaded:
            # Display results
            st.subheader("Experiment Results")
            
            # Display metrics
            if st.session_state.experiment_results_df is not None:
                # Select only numeric columns for visualization
                df = st.session_state.experiment_results_df
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                
                # Display metrics
                st.subheader("Averaged Evaluation Metrics")
                metrics = df[numeric_columns].mean()
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df)
                
                # Display results table
                st.subheader("Detailed Results")
                st.dataframe(df)
            
            # Option to load to Couchbase
            st.subheader("Save to Couchbase")
            save_to_couchbase = st.checkbox("Save experiment to Couchbase", value=False)
            
            if save_to_couchbase:
                # Check if environment variables are set
                env_vars_set = all([os.getenv("bucket"), os.getenv("scope"), 
                                  os.getenv("cluster_url"), os.getenv("cb_username"), 
                                  os.getenv("cb_password")])
                
                if env_vars_set:
                    # Allow user to specify collection or use default
                    default_collection = os.getenv("collection", "")
                    collection_name = st.text_input("Collection name (leave empty to use default)", value=default_collection)
                    
                    if st.button("Save to Couchbase"):
                        with st.spinner("Saving experiment to Couchbase..."):
                            # Load to Couchbase
                            experiment_id = st.session_state.experiment_id
                            experiment = Experiment()
                            experiment.options = st.session_state.experiment_config
                            experiment.dataset = st.session_state.experiment_dataset
                            experiment.load_to_couchbase(collection=collection_name)
                            st.success(f"Experiment data loaded to Couchbase with ID: {experiment_id}")
                else:
                    st.warning("Couchbase environment variables not set. Please configure them to save to Couchbase.")
            
            # Offer download of results
            experiment_id = st.session_state.experiment_id
            download_path = f".results-{experiment_id}/experiment_config.json"
            if os.path.exists(download_path):
                with open(download_path, "r") as f:
                    st.download_button(
                        label="Download Experiment Config",
                        data=f,
                        file_name=f"experiment_{experiment_id}_config.json",
                        mime="application/json"
                    )
    
    with tab2:
        st.header("Retrieve Experiment")
        
        # Initialize session state for retrieve experiment tab
        if 'retrieve_experiment_loaded' not in st.session_state:
            st.session_state.retrieve_experiment_loaded = False
        if 'retrieve_experiment_id' not in st.session_state:
            st.session_state.retrieve_experiment_id = ""
        if 'retrieve_experiment_config' not in st.session_state:
            st.session_state.retrieve_experiment_config = None
        if 'retrieve_experiment_df' not in st.session_state:
            st.session_state.retrieve_experiment_df = None
        
        # Clear retrieve experiment state if needed
        if st.button("Clear Experiment", key="clear_retrieve"):
            st.session_state.retrieve_experiment_loaded = False
            st.session_state.retrieve_experiment_id = ""
            st.session_state.retrieve_experiment_config = None
            st.session_state.retrieve_experiment_df = None
        
        if not st.session_state.retrieve_experiment_loaded:
            # Input fields
            retrieve_experiment_id = st.text_input("Enter Experiment ID")
            retrieve_collection = st.text_input("Enter Collection Name")
            
            if st.button("Retrieve Experiment", key="retrieve_experiment") and retrieve_experiment_id:
                try:
                    with st.spinner("Retrieving experiment..."):
                        # Create experiment instance for retrieval
                        experiment = Experiment()
                        experiment.retrieve(experiment_id=retrieve_experiment_id, collection=retrieve_collection)
                    
                    # Store in session state
                    st.session_state.retrieve_experiment_loaded = True
                    st.session_state.retrieve_experiment_id = retrieve_experiment_id
                    
                    # Check if results directory exists
                    experiment_dir = f".results-{retrieve_experiment_id}"
                    if os.path.exists(experiment_dir):
                        # Read experiment config
                        with open(f"{experiment_dir}/experiment_config.json", "r") as f:
                            config = json.load(f)
                        st.session_state.retrieve_experiment_config = config
                        
                        # Load and store results
                        results_file = os.path.join(experiment_dir, "results.json")
                        if os.path.exists(results_file):
                            with open(results_file, "r") as f:
                                results = json.load(f)
                            st.session_state.retrieve_experiment_df = pd.DataFrame(results)
                    else:
                        st.warning(f"No local results found for experiment ID: {retrieve_experiment_id}")
                except Exception as e:
                    st.error(f"Error retrieving experiment: {str(e)}")
        
        # Display retrieved experiment if loaded
        if st.session_state.retrieve_experiment_loaded:
            experiment_id = st.session_state.retrieve_experiment_id
            config = st.session_state.retrieve_experiment_config
            df = st.session_state.retrieve_experiment_df
            
            st.success(f"Retrieved experiment: {experiment_id}")
            
            # Display experiment metadata
            st.subheader("Experiment Metadata")
            if config:
                st.json(config)
            
            # Display results if available
            if df is not None:
                # Select only numeric columns for visualization
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                
                # Display metrics
                st.subheader("Evaluation Metrics")
                metrics = df[numeric_columns].mean()
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df)
                
                # Display results table
                st.subheader("Detailed Results")
                st.dataframe(df)
                
                # Offer download of results
                experiment_dir = f".results-{experiment_id}"
                results_file = os.path.join(experiment_dir, "results.json")
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        st.download_button(
                            label="Download Results (JSON)",
                            data=f,
                            file_name=f"experiment_{experiment_id}_results.json",
                            mime="application/json"
                        )
                
                if os.path.exists(f"{experiment_dir}/results.csv"):
                    with open(f"{experiment_dir}/results.csv", "r") as f:
                        st.download_button(
                            label="Download Results (CSV)",
                            data=f,
                            file_name=f"experiment_{experiment_id}_results.csv",
                            mime="text/csv"
                        )
            else:
                st.warning(f"No results found for experiment ID: {experiment_id}")

elif option == "Result Analysis":
    st.title("Analyze Evaluation Results")
    
    # Initialize session state variables if they don't exist
    if 'result_analysis_data_loaded' not in st.session_state:
        st.session_state.result_analysis_data_loaded = False
    if 'result_analysis_experiment_id' not in st.session_state:
        st.session_state.result_analysis_experiment_id = ""
    if 'result_analysis_config' not in st.session_state:
        st.session_state.result_analysis_config = None
    if 'result_analysis_df' not in st.session_state:
        st.session_state.result_analysis_df = None
    
    # Function to clear session state
    def clear_result_analysis_state():
        st.session_state.result_analysis_data_loaded = False
        st.session_state.result_analysis_experiment_id = ""
        st.session_state.result_analysis_config = None
        st.session_state.result_analysis_df = None
    
    # Function to load data
    def load_result_data(experiment_id, collection):
        try:
            # Check if results directory exists locally
            experiment_dir = f".results-{experiment_id}"
            if not os.path.exists(experiment_dir):
                if collection:
                    with st.spinner(f"Retrieving experiment {experiment_id} from Couchbase..."):
                        experiment = Experiment()
                        experiment.retrieve(experiment_id=experiment_id, collection=collection)
                else:
                    st.error(f"Experiment {experiment_id} not found locally. Please provide a collection name to retrieve from Couchbase.")
                    return False
            
            # Check if experiment exists after potential retrieval
            if os.path.exists(experiment_dir):
                # Load experiment config
                with open(f"{experiment_dir}/experiment_config.json", "r") as f:
                    config = json.load(f)
                
                # Load results
                results_file = os.path.join(experiment_dir, "results.json")
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        results = json.load(f)
                    
                    # Store in session state
                    st.session_state.result_analysis_data_loaded = True
                    st.session_state.result_analysis_experiment_id = experiment_id
                    st.session_state.result_analysis_config = config
                    st.session_state.result_analysis_df = pd.DataFrame(results)
                    return True
                else:
                    st.error(f"Results file not found in the experiment directory: {experiment_dir}")
                    return False
            else:
                st.error(f"Experiment {experiment_id} could not be retrieved.")
                return False
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            return False
    
    # Display input form if data is not loaded
    if not st.session_state.result_analysis_data_loaded:
        # Input fields for experiment ID and collection
        col1, col2 = st.columns(2)
        with col1:
            experiment_id = st.text_input("Enter Experiment ID")
        with col2:
            collection = st.text_input("Collection Name (if retrieving from Couchbase)")
        
        # Add button with a unique key
        if st.button("Load Results", key="load_results_button"):
            if not experiment_id:
                st.error("Please enter an Experiment ID to visualize results.")
            else:
                load_result_data(experiment_id, collection)
        
        st.info("Enter an Experiment ID and click 'Load Results' to visualize evaluation results.")
    else:
        # Data is loaded, show analysis UI
        st.success(f"Loaded experiment: {st.session_state.result_analysis_experiment_id}")
        
        # Add a button to load different experiment
        if st.button("Load Different Experiment", key="change_experiment"):
            clear_result_analysis_state()
            st.rerun()
        
        # Access stored data
        config = st.session_state.result_analysis_config
        df = st.session_state.result_analysis_df
        
        # Display experiment metadata
        st.subheader("Experiment Metadata")
        st.json(config)
        
        # Create visualization tabs
        viz_tabs = st.tabs(["Summary Dashboard", "Detailed Metrics", "Full Results"])
        
        with viz_tabs[0]:
            st.subheader("Evaluation Summary")
            
            # Extract numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numeric_cols) > 0:
                # Metrics selector
                selected_metrics = st.multiselect(
                    "Select metrics to visualize",
                    options=list(numeric_cols),
                    default=list(numeric_cols)[:min(5, len(numeric_cols))],
                    key="summary_metrics_selector"
                )
                
                if selected_metrics:
                    # Calculate summary statistics
                    summary_stats = df[selected_metrics].describe().T.reset_index()
                    summary_stats = summary_stats.rename(columns={'index': 'Metric'})
                    
                    # 1. Overview metrics card
                    metrics_container = st.container()
                    cols = st.columns(len(selected_metrics))
                    
                    for i, metric in enumerate(selected_metrics):
                        with cols[i]:
                            mean_val = df[metric].mean()
                            median_val = df[metric].median()
                            st.metric(
                                label=metric,
                                value=f"{mean_val:.3f}",
                                delta=f"Median: {median_val:.3f}"
                            )
                    
                    # 2. Summary statistics table
                    st.subheader("Summary Statistics")
                    formatted_summary = summary_stats.style.format({
                        'mean': '{:.3f}',
                        'std': '{:.3f}',
                        'min': '{:.3f}',
                        '25%': '{:.3f}',
                        '50%': '{:.3f}',
                        '75%': '{:.3f}',
                        'max': '{:.3f}'
                    })
                    st.dataframe(formatted_summary)
                    
                    # 5. Metric Accuracy Analysis (percentage of results meeting quality thresholds)
                    st.subheader("Metric Quality Analysis")
                    
                    # Define quality thresholds for each metric based on domain knowledge
                    metric_thresholds = {
                        "context_precision": 0.9,
                        "context_recall": 0.9,
                        "answer_relevancy": 0.7,
                        "faithfulness": 0.8,
                        "answer_correctness": 0.7,
                        "avg_chunk_size": 0.5,
                        "context_similarity": 0.8,
                        "context_score": 0.8,
                        "named_entity_score": 0.6,
                        # Add custom thresholds for other metrics if needed
                    }
                    
                    # For any metrics not in the dictionary, use 0.7 as default threshold
                    default_threshold = 0.7
                    
                    accuracy_data = []
                    for metric in selected_metrics:
                        threshold = metric_thresholds.get(metric, default_threshold)
                        above_threshold = (df[metric] >= threshold).sum()
                        total_samples = len(df[metric])
                        accuracy = (above_threshold / total_samples) * 100 if total_samples > 0 else 0
                        
                        accuracy_data.append({
                            "Metric": metric,
                            "Threshold": threshold,
                            "Samples Above Threshold": above_threshold,
                            "Total Samples": total_samples,
                            "Accuracy (%)": accuracy
                        })
                    
                    accuracy_df = pd.DataFrame(accuracy_data)
                    
                    # Create a color scale function for the accuracy column
                    def highlight_accuracy(val):
                        if val >= 90:
                            return 'background-color: rgba(76, 175, 80, 0.7)'  # Strong green
                        elif val >= 70:
                            return 'background-color: rgba(76, 175, 80, 0.4)'  # Medium green
                        elif val >= 50:
                            return 'background-color: rgba(255, 235, 59, 0.5)'  # Yellow
                        else:
                            return 'background-color: rgba(244, 67, 54, 0.4)'  # Red
                    
                    # Format and display the dataframe
                    styled_accuracy_df = accuracy_df.style\
                        .format({'Accuracy (%)': '{:.2f}', 'Threshold': '{:.2f}'})\
                        .applymap(highlight_accuracy, subset=['Accuracy (%)'])
                    
                    st.dataframe(styled_accuracy_df)
                    
                    # Add explanation for metric thresholds
                    with st.expander("About Metric Thresholds"):
                        st.markdown("""
                        ### Metric Quality Thresholds
                        
                        The accuracy percentage shows what portion of your dataset meets or exceeds the quality threshold for each metric:
                        
                        - **context_precision** (0.7): Measures how precise the retrieved contexts are
                        - **context_recall** (0.7): Measures how complete the retrieved contexts are
                        - **answer_relevancy** (0.7): Measures how relevant the response is to the question
                        - **faithfulness** (0.8): Measures how well the response stays true to the retrieved context
                        - **answer_correctness** (0.7): Measures how correct the answer is compared to ground truth
                        - **avg_chunk_size** (0.5): Evaluates optimal chunk sizing (>0.5 is considered acceptable)
                        - **context_similarity** (0.7): Measures embedding similarity between reference and retrieved contexts
                        - **context_score** (0.7): Evaluates overall context quality
                        - **named_entity_score** (0.6): Measures named entity overlap between query and context
                        
                        Higher accuracy percentages indicate better overall performance.
                        """)
                    
                    # 6. Visualization of accuracy
                    fig = px.bar(
                        accuracy_df,
                        x="Metric",
                        y="Accuracy (%)",
                        color="Accuracy (%)",
                        color_continuous_scale=["red", "yellow", "green"],
                        range_color=[0, 100],
                        title="Metric Quality Analysis",
                        labels={"Accuracy (%)": "% of Samples Meeting Threshold"}
                    )
                    
                    # Add threshold markers
                    for i, row in accuracy_df.iterrows():
                        fig.add_shape(
                            type="line",
                            x0=i-0.4, x1=i+0.4,
                            y0=70, y1=70,
                            line=dict(color="black", width=2, dash="dash"),
                            name="Good Performance (70%)"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one metric to visualize")
            else:
                st.warning("No numeric metrics found in the results")
        
        with viz_tabs[1]:
            st.subheader("Detailed Metric Analysis")
            
            # Select metric to analyze
            if len(numeric_cols) > 0:
                selected_metric = st.selectbox(
                    "Select metric to analyze",
                    options=numeric_cols,
                    key="detailed_metric_selector"
                )
                
                if selected_metric:
                    # Create detailed visualizations for this metric
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        # Check if selected metric should be bounded
                        bounded_metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy', 'answer_correctness', 'context_similarity', 'context_score']
                        is_bounded = selected_metric in bounded_metrics or any(selected_metric.endswith(suffix) for suffix in ['precision', 'recall'])

                        # Filter data if needed
                        df_filtered = df[selected_metric]
                        if is_bounded:
                            df_filtered = df[selected_metric].clip(0, 1)

                        hist_fig = px.histogram(
                            x=df_filtered,
                            title=f"Distribution of {selected_metric}",
                            marginal="box"  # Add box plot at the margin
                        )

                        # Set x-axis range for bounded metrics
                        if is_bounded:
                            hist_fig.update_layout(xaxis=dict(range=[0, 1]))

                        st.plotly_chart(hist_fig, use_container_width=True)
                    
                    with col2:
                        # CDF (Cumulative Distribution Function)
                        sorted_data = np.sort(df_filtered)
                        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                        
                        cdf_fig = go.Figure()
                        cdf_fig.add_trace(go.Scatter(
                            x=sorted_data,
                            y=cumulative,
                            mode='lines',
                            name='CDF'
                        ))
                        
                        cdf_fig.update_layout(
                            title=f"Cumulative Distribution of {selected_metric}",
                            xaxis_title=selected_metric,
                            yaxis_title="Probability",
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(cdf_fig, use_container_width=True)
                    
                    # Statistics card
                    st.subheader(f"Statistics for {selected_metric}")
                    
                    stats_cols = st.columns(5)
                    with stats_cols[0]:
                        st.metric("Mean", f"{df_filtered.mean():.3f}")
                    with stats_cols[1]:
                        st.metric("Median", f"{df_filtered.median():.3f}")
                    with stats_cols[2]:
                        st.metric("Std Dev", f"{df_filtered.std():.3f}")
                    with stats_cols[3]:
                        st.metric("Min", f"{df_filtered.min():.3f}")
                    with stats_cols[4]:
                        st.metric("Max", f"{df_filtered.max():.3f}")
                    
                else:
                    st.warning("Please select a metric to analyze")
            else:
                st.warning("No numeric metrics found in the results")
        
        with viz_tabs[2]:
            st.subheader("Full Results")
            
            # Filter and search functionality
            search_term = st.text_input("Search in results", key="search_results")
            
            if search_term:
                # Filter the dataframe based on the search term
                filtered_df = df[df.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
                st.dataframe(filtered_df)
                st.info(f"Found {len(filtered_df)} results matching '{search_term}'")
            else:
                st.dataframe(df)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"results_{st.session_state.result_analysis_experiment_id}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            
            with col2:
                json_str = df.to_json(orient="records")
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name=f"results_{st.session_state.result_analysis_experiment_id}.json",
                    mime="application/json",
                    key="download_json"
                )

elif option == "Compare Experiments":
    st.title("Compare Experiments")
    
    # Initialize session state variables for Compare Experiments tab
    if 'compare_experiments_loaded' not in st.session_state:
        st.session_state.compare_experiments_loaded = False
    if 'compare_experiment_id_1' not in st.session_state:
        st.session_state.compare_experiment_id_1 = ""
    if 'compare_experiment_id_2' not in st.session_state:
        st.session_state.compare_experiment_id_2 = ""
    if 'compare_config_1' not in st.session_state:
        st.session_state.compare_config_1 = None
    if 'compare_config_2' not in st.session_state:
        st.session_state.compare_config_2 = None
    if 'compare_df1' not in st.session_state:
        st.session_state.compare_df1 = None
    if 'compare_df2' not in st.session_state:
        st.session_state.compare_df2 = None
    
    # Clear compare experiments state if needed
    if st.button("Clear Comparison", key="clear_comparison"):
        st.session_state.compare_experiments_loaded = False
        st.session_state.compare_experiment_id_1 = ""
        st.session_state.compare_experiment_id_2 = ""
        st.session_state.compare_config_1 = None
        st.session_state.compare_config_2 = None
        st.session_state.compare_df1 = None
        st.session_state.compare_df2 = None
    
    # Input fields if experiments are not loaded
    if not st.session_state.compare_experiments_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_id_1 = st.text_input("First Experiment ID")
            collection_1 = st.text_input("First Collection Name (if not found locally)")
        
        with col2:
            experiment_id_2 = st.text_input("Second Experiment ID")
            collection_2 = st.text_input("Second Collection Name (if not found locally)")
        
        # Add Start Comparison button
        if st.button("Start Comparison", key="start_comparison"):
            if not experiment_id_1 or not experiment_id_2:
                st.error("Please enter both experiment IDs to begin comparison.")
            else:
                try:
                    # Load both experiments
                    experiment_dir_1 = f".results-{experiment_id_1}"
                    experiment_dir_2 = f".results-{experiment_id_2}"
                    
                    # Check if experiments exist locally, if not try to retrieve from Couchbase
                    if not os.path.exists(experiment_dir_1):
                        if collection_1:
                            with st.spinner(f"Retrieving experiment {experiment_id_1} from Couchbase..."):
                                experiment = Experiment()
                                experiment.retrieve(experiment_id=experiment_id_1, collection=collection_1)
                        else:
                            st.error(f"Experiment {experiment_id_1} not found locally. Please provide a collection name to retrieve from Couchbase.")
                            st.stop()
                    
                    if not os.path.exists(experiment_dir_2):
                        if collection_2:
                            with st.spinner(f"Retrieving experiment {experiment_id_2} from Couchbase..."):
                                experiment = Experiment()
                                experiment.retrieve(experiment_id=experiment_id_2, collection=collection_2)
                        else:
                            st.error(f"Experiment {experiment_id_2} not found locally. Please provide a collection name to retrieve from Couchbase.")
                            st.stop()
                    
                    # Now check if both experiments exist after potential retrieval
                    if os.path.exists(experiment_dir_1) and os.path.exists(experiment_dir_2):
                        # Load experiment configs
                        with open(f"{experiment_dir_1}/experiment_config.json", "r") as f:
                            config_1 = json.load(f)
                        with open(f"{experiment_dir_2}/experiment_config.json", "r") as f:
                            config_2 = json.load(f)
                        
                        # Load results
                        with open(f"{experiment_dir_1}/results.json", "r") as f:
                            results_1 = json.load(f)
                        with open(f"{experiment_dir_2}/results.json", "r") as f:
                            results_2 = json.load(f)
                        
                        # Convert to DataFrames
                        df1 = pd.DataFrame(results_1)
                        df2 = pd.DataFrame(results_2)
                        
                        # Store in session state
                        st.session_state.compare_experiments_loaded = True
                        st.session_state.compare_experiment_id_1 = experiment_id_1
                        st.session_state.compare_experiment_id_2 = experiment_id_2
                        st.session_state.compare_config_1 = config_1
                        st.session_state.compare_config_2 = config_2
                        st.session_state.compare_df1 = df1
                        st.session_state.compare_df2 = df2
                    else:
                        st.error("One or both experiments could not be retrieved. Please check the experiment IDs and collection names.")
                        
                except Exception as e:
                    st.error(f"Error comparing experiments: {str(e)}")
        else:
            st.info("Enter experiment IDs and collection names (if needed), then click 'Start Comparison' to begin.")
    
    # Display comparison if experiments are loaded
    if st.session_state.compare_experiments_loaded:
        experiment_id_1 = st.session_state.compare_experiment_id_1
        experiment_id_2 = st.session_state.compare_experiment_id_2
        config_1 = st.session_state.compare_config_1
        config_2 = st.session_state.compare_config_2
        df1 = st.session_state.compare_df1
        df2 = st.session_state.compare_df2
        
        # Display experiment configurations side by side
        st.subheader("Experiment Configurations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Experiment {experiment_id_1}**")
            st.json(config_1)
        
        with col2:
            st.write(f"**Experiment {experiment_id_2}**")
            st.json(config_2)
        
        # Compare metrics
        st.subheader("Metrics Comparison")
        
        # Select numeric columns for comparison
        numeric_columns = df1.select_dtypes(include=['float64', 'int64']).columns
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Metric': numeric_columns,
            f'Experiment {experiment_id_1}': df1[numeric_columns].mean(),
            f'Experiment {experiment_id_2}': df2[numeric_columns].mean(),
            'Difference': df1[numeric_columns].mean() - df2[numeric_columns].mean()
        })
        
        # Display comparison table
        st.dataframe(comparison_df)
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["Summary Dashboard", "Detailed Distributions", "Statistical Analysis"])
        
        with viz_tabs[0]:
            st.subheader("Key Metrics Overview")
            
            # Create a multi-select for metrics to visualize
            selected_metrics = st.multiselect(
                "Select metrics to visualize",
                options=list(numeric_columns),
                default=list(numeric_columns)[:min(5, len(numeric_columns))],  # Default to first 5 metrics or less
                key="compare_metrics_multiselect"
            )
            
            if selected_metrics:
                # 1. Create a bar chart for direct comparison with percent difference
                bar_comparison = pd.DataFrame({
                    'Metric': selected_metrics,
                    f'Exp {experiment_id_1}': [df1[metric].mean() for metric in selected_metrics],
                    f'Exp {experiment_id_2}': [df2[metric].mean() for metric in selected_metrics],
                    'Percent Diff': [((df2[metric].mean() / df1[metric].mean()) - 1) * 100 if df1[metric].mean() != 0 else 0 
                                    for metric in selected_metrics]
                })
                
                # Bar chart for direct comparison
                fig_bar = px.bar(
                    bar_comparison,
                    x='Metric',
                    y=[f'Exp {experiment_id_1}', f'Exp {experiment_id_2}'],
                    barmode='group',
                    title="Metric Comparison",
                    labels={'value': 'Score', 'variable': 'Experiment'},
                    color_discrete_sequence=['#636EFA', '#EF553B']
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 2. Create a radar chart for multi-metric comparison
                # Normalize metrics for radar chart to 0-1 scale
                radar_data = {}
                for metric in selected_metrics:
                    min_val = min(df1[metric].min(), df2[metric].min())
                    max_val = max(df1[metric].max(), df2[metric].max())
                    range_val = max_val - min_val
                    
                    if range_val == 0:  # Handle case where min = max
                        radar_data[metric] = [0.5, 0.5]
                    else:
                        radar_data[metric] = [
                            (df1[metric].mean() - min_val) / range_val,
                            (df2[metric].mean() - min_val) / range_val
                        ]
                
                # Create radar chart data
                radar_fig = go.Figure()
                
                radar_fig.add_trace(go.Scatterpolar(
                    r=[radar_data[metric][0] for metric in selected_metrics],
                    theta=selected_metrics,
                    fill='toself',
                    name=f'Experiment {experiment_id_1}'
                ))
                radar_fig.add_trace(go.Scatterpolar(
                    r=[radar_data[metric][1] for metric in selected_metrics],
                    theta=selected_metrics,
                    fill='toself',
                    name=f'Experiment {experiment_id_2}'
                ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Metrics Radar Chart (Normalized)"
                )
                
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # 3. Improvement/Degradation Analysis
                st.subheader("Improvement Analysis")
                
                # Calculate significant improvements
                from scipy import stats
                
                analysis_data = []
                for metric in selected_metrics:
                    t_stat, p_value = stats.ttest_ind(df1[metric], df2[metric])
                    mean1 = df1[metric].mean()
                    mean2 = df2[metric].mean()
                    diff = mean2 - mean1
                    pct_diff = (mean2 / mean1 - 1) * 100 if mean1 != 0 else 0
                    
                    significant = p_value < 0.05
                    better = (diff > 0)
                    
                    analysis_data.append({
                        'Metric': metric,
                        'Difference': diff,
                        'Percent Change': f"{pct_diff:.2f}%",
                        'Significant': "Yes" if significant else "No",
                        'Better': "Improved" if better else "Decreased",
                        'p-value': p_value
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                
                # Color code the dataframe
                def color_significant(val):
                    if val == "Yes":
                        return 'background-color: rgba(76, 175, 80, 0.3)'
                    return ''
                
                def color_better(val):
                    if val == "Improved":
                        return 'background-color: rgba(76, 175, 80, 0.3)'
                    else:
                        return 'background-color: rgba(244, 67, 54, 0.3)'
                
                styled_df = analysis_df.style\
                    .applymap(color_significant, subset=['Significant'])\
                    .applymap(color_better, subset=['Better'])
                
                st.dataframe(styled_df)
            else:
                st.warning("Please select at least one metric to visualize")
        
        with viz_tabs[1]:
            st.subheader("Metric Distributions")
            
            # Create a selector for which metric to display
            selected_metric = st.selectbox(
                "Select metric to view distribution",
                options=numeric_columns,
                key="compare_metric_selector"
            )
            
            if selected_metric:
                # Create a subplot with 2 rows and 1 column
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    subplot_titles=[
                        f"Histogram: {selected_metric}",
                        f"Box Plot: {selected_metric}"
                    ],
                    vertical_spacing=0.2
                )
                
                # Add histograms to the first row
                bounded_metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy', 'answer_correctness', 'context_similarity', 'context_score']
                is_bounded = selected_metric in bounded_metrics or any(selected_metric.endswith(suffix) for suffix in ['precision', 'recall'])

                # Filter data if needed
                df1_filtered = df1[selected_metric]
                df2_filtered = df2[selected_metric]
                if is_bounded:
                    df1_filtered = df1[selected_metric].clip(0, 1)
                    df2_filtered = df2[selected_metric].clip(0, 1)

                fig.add_trace(
                    go.Histogram(
                        x=df1_filtered,
                        name=f'Exp {experiment_id_1}',
                        opacity=0.75,
                        marker_color='#636EFA'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Histogram(
                        x=df2_filtered,
                        name=f'Exp {experiment_id_2}',
                        opacity=0.75,
                        marker_color='#EF553B'
                    ),
                    row=1, col=1
                )

                # Add box plots to the second row
                fig.add_trace(
                    go.Box(
                        y=df1_filtered,
                        name=f'Exp {experiment_id_1}',
                        marker_color='#636EFA'
                    ),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Box(
                        y=df2_filtered,
                        name=f'Exp {experiment_id_2}',
                        marker_color='#EF553B'
                    ),
                    row=2, col=1
                )

                # Update layout with appropriate axis ranges if needed
                if is_bounded:
                    fig.update_layout(
                        height=600,
                        barmode='overlay',
                        title_text=f"Distribution Comparison for {selected_metric}",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis=dict(range=[0, 1]),
                        yaxis2=dict(range=[0, 1])
                    )
                else:
                    fig.update_layout(
                        height=600,
                        barmode='overlay',
                        title_text=f"Distribution Comparison for {selected_metric}",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistical comparison
                st.subheader(f"Statistical Summary for {selected_metric}")
                
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                    f'Exp {experiment_id_1}': [
                        df1[selected_metric].mean(),
                        df1[selected_metric].median(),
                        df1[selected_metric].std(),
                        df1[selected_metric].min(),
                        df1[selected_metric].max(),
                        df1[selected_metric].quantile(0.25),
                        df1[selected_metric].quantile(0.75)
                    ],
                    f'Exp {experiment_id_2}': [
                        df2[selected_metric].mean(),
                        df2[selected_metric].median(),
                        df2[selected_metric].std(),
                        df2[selected_metric].min(),
                        df2[selected_metric].max(),
                        df2[selected_metric].quantile(0.25),
                        df2[selected_metric].quantile(0.75)
                    ],
                    'Difference': [
                        df2[selected_metric].mean() - df1[selected_metric].mean(),
                        df2[selected_metric].median() - df1[selected_metric].median(),
                        df2[selected_metric].std() - df1[selected_metric].std(),
                        df2[selected_metric].min() - df1[selected_metric].min(),
                        df2[selected_metric].max() - df1[selected_metric].max(),
                        df2[selected_metric].quantile(0.25) - df1[selected_metric].quantile(0.25),
                        df2[selected_metric].quantile(0.75) - df1[selected_metric].quantile(0.75)
                    ]
                })
                
                st.dataframe(stats_df)
            else:
                st.warning("No metric selected")
        
        with viz_tabs[2]:
            st.subheader("Statistical Tests")
            
            # Create tests for each metric
            test_results = []
            
            for metric in numeric_columns:
                # Perform statistical tests
                from scipy import stats
                
                # T-test
                t_stat, p_value = stats.ttest_ind(df1[metric], df2[metric])
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p_value = stats.mannwhitneyu(df1[metric], df2[metric], alternative='two-sided')
                
                # Effect size - Cohen's d
                mean1, mean2 = df1[metric].mean(), df2[metric].mean()
                std1, std2 = df1[metric].std(), df2[metric].std()
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                effect_size = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                
                # Interpretation
                if effect_size < 0.2:
                    effect_interpretation = "Negligible"
                elif effect_size < 0.5:
                    effect_interpretation = "Small"
                elif effect_size < 0.8:
                    effect_interpretation = "Medium"
                else:
                    effect_interpretation = "Large"
                    
                # Add to results
                test_results.append({
                    'Metric': metric,
                    'T-statistic': t_stat,
                    'T-test p-value': p_value,
                    'U-statistic': u_stat,
                    'Mann-Whitney p-value': u_p_value,
                    "Cohen's d": effect_size,
                    'Effect Size': effect_interpretation,
                    'Significant Difference': 'Yes' if p_value < 0.05 or u_p_value < 0.05 else 'No'
                })
            
            # Convert to DataFrame
            test_df = pd.DataFrame(test_results)
            
            # Style DataFrame to highlight significant differences
            def highlight_significant(val):
                if val == 'Yes':
                    return 'background-color: rgba(76, 175, 80, 0.3)'
                return ''
            
            def highlight_pvalue(val):
                if val < 0.05:
                    return 'background-color: rgba(76, 175, 80, 0.3)'
                return ''
            
            styled_test_df = test_df.style\
                .applymap(highlight_significant, subset=['Significant Difference'])\
                .applymap(highlight_pvalue, subset=['T-test p-value', 'Mann-Whitney p-value'])\
                .format({
                    'T-statistic': '{:.4f}',
                    'T-test p-value': '{:.4f}',
                    'U-statistic': '{:.4f}',
                    'Mann-Whitney p-value': '{:.4f}',
                    "Cohen's d": '{:.4f}'
                })
            
            st.dataframe(styled_test_df)
            
            # Add explanation of statistical tests
            with st.expander("Statistical Tests Explanation"):
                st.markdown("""
                ### T-test
                Tests if the means of two independent samples are significantly different. A p-value < 0.05 indicates a significant difference.
                
                ### Mann-Whitney U Test
                Non-parametric test that doesn't assume normal distribution. Compares the medians of two independent samples.
                
                ### Cohen's d
                Measures effect size - the standardized difference between two means:
                - Less than 0.2: Negligible effect
                - 0.2 - 0.5: Small effect
                - 0.5 - 0.8: Medium effect
                - More than 0.8: Large effect
                """)
                
elif option == "Docs":
    st.title("RAG Evaluation Framework")
    st.markdown("A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using RAGAS and synthesizing ground truth data from raw documents.")

    # Overview section
    st.header("Overview")
    st.markdown("""
    This framework provides tools and metrics to evaluate RAG systems across three key components:
    - Chunking evaluation
    - Retrieval evaluation
    - Generation evaluation

    The framework integrates with RAGAS, a popular RAG evaluation library, and provides a structured approach for experiment management, storage and result persistence.
    """)
    
    # Installation section
    st.header("Installation")
    st.code("""
    # Clone the repository
    # Install the package:
    cd eval
    pip install .
    """)
    
    # Configuration section
    st.header("Configuration")
    st.markdown("""
    The framework uses environment variables for Couchbase and OpenAI configurations. 
    Configure these in the sidebar or in your .env file.
    """)
    
    # Quick Start Guide
    st.header("Quick Start Guide")
    
    quick_start_tabs = st.tabs(["Synthetic Data Generation", "Basic Evaluation", "Experiment-based Evaluation", "Agent Evaluation"])
    
    with quick_start_tabs[0]:
        st.markdown("""
        ### Synthetic Data Generation
        
        The framework provides tools to generate synthetic question-answer pairs from your documents, 
        which can be used as ground truth for evaluation.
        
        For JSON and CSV documents, provide detailed metadata including the dataset schema for accurate data generation.
        """)
        
        st.code("""
        # From Python
        from eval.src.data.generator import SyntheticDataGenerator

        # Initialize the generator
        generator = SyntheticDataGenerator()

        # Generate synthetic data from a CSV file
        metadata = "Document contains product descriptions with fields: name, description, price, and category."
        generated_data = generator.synthesize_from_csv(
            path="data/products.csv",
            metadata=metadata
        )
        """)
        
        st.markdown("**Or use the 'Generate Data' option in the sidebar to use the UI.**")
    
    with quick_start_tabs[1]:
        st.markdown("### Basic Evaluation")
        st.code("""
        from eval.src.data.dataset import EvalDataset
        from eval.src.evaluator.validation import ValidationEngine
        from eval.src.evaluator.metrics import context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness

        # Create dataset
        dataset = EvalDataset(
            questions=["What is RAG?"],
            answers=[["RAG is a retrieval-augmented generation system"]],
            responses=["RAG combines retrieval with generation"],
            reference_contexts=["RAG systems use retrieval to enhance generation"],
            retrieved_contexts=[["RAG: retrieval-augmented generation"]]
        )

        # Run evaluation with metrics
        engine = ValidationEngine(
            dataset=dataset,
            metrics=[context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness]
        )
        results = engine.evaluate()
        """)
        
        st.markdown("**Or use the 'Evaluate' option in the sidebar for a UI-based approach.**")
    
    with quick_start_tabs[2]:
        st.markdown("### Experiment-based Evaluation")
        st.code("""
        from eval.src.controller.options import ExperimentOptions
        from eval.src.controller.manager import Experiment

        # Configure experiment
        experiment_options = ExperimentOptions(
            experiment_id="exp_001",
            dataset_id="dataset_001",
            metrics=[context_precision, context_recall, faithfulness],
            chunk_size=100,
            chunk_overlap=20,
            embedding_model="text-embedding-3-large",
            embedding_dimension=3072,
            llm_model="gpt-4"
        )

        # Create experiment
        experiment = Experiment(dataset=dataset, options=experiment_options)

        # Load to Couchbase
        experiment.load_to_couchbase()
        """)
        
        st.markdown("**Or use the 'Experiment' option in the sidebar for the UI version.**")
    
    with quick_start_tabs[3]:
        st.markdown("### Agent Evaluation")
        st.markdown("""
        The framework provides tools to trace and evaluate LangGraph and LangChain agents.
        """)
        
        # Remove the columns and display sequentially
        st.code(
        """
        # LangGraph tracing
        from eval.src.langgraph.trace_v2 import create_traced_agent, log_traces
        from langchain_core.messages import HumanMessage

        # Create your LangGraph agent
        # ... your agent implementation ...
        react_graph = builder.compile()

        # Create a traced version of the agent
        traced_graph = create_traced_agent(react_graph)

        def get_agent_response_wrapped(queries):
            # Use the TracedAgent wrapper to automatically trace all interactions
            results = []
            for query in queries:
                messages = [HumanMessage(content=query)]
                result = traced_graph.stream({"messages": messages})
                results.append(list(result))  
            
            log_path = log_traces()
            print(f"Traces saved to: {log_path}")
            
            return results

        # Example usage
        get_agent_response_wrapped(["What is the price of copper?", "What is the price of gold?"])

        from src.data.dataset import EvalDataset
        from src.controller.options import ExperimentOptions
        from src.controller.manager import Experiment

        reference_tool_calls = [
            [{"name": "get_metal_price", "args": {"metal_name": "copper"}}],
            [{"name": "get_metal_price", "args": {"metal_name": "gold"}}]
        ]

        gt_answers = [
            "The current price of copper is $0.0098 per gram.",
            "The current price of gold is $88.16 per gram."
        ]

        gt_tool_outputs = [
            ["0.0098"],
            ["88.1553"]
        ]

        gt_dataset = EvalDataset(
            reference_tool_calls=reference_tool_calls,
            gt_answers=gt_answers,
            gt_tool_outputs=gt_tool_outputs
        )

        experiment_options = ExperimentOptions(
            experiment_id="improved_test",
            langgraph=True
        )

        experiment = Experiment(dataset=gt_dataset, options=experiment_options)
        """
        )
        
        st.subheader("Example Output")
        st.code(
        """
[
    {
        "human_message": "What is the price of copper?",
        "tool_calls": [
            {
                "name": "get_metal_price",
                "args": {
                    "metal_name": "copper"
                }
            }
        ],
        "ai_messages": [
            "",
            "The current price of copper is $0.0098 per gram."
        ],
        "tool_outputs": [
            "0.0098"
        ],
        "ground_truth_answer": "The current price of copper is $0.0098 per gram.",
        "ground_truth_tool_outputs": [
            "0.0098"
        ],
        "reference_tool_calls": [
            {
                "name": "get_metal_price",
                "args": {
                    "metal_name": "copper"
                }
            }
        ],
        "tool_call_accuracy": 1.0,
        "answer_correctness": 1.0,
        "answer_faithfulness": 3,
        "tool_accuracy": 1.0
    },
    {
        "human_message": "What is the price of gold?",
        "tool_calls": [
            {
                "name": "get_metal_price",
                "args": {
                    "metal_name": "gold"
                }
            }
        ],
        "ai_messages": [
            "",
            "The current price of gold is $88.16 per gram."
        ],
        "tool_outputs": [
            "88.1553"
        ],
        "ground_truth_answer": "The current price of gold is $88.16 per gram.",
        "ground_truth_tool_outputs": [
            "88.1553"
        ],
        "reference_tool_calls": [
            {
                "name": "get_metal_price",
                "args": {
                    "metal_name": "gold"
                }
            }
        ],
        "tool_call_accuracy": 1.0,
        "answer_correctness": 1.0,
        "answer_faithfulness": 3,
        "tool_accuracy": 1.0
    }
]
        """
        )

        st.markdown("**See examples/agent_langgraph_improved.py for a complete example.**")
    
    # Roadmap section
    st.header("Roadmap")
    roadmap_cols = st.columns(3)
    
    with roadmap_cols[0]:
        st.subheader("Basic Features")
        st.markdown("""
        - Custom metric integration
        - Multi-turn conversation evaluation
        - Composite, multi-hop Q&A generator
        - Multiple ground truth answers
        """)
    
    with roadmap_cols[1]:
        st.subheader("Multimodal RAG Support")
        st.markdown("""
        - Image retrieval evaluation
        - Table content processing
        - Cross-modal metrics
        """)
    
    with roadmap_cols[2]:
        st.subheader("Agentic Evaluation")
        st.markdown("""
        - Tool call evaluation
        - Node transition evaluation
        """)

# Footer
st.markdown("---")
st.markdown("RAG Evaluation Framework - A comprehensive tool for evaluating Retrieval-Augmented Generation systems")