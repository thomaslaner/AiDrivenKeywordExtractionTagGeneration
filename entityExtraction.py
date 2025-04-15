import streamlit as st
import os
import sys
import json
import re
import logging
import unicodedata
import time
import csv
import glob
from typing import Dict, List, Set, Optional, Tuple, Any
from openai import OpenAI

# Set up page config
st.set_page_config(page_title="Entity Extractor", layout="wide")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Default input parameters
default_input_path = ""
default_output_path = ""
default_api_key = ""

# Static prompts - not editable in the UI
extraction_prompt = """
Goal:
Extract from the provided article the following entities:
1. AthletesAndTeams: List individuals and teams affiliated with Red Bull. List any aliases or variations of the team names and correct any spelling mistakes. If someone is known by a nickname, use nickname instead of name.
2. Disciplines: Capture every mention of competitive sports & e-sports disciplines. Consider both full names and common abbreviations.
3. Events: Identify any formally named tournaments, championships, or events (e.g.: “League of Legends World Championship”).

Additional Instructions:
- Translate all Discipline- and Event names to English
- Search entire text (including background or historical references) for all explicit and implicit references to the above categories.
- Return exactly one JSON object containing the keys “AthletesAndTeams”, “Disciplines”, and “Events”. If any of categories not mentioned, provide empty array for that key.
- Do only include mentions from the article, not from the instruction.

Output single JSON object with these exact keys, no extra text or different formatting should be returned:
{
"AthletesAndTeams": [],
"Disciplines": [],
"Events": []
}

Article:
"""

rerun_analysis_prompt = """
From web-articles extractions below, make sure all entries are in English, with no duplicates, and names spelled correctly. Return a single JSON object with same keys as inputs.
"""

# Fixed parameters - not configurable through UI
api_delay_seconds = 0.5  # Delay between API calls to prevent rate limiting
default_gpt_model = "gpt-4.1-mini"
default_model_temperature = 0.5
default_rerun_count = 0

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses OpenAI's GPT to extract entities from articles. "
    "It analyzes text to identify Red Bull athletes, sports disciplines, and events."
)

# Instructions for running the app
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Configure the input and output paths\n"
    "2. Enter your OpenAI API Key\n"
    "3. (Optional for Pro users) Adjust advanced settings like model parameters and number of processing runs\n"
    "4. Click 'Extract Entities' to process articles\n"
    "5. View results"
)

# Helper classes

class ArticleProcessor:
    """Handle article loading, cleaning, and processing operations"""
    
    @staticmethod
    def load_article(file_path: str) -> Optional[Dict[str, Any]]:
        """Load the JSON file from the given path with proper error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            st.error(f"Error: The file '{file_path}' was not found.")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Error: Failed to parse JSON file '{file_path}': {e}")
            return None
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text"""
        try:
            from bs4 import BeautifulSoup
            # Use BeautifulSoup for robust HTML tag removal
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except ImportError:
            # Fallback regex to remove <...> tags
            html_tag_pattern = re.compile(r'<[^>]+>')
            return html_tag_pattern.sub('', text)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean the text by applying multiple cleaning steps"""
        # Remove HTML tags
        text = ArticleProcessor.remove_html_tags(text)
        
        # Normalize Unicode (standardizes quotes, dashes, spaces, etc.)
        text = unicodedata.normalize("NFKC", text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove emojis using a Unicode-aware regex
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"  # other symbols
            u"\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        
        # Remove hyperlinks (URLs)
        link_pattern = re.compile(r'http\S+|www\.\S+')
        text = link_pattern.sub('', text)
        
        # Remove hashtag symbols (keep the text) and Twitter handles (@username)
        text = re.sub(r'#', '', text)
        text = re.sub(r'@\w+', '', text)

        # Final whitespace normalization (in case removals left extra spaces)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    @staticmethod
    def prepare_article(data: Dict[str, Any]) -> Dict[str, str]:
        """Extract and clean fields from the JSON article"""
        headline = ArticleProcessor.clean_text(data.get('headline', ''))
        article_body = ArticleProcessor.clean_text(data.get('articleBody', ''))
        date_published = ArticleProcessor.clean_text(data.get('datePublished', ''))

        return {
            "headline": headline,
            "articleBody": article_body,
            "datePublished": date_published
        }
    
    @staticmethod
    def get_json_files(directory_path: str) -> List[str]:
        """Get all JSON files from the specified directory"""
        if not os.path.exists(directory_path):
            st.error(f"Directory '{directory_path}' does not exist")
            return []
            
        # Find all .json files in the directory
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        
        if not json_files:
            st.warning(f"No JSON files found in '{directory_path}'")
            return []
            
        st.info(f"Found {len(json_files)} JSON files in '{directory_path}'")
        return json_files


class GptClient:
    """Handle GPT API interactions and response processing"""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.5):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    def call_api(self, prompt: str, print_output: bool = False, retries: int = 3, backoff: float = 2.0) -> Optional[str]:
        """Call GPT API with error handling and exponential backoff retries"""
        attempt = 0
        while attempt < retries:
            try:
                client = OpenAI(api_key=self.api_key)
                response = client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=self.temperature
                )
                if print_output:
                    st.text(response.output_text)
                return response.output_text
            except Exception as e:
                attempt += 1
                wait_time = backoff ** attempt
                st.error(f"API call error ({attempt}/{retries}): {e}. Retrying in {wait_time} seconds...")
                if attempt < retries:
                    time.sleep(wait_time)
                else:
                    st.error("Maximum retries reached, skipping this article.")
                    return None
    
    @staticmethod
    def process_output(gpt_output_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Process GPT output with validation"""
        if not gpt_output_str:
            return None

        # Extract JSON part from potential code blocks or extra text
        json_match = re.search(r'\{[\s\S]*\}', gpt_output_str)
        if json_match:
            gpt_output_str = json_match.group(0)
            
        try:
            data = json.loads(gpt_output_str)
            return data
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
            return None
    
    def extract_entities(self, article_text: str, extraction_prompt: str, print_output: bool = False) -> Optional[Dict[str, Any]]:
        """Extract entities from an article using the extraction prompt"""
        prompt = extraction_prompt + article_text
        output = self.call_api(prompt, print_output)
        return self.process_output(output)
    
    def consolidate_results(self, extraction_results: List[Dict[str, Any]], consolidation_prompt: str) -> Optional[Dict[str, Any]]:
        """Consolidate multiple extraction results"""
        consolidated_input = json.dumps(extraction_results, indent=2)
        full_prompt = consolidation_prompt + "\n\n" + consolidated_input
        consolidated_output = self.call_api(full_prompt)
        return self.process_output(consolidated_output)


class CsvHandler:
    """Handle CSV operations with context management"""
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        self._existing_entries: Optional[Set[Tuple[str, str, str]]] = None

    @property
    def existing_entries(self) -> Set[Tuple[str, str, str]]:
        """Lazily load existing entries as tuples of (file_path, headline, date_published)"""
        if self._existing_entries is None:
            self._existing_entries = self._load_existing_entries()
        return self._existing_entries

    def _load_existing_entries(self) -> Set[Tuple[str, str, str]]:
        if not os.path.exists(self.filename):
            return set()
        with open(self.filename, mode='r', encoding='utf-8') as f:
            return {(row['FileName'], row['Headline'], row['DatePublished']) 
                   for row in csv.DictReader(f)}

    def write_extractions(self, extractions: List[Dict]) -> None:
        """Write new extractions to CSV file"""
        new_extractions = [
            ext for ext in extractions
            if (ext.get('FileName'), ext.get('Headline'), ext.get('DatePublished')) 
               not in self.existing_entries
        ]

        if not new_extractions:
            st.info("No new entries to write")
            return

        mode = 'a' if os.path.exists(self.filename) else 'w'
        with open(self.filename, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if mode == 'w':
                writer.writeheader()
            writer.writerows(new_extractions)

        st.success(f"Wrote {len(new_extractions)} new entries")


# Configuration UI
st.title("Entity Extractor")
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    inputFilePath = st.text_input("Input Directory Path", default_input_path, 
                                help="Select the folder containing JSON articles to process")
    
with col2:
    outputFilePath = st.text_input("Output CSV Path", default_output_path, 
                                 help="Where to save the extracted entities CSV file")

APIKey = st.text_input("OpenAI API Key", default_api_key, type="password", 
                     help="Your OpenAI API key is required for entity extraction")

# Pro settings in a collapsible section
with st.expander("🔧 Advanced Settings (Pro Users Only)", expanded=False):
    st.warning("⚠️ These settings are for advanced users only. Changing these values may affect extraction quality and API usage.")
    
    st.markdown("#### Model Fine-tuning")
    
    col1, col2 = st.columns(2)
    with col1:
        gptModel = st.selectbox("GPT Model", ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"], index=0, 
                               help="Select which OpenAI model to use for analysis")
        
    with col2:
        rerunCount = st.number_input("Additional Runs per Article", 0, 5, default_rerun_count, 
                                    help="More runs can improve extraction accuracy but increase API usage")
        
    modelTemperature = st.slider("Model Temperature", 0.0, 1.0, default_model_temperature, 0.1, 
                               help="Higher values produce more creative but potentially less accurate results")
        
# Display the prompts (read-only)
with st.expander("Extraction Prompt", expanded=False):
    # Use the extraction_prompt variable instead of hard-coding the text
    st.markdown("### Extraction Prompt")
    st.markdown(f"```\n{extraction_prompt.strip()}\n```")
    
with st.expander("Consolidation Prompt", expanded=False):
    # Use the rerun_analysis_prompt variable instead of hard-coding the text
    st.markdown("### Consolidation Prompt")
    st.markdown(f"```\n{rerun_analysis_prompt.strip()}\n```")

        
# Main App Logic
if st.button("Extract Entities"):
    # Validate input and output paths first
    if not inputFilePath or inputFilePath.isspace():
        st.error("⚠️ Input directory path cannot be empty. Please specify a valid directory containing JSON articles.")
    elif not outputFilePath or outputFilePath.isspace():
        st.error("⚠️ Output CSV path cannot be empty. Please specify where to save the entities CSV file.")
    elif not APIKey:
        st.error("⚠️ Please enter a valid OpenAI API Key")
    else:
        # Initialize processor and client
        article_processor = ArticleProcessor()
        gpt_client = GptClient(
            api_key=APIKey,
            model=gptModel,
            temperature=modelTemperature
        )
        
        # Get all JSON files from input directory
        with st.spinner("Finding JSON files..."):
            json_files = article_processor.get_json_files(inputFilePath)
            
        # New check to ensure we have valid JSON files
        if not json_files:
            st.error(f"No valid JSON files found in '{inputFilePath}'. Please check the directory path.")
        elif json_files:
            # Initialize structures to hold processed data
            processed_articles = []
            extraction_results = []
            
            # Display a few sample articles
            with st.expander("Articles to Process", expanded=False):
                for i, file_path in enumerate(json_files[:5]):  # Limit display to 5 articles
                    data = ArticleProcessor.load_article(file_path)
                    if data is None:
                        continue
                    
                    cleaned = ArticleProcessor.prepare_article(data)
                    st.markdown(f"**{i+1}. {cleaned['headline']}**")
                    st.caption(f"Published: {cleaned['datePublished']}")
                    st.text(cleaned['articleBody'][:200] + "..." if len(cleaned['articleBody']) > 200 else cleaned['articleBody'])
                    st.markdown("---")
                    
                if len(json_files) > 5:
                    st.info(f"... and {len(json_files) - 5} more articles")
            
            # Process each file
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file_path in enumerate(json_files):
                status_text.info(f"Processing {idx+1}/{len(json_files)}: {os.path.basename(file_path)}")
                
                # Load and prepare article
                data = ArticleProcessor.load_article(file_path)
                if data is None:
                    st.warning(f"Skipping {file_path} due to loading error")
                    continue
                    
                # Clean and prepare article content
                cleaned_article = ArticleProcessor.prepare_article(data)
                
                # Store article data for processing
                processed_articles.append({
                    "file_path": file_path,
                    "headline": cleaned_article["headline"],
                    "article_body": cleaned_article["articleBody"],
                    "date_published": cleaned_article["datePublished"]
                })
                
                # Update progress
                progress_bar.progress((idx + 1) / len(json_files))
            
            status_text.success(f"Successfully processed {len(processed_articles)} articles")
            st.markdown("---")
            st.subheader("Extracting Entities")
            
            # Extract entities using GPT
            extraction_progress = st.progress(0)
            extraction_status = st.empty()
            
            for idx, article in enumerate(processed_articles):
                with st.expander(f"Processing: {os.path.basename(article['file_path'])}", expanded=False):
                    st.markdown(f"**{article['headline']}**")
                    st.caption(f"Published: {article['date_published']}")
                    
                    # First extraction run
                    output_container = st.empty()
                    output_container.info("Extracting entities...")
                    
                    output_dict = gpt_client.extract_entities(article['article_body'], extraction_prompt)
                    
                    # Check if extraction failed
                    if not output_dict:
                        st.error(f"Failed to extract entities")
                        continue
                    
                    # If rerunCount > 0, do multiple runs and consolidate results
                    if rerunCount > 0:
                        st.markdown(f"**Performing {rerunCount} additional runs for validation...**")
                        all_extraction_results = [output_dict]  # Store first run
                        
                        # Perform additional runs
                        for run in range(rerunCount):
                            run_status = st.empty()
                            run_status.info(f"Run {run+1}/{rerunCount}...")
                            
                            rerun_dict = gpt_client.extract_entities(article['article_body'], extraction_prompt)
                            if rerun_dict:
                                all_extraction_results.append(rerun_dict)
                                run_status.success(f"Run {run+1} completed")
                            else:
                                run_status.error(f"Run {run+1} failed")
                            
                            # Add a small delay to prevent API rate limiting
                            time.sleep(api_delay_seconds)
                        
                        # Consolidate results using rerunAnalysisPrompt if we have multiple results
                        if len(all_extraction_results) > 1:
                            consolidation_status = st.empty()
                            consolidation_status.info("Consolidating results from multiple runs...")
                            
                            # Call GPT to consolidate results
                            consolidated_dict = gpt_client.consolidate_results(all_extraction_results, rerun_analysis_prompt)
                            if consolidated_dict:
                                output_dict = consolidated_dict  # Replace with consolidated results
                                consolidation_status.success("Successfully consolidated multiple extraction runs")
                            else:
                                consolidation_status.error("Failed to get consolidated output, using first run results")
                    
                    # Extract entities from the result
                    athletes_teams = output_dict.get("AthletesAndTeams", output_dict.get("RedBullAthletesAndTeams", []))
                    disciplines = output_dict.get("Disciplines", [])
                    events = output_dict.get("Events", [])
                    
                    # Convert to arrays if they're not already
                    if isinstance(athletes_teams, str) and athletes_teams:
                        athletes_teams = [athletes_teams]
                    elif not isinstance(athletes_teams, list):
                        athletes_teams = []
                        
                    if isinstance(disciplines, str) and disciplines:
                        disciplines = [disciplines]
                    elif not isinstance(disciplines, list):
                        disciplines = []
                        
                    if isinstance(events, str) and events:
                        events = [events]
                    elif not isinstance(events, list):
                        events = []
                    
                    # Display the extracted entities
                    output_container.json(output_dict)
                    
                    st.markdown("**Extracted Entities:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Athletes/Teams:**")
                        if athletes_teams:
                            st.write(", ".join(athletes_teams))
                        else:
                            st.write("None found")
                    
                    with col2:
                        st.markdown("**Disciplines:**")
                        if disciplines:
                            st.write(", ".join(disciplines))
                        else:
                            st.write("None found")
                    
                    st.markdown("**Events:**")
                    if events:
                        st.write(", ".join(events))
                    else:
                        st.write("None found")
                    
                    # Store results
                    extraction_results.append({
                        "file_path": article['file_path'],
                        "headline": article['headline'],
                        "date_published": article['date_published'],
                        "athletes_teams": athletes_teams,
                        "disciplines": disciplines,
                        "events": events
                    })
                
                extraction_status.info(f"Processed {idx+1}/{len(processed_articles)} articles")
                extraction_progress.progress((idx + 1) / len(processed_articles))
                
                # Add a small delay to prevent API rate limiting
                if idx < len(processed_articles) - 1:  # Don't delay after the last article
                    time.sleep(api_delay_seconds)
            
            extraction_status.success(f"Successfully extracted data from {len(extraction_results)} articles")
            st.markdown("---")
            
            # Write extractions to CSV
            if extraction_results:
                with st.spinner("Writing extractions to CSV..."):
                    # Prepare final data for CSV
                    csv_extractions = []
                    for result in extraction_results:
                        # Skip entries where all extraction fields are empty
                        if not result['athletes_teams'] and not result['disciplines'] and not result['events']:
                            st.warning(f"Skipping {os.path.basename(result['file_path'])} - no entities extracted")
                            continue
                        
                        # Ensure lists contain only strings before joining
                        athletes_teams_str = ", ".join([str(item) for item in result['athletes_teams']]) if result['athletes_teams'] else ""
                        disciplines_str = ", ".join([str(item) for item in result['disciplines']]) if result['disciplines'] else ""
                        events_str = ", ".join([str(item) for item in result['events']]) if result['events'] else ""
                        
                        extraction = {
                            "FileName": os.path.basename(result['file_path']),
                            "Headline": result['headline'],
                            "DatePublished": result['date_published'],
                            "AthletesAndTeams": athletes_teams_str,
                            "Disciplines": disciplines_str,
                            "Events": events_str
                        }
                        
                        csv_extractions.append(extraction)
                    
                    # Write to CSV
                    if csv_extractions:
                        csv_handler = CsvHandler(
                            outputFilePath,
                            ["FileName", "Headline", "DatePublished", "AthletesAndTeams", "Disciplines", "Events"]
                        )
                        csv_handler.write_extractions(csv_extractions)
                        
                        # Display extraction summary
                        st.subheader("Extraction Summary")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Articles with Athletes/Teams", sum(1 for r in extraction_results if r['athletes_teams']))
                        with col2:
                            st.metric("Articles with Disciplines", sum(1 for r in extraction_results if r['disciplines']))
                        with col3:
                            st.metric("Articles with Events", sum(1 for r in extraction_results if r['events']))
                        
                        # Display table with results
                        st.subheader("Extracted Entities")
                        results_df = {
                            "File Name": [os.path.basename(result['file_path']) for result in extraction_results],
                            "Headline": [result['headline'] for result in extraction_results],
                            "Athletes/Teams": [", ".join(result['athletes_teams']) if result['athletes_teams'] else "None" for result in extraction_results],
                            "Disciplines": [", ".join(result['disciplines']) if result['disciplines'] else "None" for result in extraction_results],
                            "Events": [", ".join(result['events']) if result['events'] else "None" for result in extraction_results]
                        }
                        st.dataframe(results_df)
                    else:
                        st.warning("No entities to write to CSV")
