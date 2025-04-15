import streamlit as st
import os
import sys
import json
import re
import logging
import base64
import time
import csv
import glob
from typing import List, Dict, Optional, Set, Tuple, Any
from openai import OpenAI

# Set up page config
st.set_page_config(page_title="Image Tag Generator", layout="wide")

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
tagging_prompt = """
Describe these images with a set of tags so that they can then be used when creating content. Identify:
- Main subjects, objects, people:
    - red bull affiliated individuals (names if possible)
    - cars, planes, skis etc. with model, livery, specs
    - Technical components (e.g.: front suspension) - be precise (propellor airplane, jet plane)
- Depicted Actions, activities
- Setting, environment
- brands, logos, flags  

Return only a JSON array of tags with no additional text:
["tag1", "tag2", "tag3"]
"""

consolidation_prompt = """
Review this image and analyze the provided tags from previous model runs.
Create a final, consolidated list of accurate tags by:
1. Keeping only tags that actually appear in the image
2. Removing duplicates or near-duplicates
3. Ensuring consistent naming (e.g., choose either 'Formula 1' or 'F1', not both)
4. Adding any important missing tags

Return only a JSON array of finalized tags with no additional text:
["tag1", "tag2", "tag3"]
"""

# Fixed parameters - not configurable through UI
api_delay_seconds = 1.0  # Delay between API calls to prevent rate limiting
default_gpt_model = "gpt-4.1-mini"
default_model_temperature = 0.5
default_detail_level = "low"
default_rerun_count = 0

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses OpenAI's Vision API to generate descriptive tags for images. "
    "The tags can be used for content creation, categorization, and search."
)

# Instructions for running the app
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Configure the input and output paths\n"
    "2. Enter your OpenAI API Key\n"
    "3. (Optional for Pro users) Adjust advanced settings like model parameters and number of processing runs\n"
    "4. Click 'Generate Tags' to process images\n"
    "5. View results"
)

# Classes for image processing and API calls

class ImageProcessor:
    """Handle image processing and file operations"""
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if the file is an image based on its extension"""
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
        _, ext = os.path.splitext(file_path.lower())
        return ext in image_extensions
    
    @staticmethod
    def encode_image(image_path: str) -> Optional[str]:
        """Encode image to base64 with proper error handling"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            st.error(f"Error encoding image '{image_path}': {e}")
            return None
    
    @staticmethod
    def get_image_files(directory_path: str) -> List[str]:
        """Get all image files from the specified directory"""
        if not os.path.exists(directory_path):
            st.error(f"Directory '{directory_path}' does not exist")
            return []
            
        # Find all image files in the directory
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
            image_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        if not image_files:
            st.warning(f"No image files found in '{directory_path}'")
            return []
            
        st.info(f"Found {len(image_files)} image files in '{directory_path}'")
        return image_files
    
    @staticmethod
    def deduplicate_tags(tag_lists: List[List[str]]) -> List[str]:
        """Flatten and deduplicate tags from multiple lists while preserving order"""
        unique_tags = []
        seen = set()
        
        for tags in tag_lists:
            for tag in tags:
                # Convert to lowercase for comparison but keep original case for output
                if tag.lower() not in seen:
                    unique_tags.append(tag)
                    seen.add(tag.lower())
                    
        return unique_tags


class GptVisionClient:
    """Handle GPT Vision API interactions and response processing"""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.3, detail_level: str = "low"):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.detail_level = detail_level
    
    def call_api(self, image_path: str, prompt: str, print_output: bool = False, retries: int = 3, backoff: float = 2.0) -> Optional[str]:
        """Call GPT Vision API with error handling and exponential backoff retries"""
        base64_image = ImageProcessor.encode_image(image_path)
        if not base64_image:
            st.error(f"Failed to encode image: {image_path}")
            return None
        
        attempt = 0
        while attempt < retries:
            try:
                client = OpenAI(api_key=self.api_key)
                response = client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                { "type": "input_text", "text": prompt },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": self.detail_level
                                },
                            ],
                        }
                    ],
                    temperature=self.temperature,
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
                    st.error("Maximum retries reached, skipping this image.")
                    return None
    
    @staticmethod
    def process_output(gpt_output_str: Optional[str]) -> Optional[List[str]]:
        """Process GPT output with validation"""
        if not gpt_output_str:
            return None

        # Extract JSON part from potential code blocks or extra text
        json_match = re.search(r'\[\s*".*"\s*(,\s*".*")*\s*\]', gpt_output_str)
        if json_match:
            gpt_output_str = json_match.group(0)
            
        try:
            tags = json.loads(gpt_output_str)
            if isinstance(tags, list):
                return tags
            else:
                st.error(f"Error: Expected a list of tags but got {type(tags)}")
                return None
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
            return None
    
    def generate_tags(self, image_path: str, prompt: str, print_output: bool = False) -> Optional[List[str]]:
        """Generate tags for an image using the given prompt"""
        output = self.call_api(image_path, prompt, print_output)
        return self.process_output(output)
    
    def consolidate_tags(self, image_path: str, all_tags: List[str], prompt: str) -> Optional[List[str]]:
        """Consolidate tags by validating them against the image"""
        # Create a consolidated prompt that includes the tags to verify
        full_prompt = prompt + "\n\nPreviously identified tags: [" + ", ".join(f'"{tag}"' for tag in all_tags) + "]"
        return self.generate_tags(image_path, full_prompt)


class CsvHandler:
    """Handle CSV operations with context management"""
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        self._existing_entries: Optional[Set[str]] = None

    @property
    def existing_entries(self) -> Set[str]:
        """Lazily load existing entries as filename strings"""
        if self._existing_entries is None:
            self._existing_entries = self._load_existing_entries()
        return self._existing_entries

    def _load_existing_entries(self) -> Set[str]:
        if not os.path.exists(self.filename):
            return set()
        with open(self.filename, mode='r', encoding='utf-8') as f:
            return {row['fileName'] for row in csv.DictReader(f)}

    def write_tags(self, tag_results: List[Dict]) -> None:
        """Write new tag results to CSV file"""
        new_entries = [
            entry for entry in tag_results
            if entry.get('fileName') not in self.existing_entries
        ]

        if not new_entries:
            st.info("No new entries to write")
            return

        mode = 'a' if os.path.exists(self.filename) else 'w'
        with open(self.filename, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if mode == 'w':
                writer.writeheader()
            writer.writerows(new_entries)

        st.success(f"Wrote {len(new_entries)} new entries")


# Configuration UI
st.title("Image Tag Generator")
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    inputFilePath = st.text_input("Input Directory Path", default_input_path, help="Select the folder containing images to process")
    
with col2:
    outputFilePath = st.text_input("Output CSV Path", default_output_path, help="Where to save the generated tags")

APIKey = st.text_input("OpenAI API Key", default_api_key, type="password", help="Your OpenAI API key is required for image processing")

# Pro settings in a collapsible section
with st.expander("🔧 Advanced Settings (Pro Users Only)", expanded=False):
    st.warning("⚠️ These settings are for advanced users only. Changing these values may affect tag quality and API usage.")
    
    st.markdown("#### Model Fine-tuning")
    
    col1, col2 = st.columns(2)
    with col1:
        gptModel = st.selectbox("GPT Model", ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"], index=0, help="Select which OpenAI model to use for image analysis")
        detailLevel = st.selectbox("Detail Level", ["low", "high"], index=0, 
                                  help="Level of detail in image analysis - higher uses more tokens")

    with col2:
        rerunCount = st.number_input("Additional Runs per Image", 0, 5, default_rerun_count, 
                                    help="More runs can improve tag accuracy but increase API usage")
        modelTemperature = st.slider("Model Temperature", 0.0, 1.0, default_model_temperature, 0.1, 
                                help="Higher values produce more creative but potentially less accurate results")
        
# Display the prompts (read-only)
with st.expander("Tagging Prompt", expanded=False):
    # Use the tagging_prompt variable instead of hard-coding the text
    st.markdown("### Tagging Prompt")
    st.markdown(f"```\n{tagging_prompt.strip()}\n```")
    
with st.expander("Consolidation Prompt", expanded=False):
    # Use the consolidation_prompt variable instead of hard-coding the text
    st.markdown("### Consolidation Prompt")
    st.markdown(f"```\n{consolidation_prompt.strip()}\n```")

        
# Main App Logic
if st.button("Generate Tags"):
    # Validate input and output paths first
    if not inputFilePath or inputFilePath.isspace():
        st.error("⚠️ Input directory path cannot be empty. Please specify a valid directory containing images.")
    elif not outputFilePath or outputFilePath.isspace():
        st.error("⚠️ Output CSV path cannot be empty. Please specify where to save the tags CSV file.")
    elif not APIKey:
        st.error("⚠️ Please enter a valid OpenAI API Key")
    else:
        # Initialize processor and client
        image_processor = ImageProcessor()
        gpt_client = GptVisionClient(
            api_key=APIKey,
            model=gptModel,
            temperature=modelTemperature,
            detail_level=detailLevel
        )
        
        # Get all image files from input directory
        with st.spinner("Finding image files..."):
            image_files = image_processor.get_image_files(inputFilePath)
            
        # New check to ensure we have valid image files
        if not image_files:
            st.error(f"No valid images found in '{inputFilePath}'. Please check the directory path and ensure it contains supported image formats (JPG, PNG, etc.)")
        elif image_files:
            # Display a few sample images
            with st.expander("Images to Process", expanded=False):
                cols = st.columns(4)
                for i, img_path in enumerate(image_files[:8]):  # Limit display to 8 images
                    with cols[i % 4]:
                        st.image(img_path, caption=os.path.basename(img_path), width=150)
                if len(image_files) > 8:
                    st.info(f"... and {len(image_files) - 8} more images")
            
            # Process each image with GPT Vision API (multiple runs)
            tag_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, image_path in enumerate(image_files):
                file_name = os.path.basename(image_path)
                status_text.info(f"Processing {idx+1}/{len(image_files)}: {file_name}")
                
                with st.expander(f"Processing: {file_name}", expanded=True):
                    st.image(image_path, width=300)
                    
                    # Store all tag results from multiple runs
                    all_runs_tags = []
                    
                    # First run
                    st.markdown(f"**Run 1/{rerunCount+1}...**")
                    tags_container = st.empty()
                    tags = gpt_client.generate_tags(image_path, tagging_prompt, print_output=False)
                    
                    # Check if we got an API key error
                    if isinstance(tags, str) and "Incorrect API key provided" in tags:
                        st.error("API Key error detected. Stopping execution.")
                        break
                    
                    if not tags:
                        st.error(f"Failed to get GPT response for {file_name}")
                        continue
                    
                    all_runs_tags.append(tags)
                    tags_container.json(tags)
                    st.info(f"Run 1: Found {len(tags)} tags")
                    
                    # Additional runs if rerunCount > 0
                    if rerunCount > 0:
                        for run in range(rerunCount):
                            st.markdown(f"**Run {run+2}/{rerunCount+1}...**")
                            run_tags_container = st.empty()
                            run_tags = gpt_client.generate_tags(image_path, tagging_prompt)
                            
                            if run_tags:
                                all_runs_tags.append(run_tags)
                                run_tags_container.json(run_tags)
                                st.info(f"Run {run+2}: Found {len(run_tags)} tags")
                            else:
                                st.error(f"Run {run+2}: Failed to get tags")
                            
                            # Add a delay to prevent API rate limiting
                            time.sleep(api_delay_seconds)
                    
                    # Deduplicate tags from all runs
                    unique_tags = image_processor.deduplicate_tags(all_runs_tags)
                    st.markdown("**After deduplication:**")
                    st.json(unique_tags)
                    st.info(f"Found {len(unique_tags)} unique tags")
                    
                    # If we have multiple runs, consolidate results with a final verification pass
                    final_tags = tags  # Default to first run results
                    if len(all_runs_tags) > 1:
                        st.markdown("**Performing final consolidation run...**")
                        consolidated_container = st.empty()
                        consolidated_tags = gpt_client.consolidate_tags(image_path, unique_tags, consolidation_prompt)
                        
                        if consolidated_tags:
                            final_tags = consolidated_tags
                            consolidated_container.json(final_tags)
                            st.success(f"Consolidated to {len(final_tags)} verified tags")
                        else:
                            st.error("Consolidation failed, using deduplicated tags from all runs")
                            final_tags = unique_tags
                    
                    # Store final results
                    tag_results.append({
                        "fileName": file_name,
                        "tags": final_tags
                    })
                    
                    st.markdown("---")
                    st.markdown("**Final tags:**")
                    st.success(f"{len(final_tags)} tags: {', '.join(final_tags)}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(image_files))
                
                # Add a delay to prevent API rate limiting
                if idx < len(image_files) - 1:  # Don't delay after the last image
                    time.sleep(api_delay_seconds)
            
            status_text.success(f"Successfully processed {len(tag_results)} images")
            
            # Initialize CSV handler and prepare final data
            if tag_results:
                with st.spinner("Writing tags to CSV..."):
                    # Prepare final data for CSV
                    csv_entries = []
                    for result in tag_results:
                        # Skip entries where no tags were extracted
                        if not result['tags']:
                            st.warning(f"Skipping {result['fileName']} - no tags extracted")
                            continue
                            
                        # Convert tags list to string for storage in CSV
                        entry = {
                            "fileName": result['fileName'],
                            "tags": ", ".join(result['tags'])
                        }
                        
                        csv_entries.append(entry)

                    # Write to CSV
                    if csv_entries:
                        csv_handler = CsvHandler(
                            outputFilePath,
                            ["fileName", "tags"]
                        )
                        csv_handler.write_tags(csv_entries)
                        
                        # Display table with results
                        st.subheader("Generated Tags")
                        results_df = {
                            "File Name": [entry["fileName"] for entry in csv_entries],
                            "Tags": [entry["tags"] for entry in csv_entries]
                        }
                        st.dataframe(results_df)
                    else:
                        st.warning("No tags to write to CSV")
