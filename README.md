# AI-Driven Keyword Extraction & Tag Generation

This project provides an end-to-end, LLM-powered system for structured information extraction and tag generation from multimedia content. It is designed to help organizations understand which athletes, teams, disciplines, and events are being covered in media, and enrich visual content with consistent tags for search and content management.

## Features

- **Entity Extraction from Articles**
  - Extracts company-related athletes, teams, disciplines, and events
  - Configurable GPT-4.1-based pipeline with support for multiple reruns and temperature tuning
  - Consolidation logic to merge multiple runs for maximum recall

- **Tag Generation from Images**
  - Uses GPT-4 Vision API to describe images with high-quality tags
  - Detects subjects, actions, settings, brand elements, and technical components
  - Tag consolidation prompt ensures consistent and relevant output

- **Prompt Engineering Framework**
  - Structured prompt design (Goal, Format, Constraints, Context)
  - Evaluation loop based on test tiers: minimal, small, and full sample
  - Built-in evaluation criteria (recall, precision, F1, confidence-based scoring planned)

- **Front-End Apps (Streamlit)**
  - Configurable UIs for both text and image pipelines
  - Advanced settings panel for pro users to adjust model, temperature, runs

## Tech Stack

- **Backend:** Python 3, OpenAI GPT-4.1 via Responses API
- **Frontend:** Streamlit

## Evaluation Highlights

- Multi-model and rerun comparison (GPT-4.1 vs GPT-4.1 mini)
- A/B testing setup to compare prompts and models
- Prompt consolidation via LLM to ensure structured, consistent results
- Modular design for versioned prompt & model swapping

## Project Structure
<pre>
├── entityExtraction.py        # Entity extraction logic from article JSONs
├── entityExtraction.ipynb     # Jupyter notebook for iterative prompt tuning and testing
├── tagGeneration.py           # Image tag generation pipeline using GPT-4 Vision
├── tagGeneration.ipynb        # Visual exploration and prompt iterations for tagging
├── ProjectPresentation.pdf    # Project presentation for case study
├── LICENSE                    # MIT license
└── README.md                  # This file
</pre>

## Setup & Usage

1. Clone the repository  
2. Install dependencies  
3. Set environment variables  
4. Run extraction or tagging  
   - Use `entityExtraction.ipynb` for articles  
   - Use `tagGeneration.ipynb` for image tag generation  
5. Optionally, launch the Streamlit apps  
   - `streamlit run entityExtraction.py`  
   - `streamlit run tagGeneration.py`

## Future Enhancements
 - Confidence scores for model outputs
	- Domain-specific semantic validation
	- Adaptive prompt rerunning based on low-confidence tags
	- Fine-tuning or model personalization
 - Human-in-the-loop feedback interface

## License

MIT License – see LICENSE for details.
 
