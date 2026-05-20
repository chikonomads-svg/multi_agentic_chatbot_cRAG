# Enhanced Image Prompt App

A Streamlit application for generating AI-powered presentation slides with 2026 trending topics, featuring multi-provider LLM and image generation support.

## Features

- **Multi-Provider LLM Support**: OpenAI, Azure OpenAI, Moonshot Kimi, Anthropic Claude, Google Gemini
- **Multi-Provider Image Generation**: Azure FLUX, OpenAI DALL-E, Stability AI
- **Visual Workflow Tracker**: Real-time progress tracking with animated step indicators
- **Text-Free Backgrounds**: Professional presentation backgrounds for easy text overlay
- **User API Key Input**: Secure credential management with session-only storage

## Project Structure

```
Imageprompt_app/
├── app.py                 # Main Streamlit application
├── logic.py              # Core business logic and API integrations
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml      # Streamlit configuration
└── .gitignore           # Git ignore rules
```

## Deployment Instructions

### 1. Streamlit Cloud (Recommended)

1. **Fork/Clone this repository** to your GitHub account

2. **Sign up for Streamlit Cloud**: https://streamlit.io/cloud

3. **Deploy the app**:
   - Connect your GitHub repository
   - Select the `Imageprompt_app` folder
   - Set main file path to `app.py`

4. **Configure Secrets** (Optional but recommended):
   - In Streamlit Cloud, go to your app → Settings → Secrets
   - Add your default API keys in TOML format:
   ```toml
   OPENAI_API_KEY = "your-key-here"
   AZURE_FLUX_API_KEY = "your-key-here"
   TAVILY_API_KEY = "your-key-here"
   ```

### 2. Local Deployment

```bash
# Clone the repository
git clone https://github.com/chikonomads-svg/Enhanced_image_prompting.git
cd Enhanced_image_prompting/Imageprompt_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 3. Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## Required API Keys

Users need to provide their own API keys for the app to function:

### For Research (Optional but recommended):
- **Tavily API Key** (https://tavily.com) - For better web search results

### For Text Generation (Required):
Choose one or more:
- **OpenAI API Key** (https://platform.openai.com)
- **Azure OpenAI** - Requires endpoint URL and API key
- **Moonshot Kimi API Key** (https://platform.moonshot.cn)
- **Anthropic Claude API Key** (https://console.anthropic.com)
- **Google Gemini API Key** (https://makersuite.google.com)

### For Image Generation (Required):
Choose one or more:
- **Azure FLUX API Key** - For FLUX.2-pro image generation
- **OpenAI API Key** - For DALL-E image generation
- **Stability AI API Key** (https://platform.stability.ai) - For Stable Diffusion

## Security Features

✅ **No Hardcoded Credentials**: All API keys are entered by users and stored only in session memory  
✅ **Session-Only Storage**: Keys are never written to disk or logged  
✅ **Password Input Fields**: API keys are masked with password type inputs  
✅ **Environment Variable Support**: Keys can be loaded from environment variables  
✅ **Comprehensive .gitignore**: Prevents accidental credential commits  

## Usage Workflow

1. **Configure Models**: Enter your API keys in the sidebar and click "Configure"
2. **Research Trends**: Enter a topic (e.g., "AI Trends 2026") and search
3. **Select Topic**: Choose from researched trending topics
4. **Generate Slides**: App creates 10 dynamic slides with researched content
5. **Edit Prompts**: Customize image generation prompts if needed
6. **Finalize**: Lock prompts when satisfied
7. **Generate Images**: Create AI backgrounds for each slide
8. **Download**: Export PNG images for use in presentations

## Environment Variables

For local deployment, you can set these environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export AZURE_FLUX_API_KEY="your-azure-flux-key"
export TAVILY_API_KEY="your-tavily-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

On Windows:
```cmd
set OPENAI_API_KEY=your-openai-key
set AZURE_FLUX_API_KEY=your-azure-flux-key
```

## License

MIT License - Feel free to use and modify as needed.

## Support

For issues or questions, please open an issue on the GitHub repository.
