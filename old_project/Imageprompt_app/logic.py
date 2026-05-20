"""
Logic module for Imageprompt_app.
Dynamic slide titles based on researched content - informative and trending.
Supports multiple LLM and Image model providers.
"""

import os
import json
import logging
import re
import base64
import requests
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Provider configurations
LLM_PROVIDERS = {
    "OpenAI": {
        "name": "OpenAI GPT",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "key_name": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1"
    },
    "AzureOpenAI": {
        "name": "Azure OpenAI",
        "models": ["gpt-4o", "gpt-4", "gpt-35-turbo"],
        "default_model": "gpt-4o",
        "key_name": "AZURE_OPENAI_API_KEY",
        "base_url": "AZURE_OPENAI_ENDPOINT"  # Custom endpoint required
    },
    "Kimi": {
        "name": "Moonshot Kimi",
        "models": ["kimi-k2-5", "kimi-k2", "kimi-k1.5"],
        "default_model": "kimi-k2-5",
        "key_name": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.cn/v1"
    },
    "Anthropic": {
        "name": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "default_model": "claude-3-5-sonnet-20241022",
        "key_name": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1"
    },
    "Google": {
        "name": "Google Gemini",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "default_model": "gemini-1.5-flash",
        "key_name": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1"
    }
}

IMAGE_PROVIDERS = {
    "AzureFLUX": {
        "name": "Azure FLUX.2-pro",
        "models": ["FLUX.2-pro", "FLUX.2-dev", "FLUX.2-schnell"],
        "default_model": "FLUX.2-pro",
        "key_name": "AZURE_FLUX_API_KEY",
        "endpoint": "https://chikoai.services.ai.azure.com/providers/blackforestlabs/v1/flux-2-pro"
    },
    "OpenAI": {
        "name": "OpenAI DALL-E",
        "models": ["dall-e-3", "dall-e-2"],
        "default_model": "dall-e-3",
        "key_name": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1"
    },
    "StabilityAI": {
        "name": "Stability AI",
        "models": ["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6"],
        "default_model": "stable-diffusion-xl-1024-v1-0",
        "key_name": "STABILITY_API_KEY",
        "base_url": "https://api.stability.ai/v2beta"
    }
}


def setup_logging() -> logging.Logger:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"imageprompt_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()


class LLMClient:
    """Universal LLM client supporting multiple providers."""
    
    def __init__(self, provider: str, api_key: str, model: str = None, **kwargs):
        self.provider = provider
        self.api_key = api_key
        self.model = model or LLM_PROVIDERS[provider]["default_model"]
        self.config = LLM_PROVIDERS[provider]
        self.extra_config = kwargs
        
    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using the selected provider."""
        try:
            if self.provider == "OpenAI":
                return self._call_openai(prompt, system_prompt)
            elif self.provider == "AzureOpenAI":
                return self._call_azure_openai(prompt, system_prompt)
            elif self.provider == "Kimi":
                return self._call_kimi(prompt, system_prompt)
            elif self.provider == "Anthropic":
                return self._call_anthropic(prompt, system_prompt)
            elif self.provider == "Google":
                return self._call_google(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return f"Error generating content: {str(e)}"
    
    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        response = requests.post(
            f"{self.config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_azure_openai(self, prompt: str, system_prompt: str) -> str:
        endpoint = self.extra_config.get("azure_endpoint", "")
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        response = requests.post(
            f"{endpoint}/chat/completions?api-version=2024-02-01",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_kimi(self, prompt: str, system_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        response = requests.post(
            f"{self.config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": 0.7,
            "system": system_prompt or "You are a helpful assistant.",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            f"{self.config['base_url']}/messages",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    
    def _call_google(self, prompt: str, system_prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{
                "parts": [
                    {"text": system_prompt + "\n\n" + prompt if system_prompt else prompt}
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4000
            }
        }
        response = requests.post(
            f"{self.config['base_url']}/models/{self.model}:generateContent?key={self.api_key}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]


class ImagePromptResearcher:
    """Researcher that supports multiple LLM providers."""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client
        # Tavily is used for web search (requires its own key)
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = None
        if self.tavily_key:
            self.tavily_client = TavilyClient(api_key=self.tavily_key)
    
    def set_llm_client(self, llm_client: LLMClient):
        """Set or update the LLM client."""
        self.llm_client = llm_client
    
    def research_trending_topics(self, topic: str) -> List[Dict[str, Any]]:
        """Research trending topics using Tavily."""
        if not self.tavily_client:
            # Fallback: use LLM to generate topics
            return self._generate_topics_with_llm(topic)
        
        try:
            response = self.tavily_client.search(
                query=f"{topic} latest trends breaking news 2026",
                max_results=10, include_answer=True, include_raw_content=True
            )
            trending_topics = []
            for i, result in enumerate(response.get("results", [])[:10], 1):
                title = result.get("title", "No title")
                content = result.get("content", "")
                if self._is_english(title) and self._is_english(content):
                    trending_topics.append({
                        "id": i, "title": title,
                        "description": content[:400],
                        "url": result.get("url", ""),
                        "source": result.get("source", "Unknown")
                    })
            return trending_topics
        except Exception as e:
            logger.error(f"Tavily error: {str(e)}")
            return self._generate_topics_with_llm(topic)
    
    def _generate_topics_with_llm(self, topic: str) -> List[Dict[str, Any]]:
        """Generate topics using LLM when Tavily is unavailable."""
        if not self.llm_client:
            return [{"id": 1, "title": topic, "description": "Research data unavailable. Please configure API keys.", "url": "", "source": "Local"}]
        
        prompt = f"""Generate 5 trending topics related to "{topic}" for 2026.
        For each topic, provide:
        - A catchy title
        - A brief description (150-200 characters)
        
        Format as JSON array:
        [{{"title": "...", "description": "..."}}, ...]
        """
        
        try:
            response = self.llm_client.generate_text(prompt)
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                topics = json.loads(json_match.group())
                return [{"id": i+1, **t, "url": "", "source": "AI Generated"} for i, t in enumerate(topics)]
        except Exception as e:
            logger.error(f"LLM topic generation error: {str(e)}")
        
        return [{"id": 1, "title": topic, "description": "Research data unavailable. Please configure API keys.", "url": "", "source": "Local"}]
    
    def research_dynamic_slides(self, selected_topic: str) -> List[Dict[str, Any]]:
        """Research content for slides."""
        logger.info(f"Researching dynamic content for: {selected_topic}")
        
        if self.tavily_client:
            return self._research_with_tavily(selected_topic)
        else:
            return self._research_with_llm(selected_topic)
    
    def _research_with_tavily(self, topic: str) -> List[Dict[str, Any]]:
        """Research using Tavily API."""
        search_queries = [
            f"{topic} comprehensive overview what is",
            f"{topic} key benefits advantages why use",
            f"{topic} how to implement guide tutorial",
            f"{topic} real examples case studies success stories",
            f"{topic} common challenges problems solutions",
            f"{topic} best practices expert tips",
            f"{topic} tools software platforms",
            f"{topic} future predictions upcoming trends",
            f"{topic} statistics data research findings",
            f"{topic} getting started beginner guide"
        ]
        
        all_results = []
        for query in search_queries:
            try:
                response = self.tavily_client.search(query=query, max_results=3, include_answer=True, include_raw_content=True)
                for r in response.get("results", []):
                    content = r.get("content", "")
                    title = r.get("title", "")
                    if self._is_english(content) and len(content) > 100:
                        all_results.append({"title": title, "content": content, "url": r.get("url", "")})
            except Exception as e:
                logger.error(f"Search error: {e}")
        
        return self._extract_dynamic_slides(topic, all_results)

    def _research_with_llm(self, topic: str) -> List[Dict[str, Any]]:
        """Generate slide content using LLM."""
        if not self.llm_client:
            return self._create_default_slides(topic)

        # Ask the LLM to produce short slide titles (3-6 words) and clear bullet content.
        # Request strict JSON output so it can be parsed reliably.
        prompt = f"""Create a concise 10-slide presentation about \"{topic}\".
        Requirements for each slide:
        - title: short, punchy slide heading (3-6 words)
        - content: 3-5 clear bullet points that explain or expand the title (each bullet 10-40 words)

        Focus the slides on the following areas in this order:
        1. What is {topic}?
        2. Key benefits
        3. How it works
        4. Real-world examples
        5. Challenges
        6. Best practices
        7. Tools/platforms
        8. Future outlook
        9. Statistics/impact
        10. Getting started

        Output format: JSON array of 10 objects exactly, e.g.:
        [
          {"title": "Short Slide Title", "content": ["bullet 1", "bullet 2", ...]},
          ...
        ]

        Make sure titles are short and content bullets are human-readable and slide-ready.
        """

        try:
            response = self.llm_client.generate_text(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                slides = json.loads(json_match.group())
                for i, slide in enumerate(slides):
                    slide["slide_number"] = i + 1
                return slides[:10]
        except Exception as e:
            logger.error(f"LLM slide generation error: {str(e)}")

        return self._create_default_slides(topic)
    
    def _extract_dynamic_slides(self, main_topic: str, results: List[Dict]) -> List[Dict[str, Any]]:
        """Extract slide data from research results."""
        slides = []
        used_topics = set()
        all_content = "\n\n".join([r["content"] for r in results])
        
        extraction_methods = [
            self._extract_what_is_slide,
            self._extract_key_benefits_slide,
            self._extract_how_it_works_slide,
            self._extract_real_examples_slide,
            self._extract_challenges_slide,
            self._extract_tools_slide,
            self._extract_best_practices_slide,
            self._extract_statistics_slide,
            self._extract_future_slide,
            self._extract_getting_started_slide
        ]
        
        for method in extraction_methods:
            if len(slides) >= 10:
                break
            slide = method(main_topic, all_content, results)
            if slide and slide["title"] not in used_topics:
                used_topics.add(slide["title"])
                slide["slide_number"] = len(slides) + 1
                slides.append(slide)
        
        # Fill remaining with defaults
        default_slides = self._create_default_slides(main_topic)
        for ds in default_slides:
            if len(slides) >= 10:
                break
            if ds["title"] not in used_topics:
                used_topics.add(ds["title"])
                ds["slide_number"] = len(slides) + 1
                slides.append(ds)
        
        return slides[:10]
    
    def _create_slide(self, title: str, content: Any, source: str = "") -> Dict[str, Any]:
        return {"title": title, "content": content if isinstance(content, list) else [content], "source": source}
    
    def _extract_what_is_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        sentences = re.findall(r'([A-Z][^\.]{50,200}\.)', content)
        for sent in sentences[:5]:
            clean = self._clean_text(sent)
            if len(clean) > 40:
                return self._create_slide(f"Understanding {topic}", clean)
        return self._create_slide(f"What is {topic}?", f"{topic} represents a transformative approach revolutionizing how we work and create.")
    
    def _extract_key_benefits_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        benefits = []
        matches = re.findall(r'(?:\d+\.?\s*|-\s*|•\s*)([^\.\n]{20,80})', content)
        for m in matches:
            clean = self._clean_text(m)
            if clean and len(clean) > 15 and clean not in benefits:
                benefits.append(clean)
            if len(benefits) >= 5:
                break
        if benefits:
            return self._create_slide(f"Key Benefits of {topic}", benefits[:5])
        return self._create_slide(f"Why {topic} Matters", ["Increased efficiency", "Cost reduction", "Better outcomes", "Competitive advantage", "Future-ready skills"])
    
    def _extract_how_it_works_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        sentences = re.findall(r'([A-Z][^\.]{50,200}\.)', content)
        for sent in sentences:
            lower = sent.lower()
            if any(word in lower for word in ['works', 'process', 'function', 'operates', 'method']):
                clean = self._clean_text(sent)
                if len(clean) > 40:
                    return self._create_slide(f"How {topic} Works", clean)
        return self._create_slide(f"The Process Behind {topic}", f"{topic} leverages advanced technologies to streamline workflows and deliver superior results.")
    
    def _extract_real_examples_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        examples = []
        matches = re.findall(r'(?:example|case study|company|organization)[^\.]{40,150}', content, re.IGNORECASE)
        for m in matches[:3]:
            clean = self._clean_text(m)
            if clean and len(clean) > 30 and clean not in examples:
                examples.append(clean)
        if examples:
            return self._create_slide(f"Real-World {topic} Success Stories", examples)
        return self._create_slide(f"{topic} in Action", f"Leading organizations across industries are leveraging {topic} to achieve remarkable productivity gains.")
    
    def _extract_challenges_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        challenges = []
        matches = re.findall(r'(?:challenge|problem|issue|mistake|avoid)[^\.]{25,100}', content, re.IGNORECASE)
        for m in matches[:5]:
            clean = self._clean_text(m)
            if clean and len(clean) > 20 and clean not in challenges:
                challenges.append(clean)
        if challenges:
            return self._create_slide(f"Challenges to Avoid in {topic}", challenges)
        return self._create_slide(f"Common {topic} Pitfalls", ["Lack of proper planning", "Insufficient training", "Ignoring best practices", "Poor implementation", "Unrealistic expectations"])
    
    def _extract_tools_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        tools = []
        matches = re.findall(r'(?:tool|platform|software|app)[^\.]{20,80}', content, re.IGNORECASE)
        for m in matches[:5]:
            clean = self._clean_text(m)
            if clean and len(clean) > 15 and clean not in tools and not clean.lower().startswith("the"):
                tools.append(clean)
        if tools:
            return self._create_slide(f"Essential {topic} Tools", tools)
        return self._create_slide(f"Top {topic} Platforms", ["AI-Powered Solutions", "Cloud-Based Platforms", "Enterprise Tools", "Open-Source Options", "Integrated Suites"])
    
    def _extract_best_practices_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        tips = []
        matches = re.findall(r'(?:best practice|tip|recommend|strategy|advice)[^\.]{25,100}', content, re.IGNORECASE)
        for m in matches[:4]:
            clean = self._clean_text(m)
            if clean and len(clean) > 20 and clean not in tips:
                tips.append(clean)
        if tips:
            return self._create_slide(f"{topic} Best Practices", tips)
        return self._create_slide(f"Pro Tips for {topic} Success", ["Start with clear objectives", "Invest in training", "Monitor results", "Iterate continuously"])
    
    def _extract_statistics_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        stats = []
        contexts = re.findall(r'[^\.]{0,50}\d+%[^\.]{0,50}|[^\.]{0,50}\d+x[^\.]{0,50}|[^\.]{0,50}percent[^\.]{0,50}', content, re.IGNORECASE)
        for ctx in contexts[:3]:
            clean = self._clean_text(ctx)
            if clean and len(clean) > 30:
                stats.append(clean)
        if stats:
            return self._create_slide(f"{topic} by the Numbers", stats)
        return self._create_slide(f"{topic} Impact Statistics", ["85% report improved efficiency", "3x faster completion", "60% cost reduction", "90% satisfaction rate"])
    
    def _extract_future_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        sentences = re.findall(r'([A-Z][^\.]{50,200}\.)', content)
        for sent in sentences:
            lower = sent.lower()
            if any(word in lower for word in ['future', 'predict', 'upcoming', 'trend', 'next']):
                clean = self._clean_text(sent)
                if len(clean) > 50:
                    return self._create_slide(f"The Future of {topic}", clean)
        return self._create_slide(f"What's Next for {topic}?", f"{topic} is rapidly evolving with AI integration, becoming the industry standard and transforming businesses globally.")
    
    def _extract_getting_started_slide(self, topic: str, content: str, results: List[Dict]) -> Optional[Dict]:
        steps = []
        matches = re.findall(r'(?:get started|begin|first step|starting)[^\.]{20,80}', content, re.IGNORECASE)
        for m in matches[:4]:
            clean = self._clean_text(m)
            if clean and len(clean) > 15 and clean not in steps:
                steps.append(clean)
        if steps:
            return self._create_slide(f"Getting Started with {topic}", steps)
        return self._create_slide(f"Your {topic} Journey Starts Here", ["Assess your current needs", "Choose the right solution", "Start with a pilot project", "Scale based on results"])
    
    def _create_default_slides(self, topic: str) -> List[Dict]:
        return [
            self._create_slide(f"Understanding {topic}", f"{topic} represents a paradigm shift in how we approach modern challenges, combining cutting-edge technology with intuitive workflows."),
            self._create_slide(f"Why {topic} is Game-Changing", ["Dramatic efficiency improvements", "Significant cost savings", "Enhanced user experiences", "Scalable solutions", "Competitive differentiation"]),
            self._create_slide(f"The {topic} Ecosystem", "A comprehensive network of tools, platforms, and methodologies working together seamlessly."),
            self._create_slide(f"{topic} Implementation Roadmap", ["Assessment and planning", "Tool selection", "Team training", "Pilot deployment", "Full-scale rollout"]),
            self._create_slide(f"Measuring {topic} Success", ["Track key performance indicators", "Monitor user adoption rates", "Calculate ROI metrics", "Gather feedback", "Optimize continuously"])
        ]
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[-•\d.\)\s]+', '', text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text[:150] if len(text) > 150 else text
    
    def _is_english(self, text: str) -> bool:
        if not text:
            return True
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.7 if text else True
    
    def generate_ppt_prompts(self, slides_data: List[Dict[str, Any]], main_topic: str = "") -> List[Dict[str, str]]:
        """Generate image prompts for clean infographic-style backgrounds (text will be added programmatically)."""
        prompts = []
        
        # Clean, professional backgrounds suitable for text overlay
        # These are designed to have good contrast and space for text
        clean_backgrounds = [
            "professional presentation slide background, deep blue to purple gradient, subtle geometric patterns, corporate style, minimalist, clean layout with empty space for text, high quality, 4K",
            "modern business presentation background, dark teal to cyan gradient, soft abstract shapes, professional infographic style, ample white space for content, 4K quality",
            "sleek corporate slide background, navy blue with subtle orange accents, minimalist design, clean geometric elements, plenty of room for text overlay, professional 4K",
            "professional infographic background, dark charcoal to slate gray gradient, subtle tech patterns, modern business style, generous empty space for text, 4K",
            "modern presentation slide, deep purple to magenta gradient, soft glowing elements, corporate professional style, clean layout with space for content, 4K quality",
            "business presentation background, dark blue with gold accents, elegant minimal design, subtle luxury patterns, ample space for text overlay, professional 4K",
            "professional slide background, deep green to teal gradient, nature-tech fusion, clean minimalist style, plenty of empty space for text, 4K quality",
            "corporate presentation background, dark red to burgundy gradient, subtle geometric textures, professional style, clean layout with room for content, 4K",
            "modern business slide, dark gray to silver gradient, sleek tech aesthetic, minimalist professional design, generous space for text overlay, 4K",
            "professional infographic background, deep indigo to violet gradient, subtle abstract patterns, corporate style, clean with ample text space, 4K quality"
        ]
        
        # Subtle visual elements that don't interfere with text
        subtle_visuals = {
            "AI": "subtle neural network patterns in corners",
            "artificial intelligence": "subtle neural network patterns in corners",
            "coding": "faint code symbols as watermark",
            "programming": "faint code symbols as watermark",
            "trends": "subtle upward arrow motifs",
            "business": "minimalist graph line accents",
            "technology": "subtle circuit board texture",
            "future": "faint forward arrow watermarks",
            "benefits": "subtle checkmark icons in corners",
            "challenges": "minimalist warning icon accents",
            "tools": "subtle gear icons as decoration",
            "statistics": "faint chart grid pattern",
            "getting started": "subtle roadmap line accents",
            "implementation": "minimalist process flow lines"
        }
        
        for i, slide in enumerate(slides_data):
            title = slide["title"]
            content = slide["content"]
            background = clean_backgrounds[i % len(clean_backgrounds)]
            num = slide.get("slide_number", i + 1)
            
            # Determine subtle visual elements
            title_lower = (title + " " + main_topic).lower()
            extra_visuals = ""
            
            for keyword, visuals in subtle_visuals.items():
                if keyword in title_lower:
                    extra_visuals = f", {visuals}"
                    break
            
            # Create prompt with slide content context for relevant imagery
            content_preview = ""
            if isinstance(content, list) and content:
                content_preview = " Related to: " + ", ".join([str(c)[:50] for c in content[:2]])
            elif isinstance(content, str):
                content_preview = f" Related to: {content[:100]}"
            
            prompt = (
                f"Professional presentation slide background for '{title}'.{content_preview}. "
                f"Visual theme: {background}{extra_visuals}. "
                f"Design should complement the topic with relevant imagery and icons. "
                f"Clean professional layout with space for text overlay. "
                f"Corporate infographic style, minimalist, 4K quality, 16:9 format"
            )
            
            prompts.append({
                "slide_number": num,
                "title": title,
                "content": content,
                "prompt": prompt
            })
        
        return prompts
    
    def regenerate_prompt_variation(self, original_prompt: str, variation_type: str = "minimal") -> str:
        """Generate a variation of the image prompt with different color scheme."""
        color_swaps = [
            ("deep blue", "vibrant purple"),
            ("purple", "electric blue"),
            ("teal", "emerald green"),
            ("cyan", "amber orange"),
            ("navy blue", "ruby red"),
            ("charcoal", "midnight blue"),
            ("slate gray", "forest green"),
            ("magenta", "coral orange"),
            ("green", "turquoise"),
            ("red", "sapphire blue"),
            ("indigo", "rose pink"),
            ("violet", "gold")
        ]
        
        new_prompt = original_prompt
        for old, new in color_swaps:
            if old in new_prompt.lower():
                new_prompt = new_prompt.replace(old, new, 1)  # Replace only first occurrence
                break
        
        # Add visual enhancements
        enhancements = [
            ", enhanced lighting, vivid saturated colors, glow effects",
            ", dramatic cinematic shadows, high contrast, depth",
            ", polished glossy finish, premium luxury aesthetic, reflections",
            ", volumetric lighting, lens flares, atmospheric effects",
            ", sharper details, enhanced textures, crystal clear 4K"
        ]
        new_prompt += random.choice(enhancements)
        return new_prompt


class UniversalImageGenerator:
    """Universal image generator supporting multiple providers."""
    
    def __init__(self, provider: str, api_key: str, model: str = None, **kwargs):
        self.provider = provider
        self.api_key = api_key
        self.model = model or IMAGE_PROVIDERS[provider]["default_model"]
        self.config = IMAGE_PROVIDERS[provider]
        self.extra_config = kwargs
        self.max_retries = 3
        self.timeout = 300
    
    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> Optional[str]:
        """Generate image using the selected provider. Returns base64 string."""
        try:
            if self.provider == "AzureFLUX":
                return self._generate_azure_flux(prompt, width, height)
            elif self.provider == "OpenAI":
                return self._generate_openai(prompt, width, height)
            elif self.provider == "StabilityAI":
                return self._generate_stability(prompt, width, height)
            else:
                raise ValueError(f"Unknown image provider: {self.provider}")
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None
    
    def _generate_azure_flux(self, prompt: str, width: int, height: int) -> Optional[str]:
        """Generate image using Azure FLUX."""
        endpoint = self.extra_config.get("endpoint", self.config.get("endpoint", ""))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "n": 1,
            "model": self.model
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{endpoint}?api-version=preview",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    if 'data' in result and len(result['data']) > 0:
                        return result['data'][0].get('b64_json')
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 10)
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 15)
            except Exception as e:
                logger.error(f"Azure FLUX error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(10)
        return None
    
    def _generate_openai(self, prompt: str, width: int, height: int) -> Optional[str]:
        """Generate image using OpenAI DALL-E."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # DALL-E has specific size requirements
        size_map = {
            (1024, 1024): "1024x1024",
            (1024, 1792): "1024x1792",
            (1792, 1024): "1792x1024"
        }
        size = size_map.get((width, height), "1024x1024")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    if 'data' in result and len(result['data']) > 0:
                        return result['data'][0].get('b64_json')
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 10)
            except Exception as e:
                logger.error(f"OpenAI DALL-E error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(10)
        return None
    
    def _generate_stability(self, prompt: str, width: int, height: int) -> Optional[str]:
        """Generate image using Stability AI."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": height,
            "width": width,
            "samples": 1,
            "steps": 30
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.config['base_url']}/generation/{self.model}/text-to-image",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    if 'artifacts' in result and len(result['artifacts']) > 0:
                        return result['artifacts'][0].get('base64')
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 10)
            except Exception as e:
                logger.error(f"Stability AI error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(10)
        return None


# Legacy class for backward compatibility
class ImageGenerator(UniversalImageGenerator):
    """Legacy image generator - redirects to UniversalImageGenerator.
    NOTE: This class requires API key to be provided via environment variable or user input.
    """
    
    def __init__(self):
        # Get API key from environment variable
        api_key = os.getenv("AZURE_FLUX_API_KEY")
        if not api_key:
            raise ValueError("AZURE_FLUX_API_KEY environment variable not set. Please configure your API key.")
        
        super().__init__(
            provider="AzureFLUX",
            api_key=api_key,
            model="FLUX.2-pro",
            endpoint="https://chikoai.services.ai.azure.com/providers/blackforestlabs/v1/flux-2-pro"
        )


def get_recent_logs(limit: int = 5) -> List[str]:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(log_dir):
        return []
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files[:limit]


def read_log_file(log_path: str) -> str:
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"
