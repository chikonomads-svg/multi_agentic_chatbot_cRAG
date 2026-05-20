"""
Streamlit UI for Imageprompt_app.
2026 Trending Content with Regenerate/Finalize Prompt buttons and multi-provider support.
Users can select their own LLM and Image models with their own API keys.
"""

import streamlit as st
import os
import sys
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logic import (
    ImagePromptResearcher, UniversalImageGenerator, LLMClient,
    LLM_PROVIDERS, IMAGE_PROVIDERS, get_recent_logs, read_log_file
)
from image_text_overlay import add_text_to_slide

st.set_page_config(
    page_title="2026 Trending PPT Creator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #FF6B6B; text-align: center; }
    .sub-header { font-size: 1.1rem; color: #4ECDC4; text-align: center; margin-bottom: 2rem; }
    .topic-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; margin: 10px 0; color: white; }
    .slide-card { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; padding: 20px; margin: 15px 0; color: white; }
    .content-box { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; }
    .prompt-box { background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px; border-left: 4px solid #FFD700; }
    .image-container { background: linear-gradient(135deg, #2C3E50 0%, #4CA1AF 100%); border-radius: 15px; padding: 20px; text-align: center; }
    .finalized { background: rgba(76, 175, 80, 0.2); border: 2px solid #4CAF50; }
    .settings-card { background: linear-gradient(135deg, #2d3436 0%, #636e72 100%); border-radius: 10px; padding: 15px; margin: 10px 0; }
    .provider-badge { background: #0984e3; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; }
    .status-badge { background: #00b894; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; }
    
    /* Workflow Step Styles */
    .workflow-step {
        display: flex;
        align-items: center;
        padding: 8px 0;
        font-size: 0.9rem;
    }
    .workflow-step.completed {
        color: #00b894;
        font-weight: 500;
    }
    .workflow-step.current {
        color: #0984e3;
        font-weight: 600;
    }
    .workflow-step.pending {
        color: #636e72;
    }
    .step-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 12px;
    }
    .step-icon.completed {
        background: #00b894;
        color: white;
    }
    .step-icon.current {
        background: #0984e3;
        color: white;
        animation: pulse 2s infinite;
    }
    .step-icon.pending {
        background: #dfe6e9;
        color: #636e72;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(9, 132, 227, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(9, 132, 227, 0); }
        100% { box-shadow: 0 0 0 0 rgba(9, 132, 227, 0); }
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Core app state
        'researcher': None,
        'image_generator': None,
        'llm_client': None,
        'trending_topics': [],
        'selected_topic': None,
        'slides_data': [],
        'ppt_prompts': [],
        'edited_prompts': {},
        'finalized_prompts': {},
        'generated_images': {},
        'current_step': 1,
        
        # LLM Settings
        'llm_provider': "OpenAI",
        'llm_model': "gpt-4o-mini",
        'llm_api_key': "",
        'azure_endpoint': "",
        'llm_configured': False,
        
        # Image Settings
        'image_provider': "AzureFLUX",
        'image_model': "FLUX.2-pro",
        'image_api_key': "",
        'image_endpoint': "",
        'image_configured': False,
        
        # Tavily Settings (optional)
        'tavily_api_key': "",
        'use_tavily': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_header():
    """Render the main header."""
    st.markdown('<p class="main-header">🎨 2026 Trending PPT Creator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Latest 2026 Trends → Edit/Regenerate Prompts → Finalize → Generate Images</p>', unsafe_allow_html=True)


def render_model_settings_sidebar():
    """Render the model settings in sidebar with scrollable sections."""
    with st.sidebar:
        st.markdown("## ⚙️ Model Settings")
        st.markdown("---")
        
        # ==================== LLM SETTINGS ====================
        st.markdown("### 🤖 Text Generation (LLM)")
        
        with st.container():
            # LLM Provider Selection
            llm_provider = st.selectbox(
                "Select LLM Provider:",
                options=list(LLM_PROVIDERS.keys()),
                format_func=lambda x: LLM_PROVIDERS[x]["name"],
                index=list(LLM_PROVIDERS.keys()).index(st.session_state.llm_provider),
                key="llm_provider_select"
            )
            st.session_state.llm_provider = llm_provider
            
            # LLM Model Selection
            available_models = LLM_PROVIDERS[llm_provider]["models"]
            current_model = st.session_state.llm_model if st.session_state.llm_model in available_models else available_models[0]
            
            llm_model = st.selectbox(
                "Select Model:",
                options=available_models,
                index=available_models.index(current_model) if current_model in available_models else 0,
                key="llm_model_select"
            )
            st.session_state.llm_model = llm_model
            
            # API Key Input
            key_name = LLM_PROVIDERS[llm_provider]["key_name"]
            llm_api_key = st.text_input(
                f"Enter {key_name}:",
                type="password",
                value=st.session_state.llm_api_key,
                placeholder=f"sk-... or your {key_name}",
                help=f"Your API key is stored only in session memory and never saved to disk.",
                key="llm_api_key_input"
            )
            st.session_state.llm_api_key = llm_api_key
            
            # Azure Endpoint (only for Azure OpenAI)
            if llm_provider == "AzureOpenAI":
                azure_endpoint = st.text_input(
                    "Azure Endpoint URL:",
                    value=st.session_state.azure_endpoint,
                    placeholder="https://your-resource.openai.azure.com",
                    key="azure_endpoint_input"
                )
                st.session_state.azure_endpoint = azure_endpoint
            
            # Test/Configure Button
            col1, col2 = st.columns([3, 2])
            with col1:
                if st.button("🔧 Configure LLM", use_container_width=True, type="primary"):
                    if not llm_api_key:
                        st.error("⚠️ Please enter an API key")
                    elif llm_provider == "AzureOpenAI" and not st.session_state.azure_endpoint:
                        st.error("⚠️ Please enter Azure endpoint")
                    else:
                        try:
                            # Initialize LLM Client
                            extra_config = {}
                            if llm_provider == "AzureOpenAI":
                                extra_config["azure_endpoint"] = st.session_state.azure_endpoint
                            
                            st.session_state.llm_client = LLMClient(
                                provider=llm_provider,
                                api_key=llm_api_key,
                                model=llm_model,
                                **extra_config
                            )
                            st.session_state.llm_configured = True
                            
                            # Also initialize/update researcher with new LLM client
                            if st.session_state.researcher is None:
                                st.session_state.researcher = ImagePromptResearcher(
                                    llm_client=st.session_state.llm_client
                                )
                            else:
                                st.session_state.researcher.set_llm_client(st.session_state.llm_client)
                            
                            st.success("✅ LLM configured!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.session_state.llm_configured:
                    st.markdown('<span class="status-badge">Active</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ==================== IMAGE GENERATION SETTINGS ====================
        st.markdown("### 🎨 Image Generation")
        
        with st.container():
            # Image Provider Selection
            image_provider = st.selectbox(
                "Select Image Provider:",
                options=list(IMAGE_PROVIDERS.keys()),
                format_func=lambda x: IMAGE_PROVIDERS[x]["name"],
                index=list(IMAGE_PROVIDERS.keys()).index(st.session_state.image_provider),
                key="image_provider_select"
            )
            st.session_state.image_provider = image_provider
            
            # Image Model Selection
            available_image_models = IMAGE_PROVIDERS[image_provider]["models"]
            current_image_model = st.session_state.image_model if st.session_state.image_model in available_image_models else available_image_models[0]
            
            image_model = st.selectbox(
                "Select Image Model:",
                options=available_image_models,
                index=available_image_models.index(current_image_model) if current_image_model in available_image_models else 0,
                key="image_model_select"
            )
            st.session_state.image_model = image_model
            
            # API Key Input
            image_key_name = IMAGE_PROVIDERS[image_provider]["key_name"]
            image_api_key = st.text_input(
                f"Enter {image_key_name}:",
                type="password",
                value=st.session_state.image_api_key,
                placeholder="Your image API key",
                help=f"Your API key is stored only in session memory.",
                key="image_api_key_input"
            )
            st.session_state.image_api_key = image_api_key
            
            # Azure Endpoint for FLUX (if needed)
            if image_provider == "AzureFLUX":
                image_endpoint = st.text_input(
                    "Azure FLUX Endpoint (optional):",
                    value=st.session_state.image_endpoint,
                    placeholder="Leave empty to use default",
                    key="image_endpoint_input"
                )
                st.session_state.image_endpoint = image_endpoint
            
            # Configure Button
            col1, col2 = st.columns([3, 2])
            with col1:
                if st.button("🔧 Configure Image Model", use_container_width=True, type="primary"):
                    if not image_api_key:
                        st.error("⚠️ Please enter an API key")
                    else:
                        try:
                            extra_config = {}
                            if image_provider == "AzureFLUX" and st.session_state.image_endpoint:
                                extra_config["endpoint"] = st.session_state.image_endpoint
                            
                            st.session_state.image_generator = UniversalImageGenerator(
                                provider=image_provider,
                                api_key=image_api_key,
                                model=image_model,
                                **extra_config
                            )
                            st.session_state.image_configured = True
                            st.success("✅ Image model configured!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.session_state.image_configured:
                    st.markdown('<span class="status-badge">Active</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ==================== TAVILY SETTINGS (Optional) ====================
        with st.expander("🔍 Tavily Search (Optional)", expanded=False):
            st.markdown("Tavily provides better web search results for research.")
            
            tavily_key = st.text_input(
                "Tavily API Key (optional):",
                type="password",
                value=st.session_state.tavily_api_key,
                placeholder="tvly-...",
                key="tavily_key_input"
            )
            st.session_state.tavily_api_key = tavily_key
            
            if tavily_key and st.button("Save Tavily Key"):
                os.environ["TAVILY_API_KEY"] = tavily_key
                # Reinitialize researcher with Tavily
                if st.session_state.researcher:
                    st.session_state.researcher.tavily_key = tavily_key
                    st.session_state.researcher.tavily_client = None  # Will be recreated on next use
                st.success("Tavily key saved!")
        
        st.markdown("---")
        
        # ==================== WORKFLOW INFO ====================
        st.markdown("### 📋 Workflow Progress")
        
        # Define workflow steps
        workflow_steps = [
            ("Configure Models", st.session_state.llm_configured and st.session_state.image_configured, 1),
            ("Research 2026 Trends", st.session_state.current_step >= 2, 2),
            ("Select Topic", st.session_state.selected_topic is not None, 3),
            ("Generate Slides", st.session_state.current_step >= 3 and len(st.session_state.ppt_prompts) > 0, 4),
            ("Edit Prompts", st.session_state.current_step >= 3 and len(st.session_state.edited_prompts) > 0, 5),
            ("Finalize Prompts", len(st.session_state.finalized_prompts) > 0, 6),
            ("Generate Images", len(st.session_state.generated_images) > 0, 7),
            ("Download PNGs", len(st.session_state.generated_images) == 10, 8),
        ]
        
        # Determine current active step
        current_step_num = st.session_state.current_step
        if st.session_state.selected_topic and st.session_state.current_step == 3:
            if len(st.session_state.generated_images) == 10:
                current_step_num = 8  # All done
            elif len(st.session_state.generated_images) > 0:
                current_step_num = 7  # Generating images
            elif len(st.session_state.finalized_prompts) > 0:
                current_step_num = 6  # Finalizing
            elif len(st.session_state.edited_prompts) > 0:
                current_step_num = 5  # Editing
        
        # Render workflow steps
        for step_name, is_completed, step_num in workflow_steps:
            if is_completed:
                status_class = "completed"
                icon = "✓"
                icon_class = "completed"
            elif step_num == current_step_num or (step_num == 3 and st.session_state.current_step == 2 and st.session_state.trending_topics):
                status_class = "current"
                icon = "●"
                icon_class = "current"
            else:
                status_class = "pending"
                icon = "○"
                icon_class = "pending"
            
            st.markdown(f"""
            <div class="workflow-step {status_class}">
                <span class="step-icon {icon_class}">{icon}</span>
                <span>{step_name}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Status Summary
        st.markdown("---")
        st.markdown("### 📊 Status")
        finalized_count = len(st.session_state.finalized_prompts)
        generated_count = len(st.session_state.generated_images)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Finalized:** {finalized_count}/10")
        with col2:
            st.markdown(f"**Generated:** {generated_count}/10")
        
        if st.session_state.llm_configured:
            st.markdown(f"**LLM:** <span class='provider-badge'>{LLM_PROVIDERS[st.session_state.llm_provider]['name']}</span>", unsafe_allow_html=True)
        if st.session_state.image_configured:
            st.markdown(f"**Image:** <span class='provider-badge'>{IMAGE_PROVIDERS[st.session_state.image_provider]['name']}</span>", unsafe_allow_html=True)


def step_1_topic_input():
    """Step 1: Topic input and research."""
    st.markdown("### 📌 Step 1: Research Latest 2026 Trends")
    
    # Check if LLM is configured
    if not st.session_state.llm_configured:
        st.warning("⚠️ Please configure an LLM in the sidebar first (Model Settings → Text Generation)")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "What 2026 trending topic to research?",
            placeholder="e.g., AI Trends, Vibe Coding 2026...",
            key="topic_input"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("🔍 Research 2026", use_container_width=True) and topic:
            with st.spinner("Fetching latest 2026 trending information..."):
                try:
                    if st.session_state.researcher is None:
                        st.session_state.researcher = ImagePromptResearcher(
                            llm_client=st.session_state.llm_client
                        )
                    
                    st.session_state.trending_topics = st.session_state.researcher.research_trending_topics(topic)
                    st.session_state.current_step = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def step_2_select_topic():
    """Step 2: Select a trending topic."""
    st.markdown("### 📊 Step 2: Select a 2026 Trending Topic")
    st.write(f"Found **{len(st.session_state.trending_topics)}** latest 2026 trending topics:")
    
    cols = st.columns(2)
    for i, topic in enumerate(st.session_state.trending_topics):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="topic-card">
                <h4>🔥 {topic['title']}</h4>
                <p>{topic['description'][:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select & Create 📸", key=f"select_{topic['id']}"):
                st.session_state.selected_topic = topic
                with st.spinner(f"Researching 2026 data for '{topic['title']}'..."):
                    try:
                        st.session_state.slides_data = st.session_state.researcher.research_dynamic_slides(topic['title'])
                        st.session_state.ppt_prompts = st.session_state.researcher.generate_ppt_prompts(
                            st.session_state.slides_data, 
                            main_topic=topic['title']
                        )
                        st.session_state.edited_prompts = {p['slide_number']: p['prompt'] for p in st.session_state.ppt_prompts}
                        st.session_state.finalized_prompts = {}
                        st.session_state.current_step = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


def step_3_edit_and_generate():
    """Step 3: Edit prompts and generate images."""
    selected = st.session_state.selected_topic
    prompts = st.session_state.ppt_prompts
    
    st.markdown(f"### 📸 2026 PPT Carousel: {selected['title']}")
    
    # Check if image generator is configured
    if not st.session_state.image_configured:
        st.warning("⚠️ Please configure an Image Model in the sidebar first (Model Settings → Image Generation)")
    else:
        st.info("💡 **Note:** Images are generated with AI backgrounds + programmatically added readable text using PIL/Pillow.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("🔄 New Topic"):
            st.session_state.current_step = 1
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎨 Your 10 Dynamic PPT Slides (Edit → Regenerate → Finalize → Generate)")
    
    for prompt_data in prompts:
        slide_num = prompt_data['slide_number']
        content = prompt_data['content']
        is_finalized = slide_num in st.session_state.finalized_prompts
        
        with st.expander(f"📱 Slide {slide_num}: {prompt_data['title']}", expanded=(slide_num == 1)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show content points
                st.markdown(f"""
                <div class="slide-card">
                    <h3>Slide {slide_num}: {prompt_data['title']}</h3>
                    <div class="content-box">
                        <strong>📝 Content Points:</strong>
                """, unsafe_allow_html=True)
                
                if isinstance(content, list):
                    for item in content:
                        st.write(f"• {item}")
                else:
                    st.write(content)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Prompt editing section
                prompt_class = "prompt-box finalized" if is_finalized else "prompt-box"
                st.markdown(f'<div class="{prompt_class}">', unsafe_allow_html=True)
                st.markdown("**🎨 Image Generation Prompt:**")
                
                # Editable prompt
                edited_prompt = st.text_area(
                    f"Prompt for Slide {slide_num}",
                    value=st.session_state.edited_prompts.get(slide_num, prompt_data['prompt']),
                    height=120,
                    key=f"prompt_edit_{slide_num}",
                    disabled=is_finalized,
                    label_visibility="collapsed"
                )
                st.session_state.edited_prompts[slide_num] = edited_prompt
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Action buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if not is_finalized:
                        if st.button(f"🔄 Regenerate", key=f"regen_{slide_num}", help="Generate slight variation of this prompt"):
                            new_prompt = st.session_state.researcher.regenerate_prompt_variation(edited_prompt)
                            st.session_state.edited_prompts[slide_num] = new_prompt
                            st.rerun()
                
                with btn_col2:
                    if not is_finalized:
                        if st.button(f"✅ Finalize", key=f"finalize_{slide_num}", type="primary"):
                            st.session_state.finalized_prompts[slide_num] = edited_prompt
                            st.rerun()
                    else:
                        if st.button(f"✏️ Unfinalize", key=f"unfinalize_{slide_num}"):
                            del st.session_state.finalized_prompts[slide_num]
                            st.rerun()
                
                with btn_col3:
                    if is_finalized:
                        st.success("✅ Finalized")
            
            with col2:
                # Image display/generation section
                if slide_num in st.session_state.generated_images:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    img_data = base64.b64decode(st.session_state.generated_images[slide_num])
                    st.image(img_data, caption=f"Slide {slide_num}", use_container_width=True)
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_data,
                        file_name=f"slide_{slide_num:02d}_{selected['title'][:15].replace(' ', '_')}.png",
                        mime="image/png",
                        key=f"dl_{slide_num}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("🎨 No image yet")
                    if is_finalized:
                        if st.button(f"🎨 Generate Image", key=f"gen_{slide_num}", type="primary"):
                            generate_slide_image(slide_num)
                    else:
                        st.warning("Finalize prompt first")


def generate_slide_image(slide_num: int):
    """Generate image with background + programmatically added text."""
    if not st.session_state.image_configured:
        st.error("Please configure an image model first in the sidebar!")
        return
    
    # Get the slide data
    slide_data = None
    for prompt_data in st.session_state.ppt_prompts:
        if prompt_data['slide_number'] == slide_num:
            slide_data = prompt_data
            break
    
    if not slide_data:
        st.error(f"Slide {slide_num} data not found!")
        return
    
    prompt = st.session_state.finalized_prompts.get(
        slide_num, 
        st.session_state.edited_prompts.get(slide_num, "")
    )
    
    with st.spinner(f"🎨 Generating image for Slide {slide_num}... This may take 2-4 minutes"):
        try:
            # Step 1: Generate background image
            b64_background = st.session_state.image_generator.generate_image(
                prompt=prompt, 
                width=1024, 
                height=1024
            )
            
            if not b64_background:
                st.error(f"❌ Failed to generate background image. Please check your API key and try again.")
                return
            
            # Step 2: Add text overlay using PIL
            st.info(f"📝 Adding readable text to Slide {slide_num}...")
            
            final_image = add_text_to_slide(
                background_b64=b64_background,
                slide_title=slide_data['title'],
                content=slide_data['content'],
                slide_number=slide_num,
                total_slides=10
            )
            
            st.session_state.generated_images[slide_num] = final_image
            st.success(f"✅ Slide {slide_num} image with text generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


def main():
    """Main application entry point."""
    initialize_session_state()
    render_header()
    render_model_settings_sidebar()
    
    if st.session_state.current_step == 1:
        step_1_topic_input()
    elif st.session_state.current_step == 2:
        step_2_select_topic()
    elif st.session_state.current_step == 3:
        step_3_edit_and_generate()
    
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #666;">2026 Trending Research + Dynamic Slide Titles + Multi-Provider AI Support</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
