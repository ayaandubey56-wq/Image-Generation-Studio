"""
Multimodal Image Generation Studio - Front-End
DecodeLabs Industrial Training Kit, Project 3

A minimal interface on top of studio.py so a person can type a prompt and
see the generated image rendered, instead of running everything from a
terminal. Satisfies the "display cleanly to the user" requirement.

Run: streamlit run app.py
"""

import streamlit as st
from studio import generate_image, VALID_ASPECT_RATIOS

st.set_page_config(page_title="Image Generation Studio", page_icon="🖼️", layout="centered")

st.title("Image Generation Studio")
st.caption("Describe an image. The pipeline handles the API call, retries, "
           "moderation, streaming, and integrity checks behind the scenes.")

with st.form("generate_form"):
    prompt = st.text_area(
        "Prompt",
        height=100,
        placeholder="A serene mountain lake at sunrise, photorealistic",
    )

    col1, col2 = st.columns(2)
    with col1:
        ratios = sorted(VALID_ASPECT_RATIOS)
        aspect_ratio = st.selectbox("Aspect ratio", ratios, index=ratios.index("1:1"))
    with col2:
        style_preset = st.selectbox(
            "Style preset",
            ["", "photographic", "cyberpunk", "minimalism", "anime", "fantasy-art"],
        )

    negative_prompt = st.text_input("Negative prompt (optional)", placeholder="blurry, low quality")

    submitted = st.form_submit_button("Generate image", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        with st.spinner("Generating — this can take up to a minute..."):
            try:
                result_path = generate_image(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    output_path="output.png",
                    negative_prompt=negative_prompt,
                    style_preset=style_preset or None,
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                result_path = None

        if result_path:
            st.image(result_path, caption=prompt, use_container_width=True)
            with open(result_path, "rb") as f:
                st.download_button(
                    "Download image", f, file_name="generated_image.png",
                    mime="image/png", use_container_width=True,
                )
        else:
            st.error(
                "No image was produced. This usually means the prompt was "
                "blocked by moderation, or the asset failed the integrity/QA "
                "check — check the terminal running `streamlit run app.py` for details."
            )