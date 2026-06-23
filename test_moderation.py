"""
Confirms the security/moderation gate handling in studio.py (Stage 3) fires
correctly and surfaces a clean message instead of crashing the app.

This sends a prompt that's expected to be rejected by Stability's input
moderation (Gate 1: pre-generation). The goal is purely to test the
_handle_api_error() error path - it is NOT expected to produce an image.

Run: python test_moderation.py
"""

from studio import generate_image

TEST_PROMPT = "extremely graphic violence, gore, blood, dismemberment"

print("Sending a prompt expected to be blocked by input moderation...")
result = generate_image(prompt=TEST_PROMPT, aspect_ratio="1:1", output_path="test_blocked.png")

if result is None:
    print("PASS: the pipeline handled the rejection gracefully without crashing.")
else:
    print("Unexpected: an image was generated. The provider may not have flagged "
          "this prompt - try a different test phrase or check moderation settings.")