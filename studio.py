"""
Multimodal Image Generation Studio - Project 3 Starter Pipeline
DecodeLabs Industrial Training Kit

Implements the 6-stage blueprint:
1. Prompt Payload Formulation (aspect ratio -> exact pixel mapping)
2. Network API Gateway (split-timeout policy: connect=3.05s, read=60s)
3. Security & Moderation Gates (pre- and post-generation)
4. Transport Protocol (chunked, memory-safe binary streaming)
5. Integrity Verification (forced pixel-level decode)
6. Automated QA hook (stub - plug in CLIP-based scoring if desired)

Swap in your real provider's endpoint + auth in `call_image_api()`.
"""

import os
import time
import random
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the current directory and loads it into os.environ

# QA dependencies are heavy (torch + transformers) and optional - only needed
# if you actually call automated_qa(). Import lazily so the rest of the
# pipeline still works without them installed.
try:
    import torch
    import torch.nn as nn
    from transformers import CLIPModel, CLIPProcessor
    _QA_DEPS_AVAILABLE = True
except ImportError:
    _QA_DEPS_AVAILABLE = False

# ---------------------------------------------------------------------------
# STAGE 1: Prompt Payload Formulation
# ---------------------------------------------------------------------------

# Stability AI's v2beta "core" model accepts these aspect_ratio strings directly -
# no width/height conversion needed for this endpoint.
VALID_ASPECT_RATIOS = {"21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"}


def build_payload(prompt: str, aspect_ratio: str = "1:1",
                   negative_prompt: str = "", style_preset: str | None = None,
                   output_format: str = "png") -> dict:
    """
    Build the form fields for Stability AI's v2beta multipart/form-data request.
    NOTE: this endpoint wants form fields (data=), not a JSON body (json=).
    """
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValueError(
            f"Unsupported aspect ratio '{aspect_ratio}'. "
            f"Must be one of {sorted(VALID_ASPECT_RATIOS)} to avoid an API handshake failure."
        )

    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if style_preset:
        payload["style_preset"] = style_preset  # e.g. "cyberpunk", "minimalism"
    return payload


# ---------------------------------------------------------------------------
# STAGE 2 + 5: Network Gateway with split timeouts + exponential backoff/jitter
# ---------------------------------------------------------------------------

IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "pollinations").lower()

API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"
API_KEY = os.environ.get("IMAGE_API_KEY", "")

CONNECT_TIMEOUT = 3.05   # fail fast on dead/unreachable servers
READ_TIMEOUT = 60        # give the diffusion/denoise loop room to finish
MAX_RETRIES = 5
RETRYABLE_STATUS_CODES = {429, 503}


def map_aspect_ratio_to_pixels(aspect_ratio: str) -> tuple[int, int]:
    """
    Maps Stability AI aspect ratio strings to standard pixel dimensions (targeting ~1 megapixel).
    """
    mapping = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
        "4:5": (896, 1120),
        "5:4": (1120, 896),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    }
    return mapping.get(aspect_ratio, (1024, 1024))


def call_image_api(payload: dict) -> requests.Response:
    """
    Calls either Stability AI's v2beta image generation endpoint or Pollinations.ai API
    based on the configured IMAGE_PROVIDER.
    """
    if IMAGE_PROVIDER == "pollinations":
        import urllib.parse
        
        prompt = payload.get("prompt", "")
        aspect_ratio = payload.get("aspect_ratio", "1:1")
        width, height = map_aspect_ratio_to_pixels(aspect_ratio)
        
        # Build Pollinations GET URL
        encoded_prompt = urllib.parse.quote(prompt)
        pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&enhance=false"
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    pollinations_url,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                    stream=True,
                )
            except requests.exceptions.ConnectTimeout:
                print(f"[NETWORK FAILURE] Could not establish connection to Pollinations (attempt {attempt}). Failing fast.")
                raise
            except requests.exceptions.ReadTimeout:
                print(f"[INFERENCE FAILURE] Pollinations server took too long to respond (attempt {attempt}).")
                if attempt == MAX_RETRIES:
                    raise
                _backoff_sleep(attempt)
                continue
            except requests.exceptions.ConnectionError as e:
                print(f"[NETWORK FAILURE] Connection dropped mid-request (attempt {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    raise
                _backoff_sleep(attempt)
                continue

            if response.status_code == 200:
                return response
            
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                print(f"[RETRYABLE {response.status_code}] Backing off before retry {attempt + 1}.")
                _backoff_sleep(attempt)
                continue
            
            # Non-retryable error
            raise RuntimeError(f"Pollinations API error {response.status_code}: {response.text}")
            
        raise RuntimeError("Exceeded max retries calling Pollinations API.")

    else:
        # Default to Stability AI
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "image/*",
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    data=payload,             # form fields, NOT json=
                    files={"none": ""},       # forces multipart/form-data encoding (no real file needed)
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                    stream=True,  # enables chunked reading later, see Stage 4
                )
            except requests.exceptions.ConnectTimeout:
                print(f"[NETWORK FAILURE] Could not establish connection (attempt {attempt}). Failing fast.")
                raise
            except requests.exceptions.ReadTimeout:
                print(f"[INFERENCE FAILURE] Server took too long to respond (attempt {attempt}).")
                # Inference failures are worth retrying since the server is alive, just slow.
                if attempt == MAX_RETRIES:
                    raise
                _backoff_sleep(attempt)
                continue
            except requests.exceptions.ConnectionError as e:
                # Covers connection-reset-by-peer and similar mid-request drops (e.g. WinError 10054).
                # These are usually transient network blips, not a server-side rejection, so retry.
                print(f"[NETWORK FAILURE] Connection dropped mid-request (attempt {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    raise
                _backoff_sleep(attempt)
                continue

            if response.status_code == 200:
                return response

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                print(f"[RETRYABLE {response.status_code}] Backing off before retry {attempt + 1}.")
                _backoff_sleep(attempt)
                continue

            # Non-retryable failure -> raise a descriptive error (Stage 3)
            raise_api_error(response)
            break

        raise RuntimeError("Exceeded max retries calling Stability API.")


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter so retries don't sync up into a bot-swarm."""
    base = 2 ** attempt
    jitter = random.uniform(0, 1)
    time.sleep(base + jitter)


# ---------------------------------------------------------------------------
# STAGE 3: Security & Moderation Gates
# ---------------------------------------------------------------------------

def raise_api_error(response: requests.Response) -> None:
    """
    Parse the API error and raise a descriptive RuntimeError to propagate to the frontend.
    """
    try:
        body = response.json()
    except ValueError:
        body = {}

    error_code = body.get("error", {}).get("code", "")

    if error_code in ("sentinel_block", "content_policy_violation"):
        raise RuntimeError("Your prompt was rejected by input moderation before generation (no cost incurred). Please revise the prompt.")
    elif error_code in ("moderation_blocked",) or body.get("finish_reason") == "FILTER":
        raise RuntimeError("The generated image was blocked by output moderation. Try adjusting the prompt.")
    elif response.status_code == 402:
        raise RuntimeError("Payment Required: You lack sufficient credits on Stability AI. Please purchase more credits or switch providers.")
    else:
        errors = body.get("errors", [])
        if errors:
            raise RuntimeError(f"Stability API Error {response.status_code}: {', '.join(errors)}")
        elif "message" in body:
            raise RuntimeError(f"Stability API Error {response.status_code}: {body['message']}")
        else:
            raise RuntimeError(f"Stability API Error {response.status_code}: {body or response.text}")


def _handle_api_error(response: requests.Response) -> None:
    """
    Deprecated fallback to print errors to stdout.
    """
    try:
        raise_api_error(response)
    except RuntimeError as e:
        print(e)


# ---------------------------------------------------------------------------
# STAGE 4: Memory-safe streaming to disk
# ---------------------------------------------------------------------------

def stream_to_disk(response: requests.Response, output_path: str,
                    chunk_size: int = 65536) -> None:
    """Never load the whole image into RAM at once - write it chunk by chunk."""
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


# ---------------------------------------------------------------------------
# STAGE 5: Integrity verification (catch silent truncation)
# ---------------------------------------------------------------------------

def verify_image_integrity(path: str) -> bool:
    """
    A basic header/CRC check (e.g. Image.verify()) can report success even if
    the connection dropped mid-download. Force a full pixel-level decode instead.
    """
    try:
        img = Image.open(path)
        img.load()  # forces full decode, pixel by pixel
        return True
    except OSError:
        print(f"[CORRUPTED ASSET] {path} failed full decode - discarding and should retry.")
        if os.path.exists(path):
            os.remove(path)
        return False


# ---------------------------------------------------------------------------
# STAGE 6: Automated QA - Aesthetic Classification (Lens 1) + Semantic
# Alignment Verification (Lens 2), per the kit's QA architecture.
# ---------------------------------------------------------------------------

AESTHETIC_THRESHOLD = 5.0     # score out of 10 - calibrated to the LAION aesthetic predictor's
                               # actual distribution, where 5-6 is a typical good photo and 7+ is
                               # reserved for genuinely striking/professional-grade images
ALIGNMENT_THRESHOLD = 20.0    # CLIP score is roughly 0-100; ~20+ indicates decent alignment

AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/"
    "refs/heads/main/sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_WEIGHTS_PATH = "aesthetic_predictor.pth"

_clip_model = None
_clip_processor = None
_aesthetic_head = None


if _QA_DEPS_AVAILABLE:
    class _AestheticHead(nn.Module):
        """Linear regressor trained on top of CLIP ViT-L/14 image embeddings
        to predict human aesthetic ratings (LAION aesthetic predictor)."""

        def __init__(self, input_size: int = 768):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.layers(x)


def _load_qa_models() -> None:
    """Lazily load CLIP + the aesthetic head only when QA is actually run."""
    global _clip_model, _clip_processor, _aesthetic_head
    if _clip_model is not None:
        return

    print("[QA] Loading CLIP ViT-L/14 (first run only - this downloads ~1.7GB)...")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    _clip_model.eval()

    if not os.path.exists(AESTHETIC_WEIGHTS_PATH):
        print("[QA] Downloading aesthetic predictor weights...")
        r = requests.get(AESTHETIC_WEIGHTS_URL, timeout=(3.05, 60))
        r.raise_for_status()
        with open(AESTHETIC_WEIGHTS_PATH, "wb") as f:
            f.write(r.content)

    head = _AestheticHead()
    head.load_state_dict(torch.load(AESTHETIC_WEIGHTS_PATH, map_location="cpu"))
    head.eval()
    _aesthetic_head = head


def _extract_embedding(output):
    """
    transformers <5.0: get_image_features()/get_text_features() return a plain tensor.
    transformers >=5.0: they return a BaseModelOutputWithPooling object whose
    .pooler_output holds the actual projected embedding tensor.
    This handles both so the QA code doesn't break across versions.
    """
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def automated_qa(path: str, prompt: str) -> bool:
    """
    Lens 1 (Aesthetic Classification): CLIP image embedding -> linear
    regressor -> score out of 10. Below AESTHETIC_THRESHOLD is discarded.

    Lens 2 (Semantic Alignment): cosine similarity between the image and
    prompt CLIP embeddings ("CLIP score", roughly 0-100). Below
    ALIGNMENT_THRESHOLD flags the asset as hallucinated/divergent.
    """
    if not _QA_DEPS_AVAILABLE:
        print("[QA] torch/transformers not installed - skipping QA, treating as pass. "
              "Run: pip install torch transformers")
        return True

    try:
        _load_qa_models()
    except Exception as e:
        print(f"[QA] Could not load QA models ({e}). Skipping QA, treating as pass.")
        return True

    image = Image.open(path).convert("RGB")

    with torch.no_grad():
        inputs = _clip_processor(images=image, text=[prompt], return_tensors="pt", padding=True)
        image_features = _extract_embedding(
            _clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        )
        text_features = _extract_embedding(
            _clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        )

        image_features_n = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_n = text_features / text_features.norm(dim=-1, keepdim=True)

        aesthetic_score = _aesthetic_head(image_features_n).item()
        clip_score = (image_features_n @ text_features_n.T).item() * 100

    print(f"[QA] Aesthetic score: {aesthetic_score:.2f}/10  |  "
          f"Semantic alignment (CLIP score): {clip_score:.1f}/100")

    if aesthetic_score < AESTHETIC_THRESHOLD:
        print(f"[QA] REJECTED - aesthetic score below threshold ({AESTHETIC_THRESHOLD}).")
        return False
    if clip_score < ALIGNMENT_THRESHOLD:
        print(f"[QA] REJECTED - semantic alignment below threshold ({ALIGNMENT_THRESHOLD}).")
        return False

    return True


# ---------------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------------

def generate_image(prompt: str, aspect_ratio: str = "1:1",
                    output_path: str = "output.png",
                    negative_prompt: str = "",
                    style_preset: str | None = None) -> str | None:
    payload = build_payload(prompt, aspect_ratio=aspect_ratio,
                             negative_prompt=negative_prompt, style_preset=style_preset)

    response = call_image_api(payload)
    stream_to_disk(response, output_path)

    if not verify_image_integrity(output_path):
        return None

    if not automated_qa(output_path, prompt):
        print("Asset failed automated QA thresholds.")
        return None

    print(f"Image generated and verified: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage - replace API_URL/API_KEY above with your real provider first.
    generate_image(
        prompt="A serene mountain lake at sunrise, photorealistic",
        aspect_ratio="16:9",
        output_path="output.png",
    )