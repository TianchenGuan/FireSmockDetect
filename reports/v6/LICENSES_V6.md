# V6 — Image Generator License & Capability Research (Wildfire-Smoke Synthetic Data)

Date researched: 2026-10-20. All quotes below were pulled from the live primary sources on this date (HF repo files, HF API, aws.amazon.com legal pages, stability.ai pages). This is an engineering research summary, **not legal advice** — the RESTRICTED/UNCLEAR verdicts below flag where counsel/AWS should confirm before V6 relies on a generator.

Use cases evaluated:
- **(i)** Generate synthetic smoke training images.
- **(ii)** Train a YOLO student detector on those images and keep/release the student weights (class project / internal research now; the detection service **may later be used operationally**).
- **(iii)** Show generated samples in slides/demo.

Summary matrix (details + quotes per section):

| Generator | License | (i) gen. training images | (ii) train + release student | (iii) slides/demo |
|---|---|---|---|---|
| Qwen/Qwen-Image-2.1 | Qwen Research License Agreement (custom, non-commercial) | PERMITTED (research/eval only); RESTRICTED for commercial/operational | RESTRICTED (research-only + mandatory "Built with Qwen" attribution; commercial needs separate license) | PERMITTED (research context) |
| zai-org/GLM-Image | MIT (+ Apache-2.0 components) | PERMITTED | PERMITTED (unencumbered) | PERMITTED |
| Stability AI on Bedrock (SD3.5L, Ultra, Core, Image Services) | Stability Amazon Bedrock EULA + AWS Service Terms §50 | UNCLEAR (EULA permits; AWS §50.5 restricts on literal reading) | UNCLEAR (no Stability encumbrance; §50.5 question on generation step) | PERMITTED (don't strip provenance data) |
| amazon.nova-canvas (Bedrock) | AWS Service Terms §50 + AWS Responsible AI Policy | UNCLEAR (same §50.5) | UNCLEAR (outputs are Your Content; §50.5 question on generation step) | PERMITTED (invisible watermark/C2PA must stay) |

---

## 1. Qwen/Qwen-Image-2.1

| Item | Finding |
|---|---|
| License name | **Qwen RESEARCH LICENSE AGREEMENT** (Release Date: September 20, 2026), licensor Hangzhou Tongyi Laboratory Technology Co., Ltd. Confirmed in-repo `LICENSE` file; HF metadata `license: other`, `license_name: qwen-research`, `license_link: LICENSE`. The expectation "Qwen Research License" is **verified correct**. |
| Source | https://huggingface.co/Qwen/Qwen-Image-2.1 (README §License: "This model is licensed under the Qwen Research License Agreement") and https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE |
| Commercial use | §1.i: "'Non-Commercial' shall mean for research or evaluation purposes only." §2.a: rights granted "FOR NON-COMMERCIAL PURPOSES ONLY." §2.b: "You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us... model-business@notice.qwencloud.com." |
| Output ownership/usage | **Not addressed** — the license has no clause assigning or disclaiming ownership of generated outputs (§5.a covers only derivative works/modifications of the Materials, which you own). UNCLEAR. §5.c: licenses terminate if you sue alleging "the Materials or any output therefrom" infringe your IP. |
| Using outputs to train other models | **No prohibition.** Expressly contemplated with an attribution duty, §4.b: "If you use the Materials or any outputs or results therefrom to create, train, fine-tune, or improve an AI model that is distributed or made available, you shall prominently display 'Built with Qwen' or 'Improved using Qwen' in the related product documentation." |
| Attribution duties | §4.b above (applies to a released student detector); §3.c NOTICE-file attribution when redistributing the Materials/derivatives; §4.c: don't use "Qwen" as primary name of derivative products. |
| Student detector encumbered? | **Yes, encumbered**: (a) mandatory "Built with Qwen"/"Improved using Qwen" in the detector's documentation if distributed (§4.b); (b) the underlying grant is non-commercial only, so generating the training set for a detector destined for an operational (commercial) service arguably exceeds the grant — UNCLEAR whether the non-commercial limit reaches the already-trained student, but the safe reading is that operational use requires the separate commercial license. Governing law: PRC; exclusive jurisdiction Hangzhou courts (§8). |

**Verdicts:** (i) PERMITTED for research/evaluation only; RESTRICTED for any commercial purpose (incl. later operational service) without a commercial license. (ii) RESTRICTED — research release OK with §4.b attribution; commercial/operational path needs a separate license. (iii) PERMITTED in a research/class context.

**Size / capability / stack facts** (HF API `?blobs=true`, README):
- Weights: **33.1 GB** total safetensors (transformer 14.2 GB, Qwen text encoder 17.5 GB over 4 shards, VAE 1.35 GB), BF16. "7B parameters in its visual generation component (32 Single-Stream DiT layers)."
- **Native RGBA: YES** — "Generate regular or transparent (RGBA) images from text, edit transparent layers, and extract subjects from photographs—all in one model." Localized editing: "specify local edits via circles, painted annotations, or separate masks", up to 10 reference images. RGBA is triggered by prompt format ("This is an RGBA image with transparency...").
- Stack: `diffusers` **from GitHub main** (pipeline `QwenImage21Pipeline` not in a PyPI release yet), `torch>=2.4.0`, `transformers>=5.17`, `accelerate`, `pillow`. No `trust_remote_code` needed.
- VRAM: **no official figure published**; card only offers `pipe.enable_model_cpu_offload()` as memory optimization. BF16 weights total 33 GB, default 2048x2048 / 40 steps — expect >40 GB without offload (inference measurement needed; not stated by vendor).

---

## 2. zai-org/GLM-Image

| Item | Finding |
|---|---|
| License name | **MIT** for the model; **Apache-2.0** for two imported components. No LICENSE file exists in the repo (verified via HF API siblings list) — the grant is declared in README metadata (`license: mit`) and README §License. |
| Source | https://huggingface.co/zai-org/GLM-Image — README §License: "The overall GLM-Image model is released under the MIT License." / "This project incorporates the VQ tokenizer weights and VIT weights from X-Omni/X-Omni-En, which are licensed under the Apache License, Version 2.0. The VQ tokenizer and VIT weights remains subject to the original Apache-2.0 terms." |
| Commercial use | PERMITTED — MIT and Apache-2.0 both allow unrestricted commercial use. |
| Output ownership/usage | Not addressed (standard for MIT); **no restriction of any kind on outputs** or their use as training data. |
| Using outputs to train other models | No restriction. |
| Attribution duties | MIT/Apache notices must accompany redistribution **of the weights/code themselves**; nothing attaches to generated images or to models trained on them. |
| Student detector encumbered? | **No.** A YOLO detector trained on GLM-Image outputs carries no license obligations. |

**Verdicts:** (i) PERMITTED. (ii) PERMITTED, student weights unencumbered. (iii) PERMITTED.

**Size / capability / stack facts:**
- Weights: **35.8 GB** total safetensors (AR `vision_language_encoder` 20.2 GB — 9B AR generator initialized from GLM-4-9B-0414; DiT `transformer` 13.9 GB — 7B diffusion decoder; text encoder 0.87 GB; VAE 0.81 GB). Note: HF metadata says "7B params", which undercounts vs. the 9B+7B architecture described in the card.
- **RGBA: not documented** — no transparent/alpha generation claim anywhere on the card. Editing: YES — unified text-to-image and image-to-image ("image editing, style transfer, multi-subject consistency, identity-preserving generation"), multi-image conditioning; no explicit mask-based inpaint API (edit is instruction-driven). Output H/W must be divisible by 32.
- Stack: `diffusers` pipeline `GlmImagePipeline`; **both `transformers` and `diffusers` must be installed from GitHub source**; SGLang serving supported (`sglang serve`, `/v1/images/generations`, `/v1/images/edits`); vLLM-Omni "currently being integrated".
- VRAM: "You can set `enable_model_cpu_offload=True` to run it with `~23GB` of GPU memory, at the cost of slower inference." (README, verbatim). No figure given for non-offloaded inference.

---

## 3. Stability AI models on Amazon Bedrock (SD3.5 Large, Stable Image Ultra/Core, Image Services)

What is on Bedrock (verified at https://aws.amazon.com/bedrock/stability-ai/): **Stable Diffusion 3.5 Large**, **Stable Image Ultra**, **Stable Image Core** ("Affordable 2.6B parameter model"), plus **Stability AI Image Services** — "a set of thirteen specialized image editing tools": Edit tools (Remove Background, Erase Object, Search and Replace, Search and Recolor, **Inpaint and Outpaint**), Upscale tools (Fast/Conservative/Creative), Control tools (Structure, Sketch, Style Guide, Style Transfer). So the inpaint/erase/edit APIs the project wants do exist as managed Bedrock services.

Applicable terms stack (all reachable and read in full):
1. **AWS Customer Agreement + AWS Service Terms §50 / §50.12** (https://aws.amazon.com/service-terms/). §50.12.1: "Third-party models are available to you on Amazon Bedrock as 'Third-Party Content'. By using a third-party model, you agree to the applicable terms here [link]".
2. **STABILITY AMAZON BEDROCK END USER LICENSE AGREEMENT, Last Updated: April 29, 2024** — published in full at https://aws.amazon.com/legal/bedrock/third-party-models ("Stability AI serverless models on Amazon Bedrock are sold by Stability AI. If you use any of these models on Amazon Bedrock, you agree to the seller's end user license agreement below."). The Bedrock-specific EULA text **was reachable** and is quoted below — no guessing needed.

| Item | Finding (Stability Bedrock EULA unless noted) |
|---|---|
| Output ownership | "As between you and Stability, you own the Content that you generate using the Services to the extent permitted by applicable law." Also AWS Service Terms §50.2: "The output that you generate using AI Services is Your Content." |
| Outputs as ML training data | **The Stability Bedrock EULA contains NO restriction on using outputs as training data** (read in full; the Community License's "foundational generative AI model" clause is absent, as are revenue thresholds and attribution duties). **However**, AWS Service Terms **§50.5**: "Except where permitted by the AI Service, you may not use an AI Service to generate Content for the express purpose of training an AI model or service or developing a substantially similar AI model or service." On a literal reading the first prong covers generating synthetic data expressly to train the YOLO student. Whether an exception "permitted by the AI Service" applies to Bedrock image models is not stated in the Service Terms → **UNCLEAR; needs AWS/legal confirmation before V6 scales this up.** |
| Commercial use | PERMITTED per-use via Bedrock (pay-per-image, "sold by Stability AI" through AWS invoicing, §50.12.1/§20 Marketplace). No $1M revenue gate in the Bedrock EULA (that gate belongs to the self-hosted Community License, see note below). |
| Prohibited uses | Content guidelines: no sensitive personal info, no IP/privacy violations, nothing obscene/excessively violent, nothing "intentionally misleading, false... with the purpose of harming others", no fully-automated decisions adversely impacting legal rights, no medical advice, no justice/law-enforcement prediction uses. **Nothing about disaster imagery.** |
| Provenance | AWS Service Terms §50.12.6: output "may include... metadata, digital signatures, or watermarks... Neither you nor any End User may modify, tamper with, remove, obscure, or otherwise alter such Provenance Data." (Harmless for detector training; just don't strip it.) |
| Student detector encumbered? | No Stability-side encumbrance (no attribution, no foundational-model clause in the Bedrock EULA). The only open issue is the AWS-side §50.5 question about the generation step itself. |

**Verdicts:** (i) UNCLEAR (provider EULA permits; AWS §50.5 restricts on a literal reading). (ii) UNCLEAR for the same reason — the release of student weights itself is not restricted by any clause found. (iii) PERMITTED (outputs owned; keep provenance data intact).

**Facts:** fully managed — no VRAM/weights to host. Stable Image Core is 2.6B params (AWS page); SD3.5 Large is 8.1B per Stability's own announcements (not re-verified on the pages fetched today). Invocation via Bedrock `InvokeModel`; abuse-detection storage of inputs/outputs up to 30 days may apply to some models (§50.12.2).

**Note — self-hosted alternative (NOT Bedrock):** the **Stability AI Community License** (https://stability.ai/community-license-agreement) governs downloading SD3.5 weights directly: free for "Research or Non-Commercial Purpose" at any org size and for commercial use **only under USD $1,000,000 annual revenue** — above that an Enterprise License is required (Amazon exceeds this, so commercial self-hosting needs Enterprise). It grants "You own any outputs generated from the Models" but adds: "You will not use the Stability AI Materials or Derivative Works, or any output or results... to create or improve any foundational generative AI model (excluding the Models or Derivative Works)" — a YOLO smoke detector is not a foundational generative model, so this clause would not block use case (ii) — plus a mandatory "Powered by Stability AI" display. Stability's AUP (https://stability.ai/use-policy) prohibits deception/misinformation; no disaster-imagery or ML-training-data clauses.

---

## 4. amazon.nova-canvas (AWS first-party, brief)

| Item | Finding |
|---|---|
| Applicable terms | AWS Customer Agreement + **AWS Service Terms §50** (Nova models are "Amazon Foundation Models (Amazon Nova and Titan models)", an AI Service per §50.1) + **AWS Responsible AI Policy** (last updated 2025-01-13). No third-party EULA. |
| Output ownership | §50.2: "The output that you generate using AI Services is Your Content." → generated images are customer content, usable commercially, no attribution duty. |
| Outputs as training data | Same **§50.5** clause as above applies ("...may not use an AI Service to generate Content for the express purpose of training an AI model or service...") → **UNCLEAR** for purposes (i)/(ii); confirm with AWS/legal. No other restriction found. |
| Indemnity | Notable: as of 2026-10-20 **Nova Canvas is not named** in the §50.10 "Indemnified Generative AI Services" list (Nova Micro/Lite/Pro/Premier/Reel/Sonic/2 Omni and Titan Image Generator are; the string "Canvas" does not appear anywhere in the Service Terms). So AWS's generative-output IP indemnity may not cover Nova Canvas outputs. |
| Watermarking | Nova Canvas "Adds an invisible watermark to all generated images" + C2PA content credentials (docs.aws.amazon.com/nova/latest/userguide/image-generation.html); §50.12.6 forbids removing/altering this Provenance Data. Watermarks do not impede detector training. |
| AUP / disaster imagery | AWS Responsible AI Policy prohibitions: "intentional disinformation or deception", privacy violations, likeness without consent, child harm, harassment, safety-filter circumvention, lethal weapon functions. **No disaster-imagery clause.** Relevant for later operation: "If you use the AI/ML Services to make consequential decisions [incl. those impacting a person's... health, or safety], you must... implement appropriate human oversight, testing, and other use case-specific safeguards" — applies to an operational wildfire-detection service built on AWS AI services. |

**Verdicts:** (i) UNCLEAR (§50.5). (ii) UNCLEAR on the generation step; no clause encumbers or claims the trained student weights themselves. (iii) PERMITTED (keep watermark/C2PA metadata intact).

**Facts:** model ID `amazon.nova-canvas-v1:0`; text-to-image, image conditioning, color-guided, image variation, **inpainting and outpainting (mask image or mask prompt), background removal (transparent-background output)**, subject-consistency fine-tuning; max generation 4.19 MP (e.g., 2048x2048), max prompt 1024 chars; regions us-east-1, eu-west-1, ap-northeast-1; managed API — no VRAM/weights.

---

## Acceptable-use check for "disaster imagery"

Checked: Qwen Research License (rules of use §4), GLM-Image (MIT — none), Stability Bedrock EULA Prohibited Uses, Stability AI Use Policy (stability.ai/use-policy), AWS Responsible AI Policy, AWS Service Terms §50. **None contains a clause restricting disaster/wildfire/emergency imagery.** The adjacent clauses that do exist are deception/misinformation prohibitions (Stability: content that is "intentionally misleading, false... with the purpose of harming others"; AWS RAI: "intentional disinformation or deception"; Stability AUP: "misinformation or disinformation") — generating synthetic smoke scenes as internal training data does not implicate them; publishing synthetic disaster imagery as if real would.

## Practical ranking for V6

1. **GLM-Image (MIT)** — only generator with all three use cases cleanly PERMITTED including a future operational service; no attribution, student weights unencumbered. Cost: 35.8 GB weights, source-build stack, ~23 GB VRAM w/ offload, no native RGBA (compositing would need Remove-Background-style post-processing or alpha matting).
2. **Qwen-Image-2.1** — technically strongest for smoke compositing (native RGBA + mask-local editing) and fine for the class-project phase, but research-only: an operational service would need a commercial license from Tongyi, and any released student must carry "Built with Qwen".
3. **Bedrock (Stability / Nova Canvas)** — zero infra and full edit/inpaint APIs, outputs owned by us, but AWS Service Terms §50.5 must be resolved with AWS/legal before using Bedrock output as training data at scale.
