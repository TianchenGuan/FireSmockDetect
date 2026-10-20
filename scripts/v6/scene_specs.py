"""Frozen scene specifications for the V6 generator comparison.

Every generation route (Bedrock SD3.5-L assets, Bedrock inpaint, local
Qwen-Image/GLM-Image) renders from the SAME spec list so acceptance rate,
diversity, label reliability, throughput and cost/accepted-example are
comparable. Frozen with the pre-registration; do not edit between routes.

Asset prompts ask for smoke ON PURE BLACK so a luminance matte gives usable
alpha (additive scattering); RGBA-native generators may instead be asked for
transparent background (template 'rgba').
"""
from __future__ import annotations

ASSET_BASE = ("wispy {kind} rising from a single ground source, {shape}, "
              "{density}, photorealistic volumetric smoke, {light} lighting, "
              "isolated on a pure black background, no ground, no objects, "
              "no fire, no flames, no text")

ASSET_SPECS = [
    # (id, kind, shape, density, light)  — 20 positive smoke asset specs
    ("as01", "thin white smoke", "narrow vertical column", "very faint, barely visible", "midday"),
    ("as02", "thin white smoke", "slightly wind-bent column", "faint translucent", "midday"),
    ("as03", "grey-white wildfire smoke", "small puff cluster just forming", "faint", "morning"),
    ("as04", "bluish-grey smoke", "diffuse drifting veil", "very faint", "hazy afternoon"),
    ("as05", "white smoke", "mushrooming small plume", "moderate", "midday"),
    ("as06", "grey smoke", "wind-sheared horizontal drift", "moderate", "late afternoon"),
    ("as07", "white-grey smoke", "tall thin column with spreading top", "moderate", "morning"),
    ("as08", "dense grey wildfire smoke", "billowing plume", "dense", "midday"),
    ("as09", "dark grey smoke", "turbulent rising plume", "dense", "overcast"),
    ("as10", "white smoke", "two merging thin columns", "faint", "midday"),
    ("as11", "pale smoke", "ragged dispersing puffs", "very faint remnant", "evening"),
    ("as12", "grey-white smoke", "low creeping smoke bank from a point source", "moderate", "morning"),
    ("as13", "white steam-like smoke", "short puffy column", "moderate", "cold morning"),
    ("as14", "yellowish-grey brush-fire smoke", "leaning column", "moderate", "golden hour"),
    ("as15", "grey smoke", "distant tiny plume silhouette", "faint, far away", "midday"),
    ("as16", "white smoke", "column with detached drifting cap", "faint", "windy midday"),
    ("as17", "grey-brown smoke", "churning plume with fine wisps", "dense", "midday"),
    ("as18", "white smoke", "needle-thin wisp", "extremely faint", "midday"),
    ("as19", "grey smoke", "broad low dispersing fan", "faint", "overcast"),
    ("as20", "white-grey smoke", "pulsing puffs in a line downwind", "moderate", "afternoon"),
]

# Negative/confounder assets: composited the same way onto backgrounds so the
# student cannot learn "anything pasted = smoke". Same black-bg convention.
CONFOUNDER_SPECS = [
    ("cf01", "patch of white morning fog", "low horizontal bank", "faint", "dawn"),
    ("cf02", "small cumulus cloud fragment", "puffy blob", "moderate", "midday"),
    ("cf03", "road dust cloud kicked up behind a vehicle", "low trailing cloud", "moderate", "midday"),
    ("cf04", "white water-vapor steam from a cooling vent", "short dissolving puff", "faint", "cold morning"),
    ("cf05", "thin stratus cloud wisp", "horizontal streak", "very faint", "evening"),
    ("cf06", "pale dust devil", "narrow rotating column", "faint", "hot afternoon"),
    ("cf07", "low valley mist patch", "diffuse veil", "very faint", "dawn"),
    ("cf08", "fluffy white cloud", "small detached cloud", "dense", "midday"),
]

# Inpaint prompts operate on real backgrounds with an explicit edit mask.
INPAINT_POS = ("a {density} plume of {kind} rising from the terrain in the "
               "masked area, matching the scene's lighting and haze, "
               "photorealistic, no fire visible")
INPAINT_NEG_EDIT = ("the same landscape with natural {thing} in the masked "
                    "area, photorealistic, no smoke, no fire")
INPAINT_THINGS = ["morning fog", "a small cloud shadow", "dust haze",
                  "heat shimmer", "nothing changed"]

NEGATIVE_PROMPT = ("fire, flames, embers, text, watermark, logo, people, "
                   "buildings, vehicles, cartoon, painting, illustration")


def asset_prompt(spec, rgba_native: bool = False) -> str:
    _, kind, shape, density, light = spec
    p = ASSET_BASE.format(kind=kind, shape=shape, density=density, light=light)
    if rgba_native:
        p = p.replace("isolated on a pure black background",
                      "isolated on a fully transparent background")
    return p
