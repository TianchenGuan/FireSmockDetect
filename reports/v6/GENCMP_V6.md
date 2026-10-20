# V6 generator comparison (frozen scene specs)

| route | generated | accepted | acceptance | median lat (s) | $/accepted |
|---|---|---|---|---|---|
| sd35_large | 48 | 41 | 0.854 | 5.8 | 0.094 |
| stable_image_core | 48 | 8 | 0.167 | 2.5 | 0.24 |
| stable_image_inpaint | 30 | 20 | 1.0 | 17.6 | 0.07 |

Bedrock outputs are comparison/demo only (AWS Service Terms 50.5 ambiguity, prereg A1); the training corpus generator is GLM-Image (MIT). All acceptance checks are AUTOMATED; VLM plausibility (model-assisted) joins from data/v6/review/.
