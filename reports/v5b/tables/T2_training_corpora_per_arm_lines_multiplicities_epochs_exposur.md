_generated 2026-10-03T03:55Z by scripts/v5b/make_report_tables_v5b.py from saved artifacts - do not edit by hand_

### T2. Training corpora per arm: lines, multiplicities, epochs, exposures (matched budget), masked / pseudo-label counts

Exposures = lines x epochs, matched to arm A (38,454 x 60 = 2,307,240) within -0.3%/-0.65%. C = B + conservative ignore regions (teacher-agreed, GT-free rectangles filled with image mean); D = B + filtered teacher pseudo boxes on already-labelled E/Boreal images; D2 = D rules + looser all-source pseudo-labels (near-replicate of D: only 408 boxes, 0 on FireMan); D3 = B + teacher pseudo boxes on genuinely UNLABELLED smoke-positive material (the real teacher-to-student test). C/D are small by construction: on already-labelled data the teachers agree with the authors.

| arm | train lines | components raw x multiplicity = effective | epochs | exposures | delta vs A | masked (C) | pseudo boxes (D/D2/D3) | label origin |
|---|---|---|---|---|---|---|---|---|
| A (control; expV5bA_ctrl_s3, seed-0 replicate) | 38,454 | E x1 | 60 | 2,307,240 | 0.000% | - | - | author only |
| B | 53,511 | E 38,454x1=38,454, boreal 2,062x2=4,124, fireman 8,279x1=8,279, sheffield 1,327x2=2,654 | 43 | 2,300,973 | -0.272% | - | - | author 50,085, author_verified_empty 37 |
| C | 53,511 | E 38,454x1=38,454, boreal 2,062x2=4,124, fireman 8,279x1=8,279, sheffield 1,327x2=2,654 | 43 | 2,300,973 | -0.272% | 300 rects on 148 imgs ({'sdis': 145, 'dfire': 2, 'boreal': 1}) | - | author 49,937, author_labels+masked_pixels 148, author_verified_empty 37 |
| D | 53,508 | E 38,454x1=38,454, boreal 2,062x2=4,121, fireman 8,279x1=8,279, sheffield 1,327x2=2,654 | 43 | 2,300,844 | -0.277% | - | 495 boxes on 495 imgs; by source {'sdis': 449, 'dfire': 43, 'boreal': 3} | author 49,590, author+model-assisted 495, author_verified_empty 37 |
| D2 | 53,508 | E 38,454x1=38,454, boreal 2,062x2=4,123, fireman 8,279x1=8,279, sheffield 1,327x2=2,652 | 43 | 2,300,844 | -0.277% | - | 408 boxes on 408 imgs; by source {'sdis': 394, 'dfire': 11, 'boreal': 1, 'sheffield': 2} | author 49,677, author+model-assisted 408, author_verified_empty 37 |
| D3 | 57,307 | E 38,454x1=38,454, boreal 2,062x2=4,124, fireman 8,279x1=8,279, sheffield 1,327x2=2,654, d3 3,796x1=3,796 | 40 | 2,292,280 | -0.648% | - | 3796 unlabelled imgs kept with pseudo boxes (rule: D2 rules without GT filter (no author boxes exist): (a) FUSED calibrated WBF >= 0.9918 (ac...) | author 50,085, author_verified_empty 37, model-assisted 3,796 |

**new-source share and exclusions**

| arm | FireMan explicit negs | Sheffield explicit negs | new-source effective lines | new-source / E | excluded (from the manifest) |
|---|---|---|---|---|---|
| B | 1,572 | 24 | 15,057 | 39.2% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| C | 1,572 | 24 | 15,057 | 39.2% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| D | 1,572 | 24 | 15,054 | 39.1% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| D2 | 1,572 | 24 | 15,054 | 39.1% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |
| D3 | 1,572 | 24 | 15,057 | 39.2% | FireSpot (licence pending); WSDataset (blocked); AuSmoke/MNS (permission pending; 10,904 T lines); FIgLib (eval-only); Boreal heinola (aerial VAL) / ruokolahti ... |

Sources:
- `data/v5b/manifests/CORPUS_B_v1.json`
- `data/v5b/manifests/CORPUS_C_v1.json`
- `data/v5b/manifests/CORPUS_D_v1.json`
- `data/v5b/manifests/CORPUS_D2_v1.json`
- `data/v5b/manifests/CORPUS_D3_v1.json`
