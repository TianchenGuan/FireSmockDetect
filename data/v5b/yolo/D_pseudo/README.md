# D_pseudo -- MODEL-ASSISTED pseudo boxes for arm D (never author labels, never human-reviewed)
Generated 2026-10-02T09:00:11Z by scripts/v5b/build_corpus_v5b.py from models/v5b/teacher/*__train.jsonl.
Rule: >=2 of 3 teachers agree (IoU>=0.5), each >= its calibrated 20%-neg-FP conf ({"dino_b4_lora_L": 0.9727, "dino_b6_lora_Hplus": 0.9802, "rfdetr_L": 0.7733}), no GT box within IoU>=0.3; Boreal human-verified-empty frames excluded. Label files = GT lines first, then pseudo lines. Sidecar: pseudo_boxes.csv.
