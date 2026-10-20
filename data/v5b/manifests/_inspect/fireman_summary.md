## rows by modality/split
| modality | test | train | val |
|---|---|---|---|
| rgb | 6009 | 58238 | 5624 |
| thermal | 5073 | 26294 | 1685 |

## RGB smoke label type x origin
| label_type | author |
|---|---|
| box | 64867 |
| image | 5004 |

## RGB smoke_presence x has boxes
| smoke_presence | False | True |
|---|---|---|
| no | 4979 | 3 |
| yes | 25 | 64864 |

## RGB fire_presence x smoke_presence
| fire_presence | no | yes |
|---|---|---|
| ABSENT | 0 | 228 |
| no | 4854 | 2244 |
| yes | 128 | 62417 |

## RGB author_box_source
| author_box_source | count |
|---|---|
| file | 64341 |
| none | 5004 |
| manual | 346 |
| semi-auto | 180 |

## RGB smoke_intensity
| smoke_intensity | count |
|---|---|
| medium | 34594 |
| high | 27995 |
| low | 7282 |

## RGB camera_perspective / smoke_interaction
| camera_perspective | NA | clear_view | through_smoke |
|---|---|---|---|
| NA | 1563 | 0 | 0 |
| side_view | 0 | 54605 | 3183 |
| top_view | 0 | 7147 | 3373 |

## RGB resolutions
| width | height | n |
|---|---|---|
| 640 | 512 | 20090 |
| 958 | 760 | 2756 |
| 958 | 764 | 1687 |
| 958 | 768 | 8530 |
| 1920 | 1080 | 3233 |
| 3840 | 2160 | 33575 |

## per video (RGB)
| video_id | parent_unit | burn_date | proposed_split | n | box_pos | expl_neg | img_only_pos | fire | conflicts | boxes | w | h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rgb/test/video_001 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 329 | 0 | 329 | 0 | 0 | 0 | 0 | 3840 | 2160 |
| rgb/test/video_002 | fireman_burnC_taivalkoski_2023-08-26 | 2023-08-26 | holdout | 4111 | 4111 | 0 | 0 | 4111 | 0 | 4111 | 640 | 512 |
| rgb/test/video_003 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 964 | 964 | 0 | 0 | 964 | 0 | 964 | 958 | 764 |
| rgb/test/video_004 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 292 | 292 | 0 | 0 | 292 | 0 | 292 | 3840 | 2160 |
| rgb/test/video_005 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 313 | 313 | 0 | 0 | 0 | 0 | 313 | 3840 | 2160 |
| rgb/train/video_001 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 180 | 180 | 0 | 0 | 180 | 0 | 180 | 3840 | 2160 |
| rgb/train/video_002 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 570 | 570 | 0 | 0 | 570 | 0 | 570 | 3840 | 2160 |
| rgb/train/video_003 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 7703 | 7621 | 82 | 0 | 7458 | 0 | 8137 | 3840 | 2160 |
| rgb/train/video_004 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 3417 | 3073 | 338 | 6 | 1616 | 9 | 3073 | 3840 | 2160 |
| rgb/train/video_005 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 869 | 33 | 836 | 0 | 33 | 0 | 33 | 3840 | 2160 |
| rgb/train/video_006 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 840 | 840 | 0 | 0 | 840 | 0 | 840 | 3840 | 2160 |
| rgb/train/video_007 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 1469 | 1469 | 0 | 0 | 1469 | 0 | 1469 | 3840 | 2160 |
| rgb/train/video_008 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 1499 | 1499 | 0 | 0 | 1499 | 0 | 2779 | 3840 | 2160 |
| rgb/train/video_009 | fireman_burnC_taivalkoski_2023-08-26 | 2023-08-26 | holdout | 3008 | 3008 | 0 | 0 | 3008 | 0 | 3008 | 640 | 512 |
| rgb/train/video_010 | fireman_burnC_taivalkoski_2023-08-26 | 2023-08-26 | holdout | 5446 | 5446 | 0 | 0 | 5446 | 0 | 5446 | 640 | 512 |
| rgb/train/video_011 | fireman_burnC_taivalkoski_2023-08-26 | 2023-08-26 | holdout | 5841 | 5841 | 0 | 0 | 5841 | 0 | 5841 | 640 | 512 |
| rgb/train/video_012 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 2756 | 2691 | 65 | 0 | 2756 | 0 | 2691 | 958 | 760 |
| rgb/train/video_013 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 723 | 516 | 188 | 19 | 503 | 19 | 516 | 958 | 764 |
| rgb/train/video_014 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 2537 | 2537 | 0 | 0 | 2537 | 0 | 2537 | 3840 | 2160 |
| rgb/train/video_015 | fireman_burnD_overcast_clearcut_edge | 2022-07-06 or 2023-06-14 (unresolved; the date not used by burn A) | val | 8530 | 7376 | 1154 | 0 | 7375 | 0 | 7376 | 958 | 768 |
| rgb/train/video_016 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 373 | 373 | 0 | 0 | 373 | 0 | 373 | 3840 | 2160 |
| rgb/train/video_017 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 327 | 327 | 0 | 0 | 122 | 0 | 327 | 3840 | 2160 |
| rgb/train/video_018 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 1811 | 0 | 1811 | 0 | 0 | 0 | 0 | 3840 | 2160 |
| rgb/train/video_019 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 607 | 607 | 0 | 0 | 607 | 0 | 607 | 3840 | 2160 |
| rgb/train/video_020 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 630 | 630 | 0 | 0 | 630 | 0 | 630 | 3840 | 2160 |
| rgb/train/video_021 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 953 | 953 | 0 | 0 | 953 | 0 | 953 | 1920 | 1080 |
| rgb/train/video_022 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 953 | 953 | 0 | 0 | 953 | 0 | 953 | 1920 | 1080 |
| rgb/train/video_023 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 2281 | 2281 | 0 | 0 | 2281 | 0 | 2727 | 3840 | 2160 |
| rgb/train/video_024 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 4915 | 4915 | 0 | 0 | 4915 | 0 | 7040 | 3840 | 2160 |
| rgb/val/video_001 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 1409 | 1409 | 0 | 0 | 1409 | 0 | 1409 | 3840 | 2160 |
| rgb/val/video_002 | fireman_burnC_taivalkoski_2023-08-26 | 2023-08-26 | holdout | 1053 | 1053 | 0 | 0 | 847 | 1 | 1053 | 640 | 512 |
| rgb/val/video_003 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 1327 | 1151 | 176 | 0 | 1126 | 0 | 1151 | 1920 | 1080 |
| rgb/val/video_004 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2022-08-16 (candidate; README lists Aug 16 2022; Boreal Ruokolahti burn = 2022-08-15) | quarantine | 1204 | 1204 | 0 | 0 | 1204 | 0 | 1204 | 3840 | 2160 |
| rgb/val/video_005 | fireman_burnA_sunny_bog | 2022-07-06 or 2023-06-14 (unresolved) | train | 631 | 631 | 0 | 0 | 631 | 0 | 631 | 640 | 512 |

## per unit (RGB)
| parent_unit | proposed_split | n_frames | n_videos | box_pos | expl_neg | boxes |
|---|---|---|---|---|---|---|
| fireman_burnA_sunny_bog | train | 23263 | 15 | 19861 | 3396 | 21657 |
| fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | quarantine | 18619 | 13 | 18171 | 429 | 20742 |
| fireman_burnC_taivalkoski_2023-08-26 | holdout | 19459 | 5 | 19459 | 0 | 19459 |
| fireman_burnD_overcast_clearcut_edge | val | 8530 | 1 | 7376 | 1154 | 7376 |

## dup_class
| modality | dup_near_dfire | none |
|---|---|---|
| rgb | 19 | 69852 |
| thermal | 33 | 33019 |

## label conflicts
| label_conflict | count |
|---|---|
|  | 69842 |
| presence_yes_no_boxes | 25 |
| presence_no_but_boxes | 3 |
| tag_conflict:camera | 1 |

## thermal
| video_id | parent_unit | n | fire_boxes | smoke_presence | w | h |
|---|---|---|---|---|---|---|
| thermal/test/video_t_01 | fireman_burnC_taivalkoski_2023-08-26 | 4107 | 4107 | unknown | 640 | 512 |
| thermal/test/video_t_02 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 966 | 1871 | unknown | 958 | 764 |
| thermal/train/video_t_01 | fireman_burnC_taivalkoski_2023-08-26 | 3005 | 3005 | yes | 640 | 512 |
| thermal/train/video_t_02 | fireman_burnC_taivalkoski_2023-08-26 | 5442 | 5442 | yes | 640 | 512 |
| thermal/train/video_t_03 | fireman_burnC_taivalkoski_2023-08-26 | 5838 | 5838 | yes | 640 | 512 |
| thermal/train/video_t_04 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 2756 | 2716 | unknown | 958 | 760 |
| thermal/train/video_t_05 | fireman_burnB_evening_dusk_RUOKOLAHTI_CANDIDATE | 723 | 507 | unknown | 958 | 764 |
| thermal/train/video_t_06 | fireman_burnD_overcast_clearcut_edge | 8530 | 8497 | unknown | 958 | 768 |
| thermal/val/video_t_01 | fireman_burnC_taivalkoski_2023-08-26 | 1053 | 870 | unknown | 640 | 512 |
| thermal/val/video_t_02 | fireman_burnA_sunny_bog | 632 | 632 | yes | 640 | 512 |

