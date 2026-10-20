# V5 — Email drafts for USER review (NOT sent)

Prepared 2026-10-02 (UTC). Two drafts in English, each with a short Chinese summary
and a list of facts the user must verify before sending. Placeholders are in
[SQUARE BRACKETS]. Our institutional sender, signature and any legal wording must
be supplied by the user; nothing below has been sent or pre-announced.

---------------------------------------------------------------------------

## 1. To FGI — request for the full raw UAV flight recordings behind the Boreal Forest Fire dataset

**To:** Julius Pesonen <julius.pesonen@nls.fi>  (corresponding author; verified from the Nature article page "Correspondence to Julius Pesonen")
**Cc (suggested):** Eija Honkavaara (Research Manager, FGI; e-mail NOT verified — FGI/NLS pattern is firstname.lastname@nls.fi), [our institutional contact]
**Subject:** Request for full-length UAV recordings (incl. pre-ignition / no-smoke segments) from the Boreal Forest Fire dataset campaigns (Sci Data 2025)

Dear Mr Pesonen,

Thank you for publishing the Boreal Forest Fire dataset (Pesonen et al., Scientific Data 2025, doi:10.1038/s41597-025-05634-0; Fairdata doi:10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9). We downloaded the complete 125 GB release from Fairdata under CC BY 4.0 and verified every file against the Metax checksums. It is currently the only genuine UAV wildfire-smoke dataset in our work, and we have found it very valuable.

We are a small team at [ORGANISATION] building an internal research prototype for early wildfire-smoke detection from fixed cameras and UAVs. Our current bottleneck is not smoke imagery but its complement: continuous aerial footage of the same forests, flown with the same platform, in which no smoke is present, and complete recordings of the moments before and after ignition. We need these to measure false-alert rates per hour of flight and the delay to the first valid warning, which the 30-second clips of Subset B were understandably not designed for.

Your Methods section notes that the original footage was reviewed and trimmed (removing stationary ground-level, identifying or otherwise irrelevant segments) before being cut into 30-second clips. We would like to ask whether it would be possible to obtain, for research use only, the full-length original recordings from the four 2022 burns (Evo, Heinola, Karkkila, Ruokolahti), specifically:

1. the complete flight recordings, including segments before ignition, segments where the camera is pointed away from the burn (no visible smoke), and transitions from no-smoke to first visible smoke;
2. the associated DJI flight logs / SRT subtitle tracks or any other timestamp, GPS and gimbal data, so that we can align frames with time of day and camera pitch;
3. if available, the approximate ignition time of each burn (or the burn plans / notes from which it can be read).

If it is simpler, access to a subset (for example the Heinola and Ruokolahti flights, which our protocol holds out for evaluation) would already be a great help. We can accept any transfer method you prefer (Fairdata/IDA share, Funet FileSender, SFTP, or a shipped drive at our cost) and are happy to sign a data-use statement.

So that there is no doubt about how the material would be used:
- the footage would be used only for internal research and evaluation of our prototype; it would not be redistributed;
- we keep the dataset's site-level split fixed: Heinola material is used only for validation/calibration and Ruokolahti only for final testing; nothing from those two sites enters model training, and any raw video we receive from them would inherit the same assignment;
- any derived annotations we create (frame-level smoke boxes, onset timestamps, "no-smoke" verification labels) would be offered back to you; we would like to ask under which licence you would prefer such derived annotations to be released, if at all, and whether you would want to be co-credited;
- we will cite the Scientific Data article and the Fairdata DOI in any internal or external report.

We would also be glad to hear whether your group has, or plans, further flights (burn or no-burn) that could be shared under similar terms, and whether there is a preferred contact at FGI for data requests of this kind.

Thank you very much for considering this request.

Kind regards,
[NAME]
[TITLE, TEAM]
[ORGANISATION, ADDRESS]
[E-MAIL / PHONE]

### 中文摘要（草稿 1）
向 Boreal Forest Fire 数据集通讯作者 Julius Pesonen（FGI/NLS，julius.pesonen@nls.fi）索取 2022 年四次计划烧除（Evo、Heinola、Karkkila、Ruokolahti）的完整原始无人机录像，重点是：点火前片段、镜头转向无烟区域的片段、"无烟→首次可见烟"的过渡段；同时索取 DJI 飞行日志 / SRT 字幕（时间戳、GPS、云台角度）和各次烧除的点火时间。说明用途为内部早期烟雾检测研究原型，不再分发；承诺 Heinola 只用于校准/验证、Ruokolahti 只用于最终测试，拿到的原始视频继承现有划分；询问派生标注的许可方式与署名意愿；询问是否有其他可共享的飞行数据。

### 发送前请核实（草稿 1）
- [ ] 收件人邮箱 julius.pesonen@nls.fi：来自 Nature 文章页 mailto 链接（2026-10-02 抓取）。请在发送前再确认该地址仍有效（作者 ORCID 显示同时隶属 FGI 与 Aalto University）。
- [ ] Cc 对象 Eija Honkavaara 的邮箱未经核实（我们只知道 FGI 研究经理身份，ORCID 0000-0002-7236-2145）。要么查到确切地址，要么删掉 Cc。
- [ ] 我们的机构名称、发件人、职务、地址、签名；是否需要法务审阅"research use only / data-use statement"措辞。
- [ ] 我们能接受的传输方式（IDA 共享、Funet FileSender、SFTP、寄硬盘）以及是否愿意承担寄送费用。
- [ ] 是否愿意把派生标注回馈给作者并接受他们指定的许可；是否同意署名/致谢。
- [ ] 若要提前征得 Fairdata 侧配合，仓库服务台为 servicedesk@csc.fi（fairdata.fi/en/contact 已核实），但数据权利人是 FGI，首要联系人仍是作者。

---------------------------------------------------------------------------

## 2. To the AusSmoke / MultiNatSmoke authors (ANU) — explicit licence terms for research use

**To:** henryzhao0615@gmail.com  (the only contact stated in the repository README "## Contact"; corresponds to author Hongjin Zhao, HF account hongjinzhao0615)
**Cc (suggested, to verify):** Nick Barnes <nick.barnes@anu.edu.au>, Marta Yebra <marta.yebra@anu.edu.au>  (reconstructed from the ANU research portal, which shows "Email nick.barnes anu.edu au" / "Email marta.yebra anu.edu au"); Weihao Li (first author; no address found)
**Subject:** AusSmoke / MultiNatSmoke (WACV 2026) — request for explicit licence terms for research use

Dear Dr Zhao and colleagues,

Congratulations on "AusSmoke meets MultiNatSmoke: a fully-labelled diverse smoke segmentation dataset" (WACV 2026). We are a small team at [ORGANISATION] working on an internal research prototype for early wildfire-smoke detection, and the AusSmoke collection — early-stage smoke from planned burns in the ACT, captured from the Mount Stromlo PTZ camera, handheld cameras and drones, with frame offsets from first visible smoke — is exactly the kind of data our evaluation lacks.

We downloaded MultiNatSmokeDataset.zip from the Hugging Face repository (hongjinzhao0615/MultiNatSmoke, revision 5b7a4e0f…) and read the GitHub README. We could not find a licence statement for the AusSmoke images and masks, or for the MultiNatSmoke aggregate as a whole; the README's licence column lists the constituent public datasets but shows "–" for AusSmoke, and neither the repository nor the Hugging Face card carries a LICENSE file. Before we go any further we would therefore like to ask you directly:

1. Under what terms may the AusSmoke subset (images and masks) be used for internal, non-commercial-in-nature research and evaluation by our team? If you intend a standard licence (e.g. CC BY 4.0 or CC BY-NC 4.0), could you confirm it in writing or add it to the repository?
2. Do the same terms apply to the MultiNatSmoke aggregate, i.e. to the segmentation masks you added on top of the constituent datasets?
3. Are there any restrictions from the ACT Parks and Conservation Service / ACT Rural Fire Service, or from the camera owners, that we should respect (for example, no redistribution, acknowledgement wording, or notification of publications)?
4. If you plan a formal data release or a data paper with a licence, we would be happy to wait for it and cite it.

For transparency about what we have done so far: to avoid evaluation contamination we quarantined the parts of MultiNatSmoke that are copies of datasets we already hold or evaluate on — all FIgLib-derived subsets (FIgLib, SmokeSeg, AI-for-Mankind, Firecam, SMOKE5K), the D-Fire copy, and the Boreal Forest Fire copy — and we have not used them. We used AusSmoke frames in one internal experiment, grouped by source video so that no video appears in both training and evaluation; we have since frozen any further use pending your reply, and we will remove the data if you ask us to. Nothing has been redistributed or published.

We would also be grateful to know whether the original AusSmoke videos (rather than the ~1 fps frame extracts), or any smoke-free recordings from the same cameras, could be made available for research, since continuous footage is what we need to measure false-alert rates over time.

Thank you very much for your time, and for making this work available to the community.

Kind regards,
[NAME]
[TITLE, TEAM]
[ORGANISATION, ADDRESS]
[E-MAIL / PHONE]

### 中文摘要（草稿 2）
向 AusSmoke/MultiNatSmoke 作者（ANU Bushfire Research Centre of Excellence；README 唯一联系方式 henryzhao0615@gmail.com，建议 Cc 资深作者 Nick Barnes、Marta Yebra）说明：仓库与 HF 页面均无许可文件，AusSmoke 一栏许可为"–"；请求书面确认 AusSmoke 子集与整个聚合数据集的研究使用许可、是否有来自 ACT 公园保护局/乡村消防局或相机所有者的限制、是否计划正式带许可发布。坦白告知：我们已隔离 FIgLib/D-Fire/Boreal 的重复副本未使用；AusSmoke 帧曾在一次内部实验中按视频分组使用，现已冻结进一步使用，如作者要求可删除；未再分发。另询问是否可提供原始视频或同相机的无烟录像。

### 发送前请核实（草稿 2）
- [ ] 主收件人 henryzhao0615@gmail.com 是 README 中唯一写明的联系方式（2026-10-02 抓取），对应作者 Hongjin Zhao；但个人 Gmail 不等于机构授权——如果需要有效力的许可确认，建议同时联系机构邮箱（见下一条）。
- [ ] Cc 地址 nick.barnes@anu.edu.au、marta.yebra@anu.edu.au 是从 ANU 研究门户上去除混淆后的重建结果，未经直接验证；请核对或改用 ANU 官方人员目录。第一作者 Weihao Li 未找到地址。
- [ ] "我们在一次内部实验中使用了 AusSmoke 帧" 这段披露是否符合你希望对外说的口径（事实依据：reports/v4/DATA_CARD_V4.md、forest_holdout_v4.json；v4 使用已在 PROTOCOL_V5 中标注为"已披露、不再扩展"）。
- [ ] 我们对"非商业性质的内部研究"措辞是否准确——若我们所在机构为营利企业，需由法务确认该表述和可接受的许可类型（例如 CC BY-NC 对我们是否可用）。
- [ ] 机构名称、发件人、签名；是否愿意承诺"如作者要求即删除"。
- [ ] 若作者回复许可为 CC BY-NC 或要求书面协议，PROTOCOL_V5 §2 中 AuSmoke/MultiNatSmoke 的状态需相应更新（目前为 PERMISSION PENDING）。

---------------------------------------------------------------------------

## 3. Facts behind both drafts (sources fetched 2026-10-02)
- Nature article s41597-025-05634-0: "Correspondence to Julius Pesonen"; mailto julius.pesonen@nls.fi; affiliations FGI (Espoo), Aalto University, University of Jyväskylä, University of Vaasa; published 2025-08-13; Methods: "the original drone footage was reviewed to remove non-purposeful or sensitive segments ... then segmented into fixed 30-second clips".
- Metax record (data/v4/raw/boreal/_metax_dataset.json): creators incl. Pesonen, Raita-Hakola, Joutsalainen, Hakala, Akhtar, Karjalainen, Koivumäki, Markelin, Suomalainen, Alves de Oliveira, Pölönen, Honkavaara; publisher & rights holder "FGI Dept. of Remote sensing and photogrammetry, National Land Survey of Finland"; licence CC BY 4.0; e-mails hidden.
- ORCID: Pesonen 0009-0000-9175-7129 (FGI + Aalto, current); Honkavaara 0000-0002-7236-2145 ("Research Manager", FGI). No public e-mails on ORCID.
- Fairdata contact page: servicedesk@csc.fi (CSC customer support).
- MultiNatSmoke GitHub (henryzhao0615/MultiNatSmoke): README "## Contact henryzhao0615@gmail.com"; licence table row "AusSmoke" absent / constituent licences listed; repo root has no LICENSE; GitHub API license = null; last push 2026-05-04. HF dataset card: no licence field. WACV 2026 PDF: authors and "Bushfire Research Centre of Excellence, Australian National University"; no e-mails; data collection with ACT Parks and Conservation Service / ACT Rural Fire Service; Axis Q6318-LE PTZ on Mount Stromlo.
- ANU research portal: nick-barnes and marta-yebra pages render e-mails as "nick.barnes anu.edu au" / "marta.yebra anu.edu au"; hongjin-zhao page 404.
