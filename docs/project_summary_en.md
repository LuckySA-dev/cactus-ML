# Cactus Classifier by KMUTNB — Full Project Summary

Machine Learning course project, King Mongkut's University of Technology North Bangkok (KMUTNB).
All numbers below are from actual runs (2026-10-25). Please do not invent or round numbers up.

## 1. Problem
- Dwarf cacti are popular houseplants in Thailand (sold at farms, plant markets and online), but many genera look very similar to beginners, and shop labels are often missing or wrong.
- Each genus needs different care (some want full sun, some filtered light, some rot easily if overwatered). Wrong care leads to rot, sunburn or stunted growth.
- Goal: a user takes a phone photo and gets the genus/species name plus care advice suited to that plant.

## 2. What the system does
Pipeline: phone photo (JPG / PNG / WEBP / HEIC) → EXIF rotation + Resize 256 → CenterCrop 224 → CNN (DenseNet-121 by default) predicts 20 sub-classes → probabilities are summed up to 7 genera → confidence check:
- Confidence ≥ 60%: show a "plant tag" result card (genus + species + confidence) and care advice (water / light / soil / common mistakes + references).
- Confidence < 60%: answer "not sure" with photo-retake tips, and give no care advice (wrong-genus care could harm the plant).

Two clearly separated parts:
1. ML — image classification: photo → genus + species + confidence (this is the graded ML part).
2. Rule-based — care advice looked up from care_rules/care_rules.json using the predicted genus/species (not ML).

Targets (from the reference cactus-classification papers):
- Genus-level test accuracy ≥ 91% (reference baseline).
- Gap between test set and real-world photos ≤ 9.86% (the reference paper's gap).

## 3. Classes
Level 1 — 7 genera (chosen for popularity in Thailand and ≥ 1,000 potted-plant photos per genus on iNaturalist):

| Genus | Thai name | Level-2 sub-classes |
|---|---|---|
| Astrophytum | แอสโตรไฟตัม | asterias (star cactus), myriostigma (bishop's cap), ornatum, other species |
| Echinocactus | ถังทอง (golden barrel) | single species (Kroenleinia grusonii) |
| Echinopsis | เอคคินอปซิส | chamaecereus (peanut cactus), oxygona, other species |
| Gymnocalycium | ยิมโนคาไลเซียม | colored (moon cactus), variegated, normal — split by how they are traded in Thailand |
| Mammillaria | แมมมิลาเรีย | elongata (ladyfinger), plumosa (feather), prolifera, vetula, other species |
| Opuntia | หูกระต่าย (bunny ears) | single species (O. microdasys) |
| Parodia | พาโรเดีย | leninghausii (yellow tower), magnifica, other species |

20 leaf classes in total. "Flat" approach: predict all 20 directly, then sum leaf probabilities into genera. It beat a hierarchical (two-stage) approach in our comparison: genus 0.942 vs 0.938, species 0.878 vs 0.862.

## 4. Data
Overview:
- 5,035 images used for training/testing, from 4,445 distinct plants.
- 100% of the images used were checked by eye; 526 images were removed during QC.

Sources:
- iNaturalist: 4,390 images (cultivated/captive plants, CC licenses; source, photographer and license recorded per image in metadata.csv).
- Web images collected by the team: 427 (reviewed one by one; watermarked stock photos and drawings removed).
- Our own photos from a cactus farm (iPhone): 218. Half of the plants are held out as a real-world test set the model never sees.

Images per genus: Mammillaria 1,308 · Astrophytum 866 · Echinopsis 799 · Parodia 749 · Gymnocalycium 477 · Opuntia 424 · Echinocactus 412.

Splits (70/15/15, stratified per sub-class):
- train 3,428
- validation 750 (checkpoint selection and tuning; the test set was never used for tuning)
- test 750 (main evaluation)
- real-world test 107 (our farm photos only; measures generalization)

Leakage prevention:
- All photos of the same plant stay in the same split (split by plant_id). The team photographed each plant from several angles, so mixing them would inflate accuracy.
- Duplicate images are removed with perceptual hashing (phash) before splitting, because several sources re-share iNaturalist photos.
- An automated audit after splitting found no plant and no duplicate or near-duplicate image (phash distance ≤ 4) across splits.

Quality control, 4 layers:
1. Random 10% per class reviewed by eye: 0 wrong-genus labels in 280 images.
2. Model-assisted flagging: 5-fold out-of-fold predictions (confident-learning style), 4 rounds; humans reviewed only the images the model disagreed with.
3. Every remaining image (3,845) reviewed by eye, on 241 contact sheets of 16 images each.
4. 100% automated checks: label matches folder, species matches genus, file opens, no leakage.

Removed (526):
- flower covering the plant / plant not visible: 213
- multiple genera in one photo: 178
- blurry / dark / plant too small: 97
- wrong label: 38 (mostly species-level; genuine wrong-genus labels were very rare)

Plants grown in the ground rather than in pots (839 images) were kept, because the plant features are still clear.

iNaturalist note: cultivated plants are flagged "captive" and can never reach research grade, so we accepted any quality grade with ≥ 1 agreeing identification and relied on our own QC instead.

## 5. Models
Six CNN architectures, all ImageNet-pretrained via timm, all trained with exactly the same hyperparameters for a fair comparison:

| Model | timm name | Params | CPU time/image | Used in app |
|---|---|---|---|---|
| DenseNet-121 | densenet121 | 7.0M | 70 ms | #1 (default) |
| EfficientNet-B3 | efficientnet_b3 | 10.7M | 49 ms | #2 |
| EfficientNetV2-B0 | tf_efficientnetv2_b0 | 5.9M | 34 ms | #3 |
| MobileNetV3-Large | mobilenetv3_large_100 | 4.2M | 18 ms | |
| ResNet34 | resnet34 | 21.3M | 30 ms | |
| ResNet50 | resnet50 | 23.5M | 59 ms | |

- DenseNet-121, EfficientNet-B3 and ResNet34 were added following the reference papers (DenseNet-121 won in one paper; the cactus paper used ResNet34).
- The classifier head is zero-initialized, so every model starts at the same loss (ln 20).
- The app uses the top 3 by species-level test accuracy, each with the seed that scored best on validation.

## 6. Training
Recipe:
- Runs: 6 models × 3 seeds (42, 43, 44) = 18 final runs, plus 30 tuning runs; ≈ 6.4 hours of training in total.
- Input 224×224 px, batch size 64.
- Stage 1: backbone frozen, train the head only for 10 epochs; AdamW lr 1e-3; backbone BatchNorm kept in eval mode.
- Stage 2: fine-tune the whole network for up to 40 epochs; AdamW lr 3e-4 (backbone) / 3e-3 (head); cosine annealing; early stopping with patience 8. On average 36 epochs were run, and the best checkpoint was at epoch 30.
- Optimizer: AdamW, weight decay 1e-4.
- Loss: cross-entropy with label smoothing 0.1 (gives better-calibrated confidence, needed for the 60% threshold).
- Regularization: dropout 0.3 before the head; EMA of the weights (decay 0.998) during stage 2.
- Mixed precision: bfloat16 autocast for training; evaluation in fp32, so the confidence values match the app.
- Checkpoint selection: best validation macro-F1 (not accuracy).
- Reproducibility: seeds set everywhere plus deterministic algorithms; re-training reproduces identical results (verified).

Tuning (validation only):
- Learning rate: the original 1e-5/1e-4 was too low. Tuned to 3e-4/3e-3, validation F1 rose from 0.890 to 0.930.
- EMA + stronger augmentation: validation F1 +2.1 points, winning on every seed, so it became the main recipe. Test improved for every model: +0.9 to +1.9 points (genus) and +1.7 to +3.1 points (species).
- Tried and did not help (negative results): test-time augmentation, ensembling several models, class weighting, undersampling large classes.
- Main lesson: about half of the remaining errors came from bad images or wrong labels, so data quality was the main lever. That is why we invested in full QC.

Data augmentation (train set only, in this order):
1. RandomResizedCrop — crop 35–100% of the image, resize to 224×224. Real photos are often taken from far away (small plant in frame) or very close.
2. RandomHorizontalFlip (p = 0.5) — cacti are left-right symmetric. No vertical flip, because plants have a clear top and bottom.
3. RandomRotation ±20° — phones are often held slightly tilted.
4. RandomPerspective (distortion 0.2, p = 0.3) — users often shoot from above or at an angle.
5. GaussianBlur (kernel 5, σ 0.1–2, p = 0.2) — simulates out-of-focus phone photos and hand shake.
6. ColorJitter (brightness ±25%, contrast ±25%, saturation ±20%, hue ±0.02) — different lighting. Hue is kept tiny because color defines some classes (e.g. colored moon cactus).
7. RandomErasing (p = 0.25) — stops the model relying on a single cue such as the flower.
8. Normalize — mean/std from each model's pretrained weights (from timm, not hard-coded).

Validation, test and the app use only Resize 256 → CenterCrop 224 → Normalize.

## 7. Results
Six-model comparison (mean ± SD over 3 seeds; test 750 images, real-world 107 images):

| Model | Genus (test) | Species (test) | Genus (real-world) | Species (real-world) | Gap |
|---|---|---|---|---|---|
| DenseNet-121 | 95.8 ± 0.3% | 91.0 ± 0.5% | 84.4 ± 1.9% | 69.4% | 11.3% |
| EfficientNet-B3 | 95.9 ± 0.3% | 90.4 ± 0.4% | 86.6 ± 2.2% | 71.7% | 9.3% |
| EfficientNetV2-B0 | 95.1 ± 0.2% | 89.8 ± 0.5% | 84.7 ± 0.5% | 71.4% | 10.4% |
| MobileNetV3-L | 94.6 ± 0.1% | 89.0 ± 0.9% | 83.8 ± 2.4% | 69.4% | 10.8% |
| ResNet34 | 93.4 ± 0.7% | 88.0 ± 0.7% | 84.7 ± 1.4% | 68.4% | 8.7% |
| ResNet50 | 93.5 ± 0.5% | 87.7 ± 0.4% | 82.2 ± 1.6% | 68.4% | 11.3% |

Key findings:
- Every model beats the 91% genus-accuracy target. Best is 95.9% (EfficientNet-B3); DenseNet-121 reaches 91.0% at species level.
- The top 3 are within 1 point of each other and mostly not significantly different (McNemar test).
- Per-genus F1 for DenseNet-121 (test, seed 42): Astrophytum 0.98, Opuntia 0.98, Gymnocalycium 0.96, Mammillaria 0.96, Echinocactus 0.95, Echinopsis 0.95, Parodia 0.91 (weakest).

Real-world photos and the gap (gap = test accuracy − real-world accuracy, our generalization measure):
- EfficientNet-B3 (9.3%) and ResNet34 (8.7%) are below the reference paper's 9.86%. The other four are slightly above (10.4–11.3%).
- The gap widened after full QC for two reasons:
  - the test set became cleaner (flower-covered and blurry images that used to drag test scores down were removed);
  - the real-world set has only 107 images, so it fluctuates by about ±3 points.
- Species-level accuracy on real-world photos is still hard (68–72%).

Hardest species-level classes (F1): Parodia other 0.50, Gymnocalycium variegated 0.67 (only 60 images), Parodia magnifica 0.74, Echinopsis other 0.75. The "other species" groups are hard because they mix many species. Confusion of Echinopsis with Mammillaria/Gymnocalycium was expected from the pilot study.

Confidence threshold:
- Below 60% confidence, the app says "not sure" instead of guessing and gives no care advice.
- DenseNet-121 on the test set: it answers 92.4% of images, and 98.0% of those answers are correct. For the rest it asks for a retake.

Explainability (Grad-CAM):
- Shows which parts of the photo drove the decision (spotlight view dims unused regions).
- The models look at the plant body, not the pot or background.
- Flowers were a strong cue (a large flower could cause wrong predictions). This motivated RandomErasing and removing flower-covered images in QC. With the new recipe, a large-flower Gymnocalycium that used to be misclassified is now correct in all 3 models, which look at the stem.

## 8. The working system
Web app (Streamlit), page 1 "Identify + care" (end users):
- Upload a photo (HEIC from iPhone supported, auto-rotated by EXIF) or use the camera (opens only when the user taps it).
- Result shown as a "plant tag" card: Thai name, scientific name, species, and confidence in words (high ≥ 85% / fairly confident ≥ 60% / not sure).
- Three care cards (water, light, soil), a "watch out" box and reference links, labelled "species-specific" or "general for the genus".
- If not sure: 4 tips for retaking the photo.
- "Why this answer": genus probabilities, species within the top genus, and Grad-CAM.
- Sidebar: choose one of 3 models, ranked by accuracy (1. DenseNet-121 91.0%, 2. EfficientNet-B3 90.4%, 3. EfficientNetV2-B0 89.8%), or compare all 3 at once.
- Empty state shows a 3-step how-to and photo tips. Mobile-friendly. Uploaded photos are not stored.

Web app, page 2 "Model evaluation" (presentation):
- Results tab:
  - KPIs against the 91% target and the 9.86% gap;
  - 6-model comparison charts (mean ± SD) and table with McNemar tests;
  - drill-down by model / seed / dataset / level: confusion matrix, per-class accuracy, confidence-threshold slider, gallery of confident mistakes.
- Project details tab: task, data, QC, models, training recipe, augmentation (explanations plus example images per step), evaluation and system. All numbers are read automatically from the result files.

Rule-based care advice:
- care_rules/care_rules.json: 7 genera + 14 species. Each entry has water, light and soil (summary + detail), common mistakes, and references (LLIFLE, NC State Extension, University of Arkansas Extension, etc.).
- Species advice is used when species confidence ≥ 60%; otherwise genus advice. "Other species" classes use the genus advice.
- Seasons converted from temperate-climate sources to Thailand: hot/rainy season (Mar–Oct) = growing season; cool season (Nov–Feb) = rest period.
- Content was cross-checked against the sources, and claims the sources did not support were removed.

REST API (FastAPI), for future front-ends:
- GET /api/health — server readiness.
- GET /api/models — the 3 available models with their mean scores.
- POST /api/predict — upload a photo; returns genus, species, confidence and care advice (optional Grad-CAM).
- POST /api/compare — one photo predicted by all 3 models.
- Safeguards: 10 MB upload limit, real-image validation (PIL), model name must be on the allow-list, CORS configured via environment variable.
- 11 automated pytest tests, all passing.

Deployment:
- Code on GitHub; the 3 app models are stored with Git LFS (~96 MB).
- Deployable on Streamlit Community Cloud (app/app.py, Python 3.12, CPU build of PyTorch with pinned versions).
- Tested from a clean checkout on CPU: both pages run without errors.

## 9. Limitations and future work
Limitations:
- Only 7 genera are known. A photo of another genus is mapped to the closest one, though confidence is usually low, so the app says "not sure".
- Only 107 real-world farm photos, and none of Echinopsis (too few plants at the farm), so real-world numbers fluctuate by about ±3 points.
- Species-level accuracy on real-world photos is 68–72%, weakest for the "other species" groups and variegated Gymnocalycium (little data).
- About 12 iNaturalist images have questionable species labels. They are flagged in the notes for expert review and not changed.
- Care advice is a team-written draft based on references and has not yet been reviewed by a plant expert.
- The gap of 4 of the 6 models is slightly above the reference paper's.

Future work:
- Collect more of our own photos (especially Echinopsis and variegated Gymnocalycium) to reduce the gap and measure real-world accuracy more precisely.
- Add more popular genera (backups already identified: Cephalocereus, Melocactus) and a "not a cactus" class.
- Build a mobile app or a new front-end on the existing API, and export models to ONNX for faster mobile inference.
- Have a plant expert review the care advice.

## 10. Tech stack
- Language: Python 3.13 (training machine), Python 3.12 (deployment).
- Deep learning: PyTorch 2.11 (CUDA 12.8), timm 1.0.29 (pretrained models), torchvision v2 transforms.
- Training hardware: NVIDIA RTX 5060 Ti 16 GB (Blackwell), bfloat16 mixed precision.
- Training techniques: AdamW, cosine annealing, EMA (timm ModelEmaV3), label smoothing, early stopping.
- Evaluation: scikit-learn (F1, confusion matrix, StratifiedGroupKFold), SciPy (McNemar test).
- Explainability: pytorch-grad-cam (Grad-CAM).
- Data: iNaturalist API, Pillow + pillow-heif (HEIC), ImageHash (phash deduplication), pandas; metadata.csv with one row per image (source, license, plant ID, QC status, split).
- Charts: matplotlib + seaborn for the report (colorblind-checked palette), Altair for in-app charts.
- Web app: Streamlit 1.64 (multipage, custom theme, Anuphan + Mali fonts).
- API: FastAPI + Uvicorn + Pydantic; pytest + httpx for tests.
- Code quality: ruff linting; seeds everywhere and deterministic training for reproducible results.
- Version control: Git + GitHub, Git LFS for model checkpoints.
- Deployment: Streamlit Community Cloud (CPU).

Code structure:
- src/config.py — all constants (classes, paths, hyperparameters)
- src/data/ — iNaturalist download, team-photo import, QC sheets, suspect flagging, dataset splitting
- src/train.py — 2-stage training + EMA + early stopping
- src/evaluate.py — test + real-world evaluation → reports/results.csv
- src/compare.py — 6-model comparison (mean ± SD) + McNemar
- src/visualize.py — report figures + augmentation examples
- src/gradcam.py — Grad-CAM
- src/predict.py — photo → genus + species + care advice (shared by the app and the API)
- app/ — Streamlit app (2 pages)
- api/main.py — REST API (FastAPI)
- care_rules/ — rule-based care advice
- tests/ — API tests
