Here’s the version I’d actually execute. It preserves the high-signal research story while making it very hard for the project to spiral.

# Sparse-View 3D Gaussian Splatting Project Plan

## Goal

Build a correct, understandable 3D Gaussian Splatting implementation, then answer one focused question:

> **How does adaptive densification behave when multi-view supervision becomes sparse, and can a monocular-depth prior improve the resulting geometry?**

The project has three layers:

**1. Must finish:** working 3DGS + small sparse-view study
**2. Target:** one geometry-aware depth intervention
**3. Stretch:** broader validation only if the first two produce something interesting

Do **not** commit to the stretch work in advance.

---

# Phase 1 — Get a correct baseline working

Your repository currently has projection/compositing/basic training infrastructure, but the PyTorch rasterizer, adaptive density control, and end-to-end COLMAP path are still incomplete. 

### 1A. COLMAP + cameras

Implement the minimum clean scene-loading path:

* Load camera intrinsics.
* Load camera extrinsics.
* Load RGB images.
* Load sparse COLMAP points.
* Preserve camera IDs.
* Downscale images **and intrinsics together**.
* Initialize Gaussian scale based on local point spacing rather than a fixed constant.

Add one important test:

> Project known COLMAP 3D points back into their associated cameras and confirm they land where expected.

This is your defense against silent camera-convention bugs.

### 1B. Finish the readable PyTorch renderer

Implement:

1. 3D Gaussian covariance.
2. World → camera transformation.
3. Perspective projection.
4. Covariance projection via the Jacobian.
5. Visibility/culling.
6. SH color evaluation.
7. Depth ordering.
8. 2D Gaussian evaluation.
9. Front-to-back alpha compositing.
10. RGB + alpha + expected depth output.

Don't optimize it.

Its purpose is:

> **I have a renderer whose mathematics I understand and can inspect.**

Production training can use the CUDA backend.

### 1C. CUDA sanity check

Do **only enough parity work to trust it**:

* RGB approximately agrees on a tiny scene.
* Depth approximately agrees.
* Visibility/radii make sense.
* Representative gradients are finite.

Do not spend days pursuing numerical identity between the two implementations.

### 1D. Implement density control incrementally

In this order:

**Prune → Clone → Split → Combined schedule**

For each topology mutation, make sure the Adam optimizer state still corresponds correctly to the parameter tensors.

You need:

* screen-space gradient accumulation;
* clone;
* split;
* prune;
* scheduled densification.

Opacity reset is secondary. If reproducing it becomes messy, defer it until the core pipeline works.

### Phase 1 exit gate

Don't proceed until you can:

* train one small synthetic scene to overfit;
* train one real DTU scene;
* produce improving held-out renders;
* render depth;
* perform real clone/split/prune events without breaking training;
* record Gaussian count and loss.

You **do not** need production-quality resume/recovery/configuration machinery at this point.

---

# Phase 2 — Add minimal evaluation

Do not build an experiment platform.

You need:

* fixed camera IDs for train/test;
* one config file per run;
* seed;
* metrics output to JSON/CSV;
* one plotting script.

Track:

### Rendering

* PSNR
* LPIPS

SSIM is nice, but if it causes friction, add it later.

### Geometry

For the initial study:

* held-out depth AbsRel
* valid-depth / opacity coverage

Do **not** initially build the entire fused-surface DTU F-score pipeline.

That becomes worthwhile once you know you have an interesting result.

### Model behavior

Also log:

* number of Gaussians;
* training loss;
* training time if trivial.

---

# Phase 3 — Four-run sparse-view pilot

This is the key scope reduction.

Use **one DTU scene** and **one seed**.

Run only:

| Camera supervision | Densification |
| ------------------ | ------------- |
| 100%               | Baseline      |
| 100%               | Aggressive    |
| 10%                | Baseline      |
| 10%                | Aggressive    |

For `aggressive`, modify **only the clone/split gradient threshold**.

Keep fixed:

* initialization;
* optimizer;
* iterations;
* pruning;
* refinement schedule;
* resolution;
* opacity behavior.

That makes the comparison interpretable.

### Primary question

You're looking for an interaction like:

> At 100% camera coverage, increased densification is harmless or useful, while at 10% coverage it creates substantially more Gaussians and worsens held-out geometric/generalization performance.

Measure:

* train PSNR;
* held-out PSNR / LPIPS;
* train-test gap;
* held-out depth AbsRel;
* valid depth coverage;
* final Gaussian count.

### Decision Gate #1

#### If there is no meaningful difference

**Stop.**

Don't force the depth hypothesis.

Investigate why:

* perhaps densification isn't aggressive enough;
* perhaps 10% is still sufficiently constrained;
* perhaps the suspected effect simply doesn't exist.

A well-supported negative result is acceptable.

#### If there is a clear sparse-view effect

Proceed to Phase 4.

---

# Phase 4 — Add one depth regularizer

Now introduce exactly **one** method modification.

Use a frozen monocular depth model to generate depth once for each training image.

For every training camera:

1. Obtain predicted monocular inverse depth.
2. Project visible COLMAP points into the image.
3. Get their true COLMAP camera-space depths.
4. Fit scale + shift between monocular inverse depth and COLMAP inverse depth.
5. Reject obviously bad alignments.
6. Save the aligned depth prior.

During 3DGS training, render expected depth and use:

$$
L = L_{\mathrm{RGB}} + \lambda_d L_{\mathrm{depth}}
$$

with a robust masked depth loss.

Don't add:

* normal losses;
* smoothness;
* uncertainty;
* cross-view warping;
* multiple depth networks.

One intervention. One question.

---

# Phase 5 — One-condition intervention test

Go straight to whichever condition failed most clearly.

Most likely:

> **10% cameras + aggressive densification**

Compare:

**RGB-only**
vs.
**RGB + depth**

on:

* train/test RGB metrics;
* depth AbsRel;
* coverage;
* Gaussian count.

Use **one reasonable λ** initially.

### Decision Gate #2

If depth training is unstable or produces no interesting change, don't immediately launch hyperparameter sweeps.

Try at most **2–3 λ values**.

If it still doesn't help, write up the negative result.

If it improves held-out geometry without badly damaging rendering:

> **you've completed the core project.**

Everything after this is validation.

---

# Phase 6 — Make the result credible

Only once you have a result worth validating should you expand.

### First expansion

Run the important comparison on **two additional DTU scenes**:

> RGB-only vs RGB+depth under the sparse/aggressive condition.

Now you have three scenes.

### Second expansion

If behavior is consistent, add another **1–2 seeds only for the key comparison**.

Don't replicate every historical baseline condition.

### Third expansion

Now implement the stronger DTU geometry pipeline:

> rendered depths → back-project → fuse/filter → compare with GT scan

Report something like:

* F-score;
* Chamfer distance;

in addition to held-out depth error.

At this point you can make a much stronger claim about actual reconstructed geometry.

---

# Phase 7 — Only-if-useful controls

These are explicitly **stretch work**:

* 25% camera coverage;
* no-densification control;
* matched Gaussian-count comparison;
* more seeds;
* more scenes;
* normal error;
* alternative depth priors;
* composite aggressive densification schedules.

None should block finishing the project.

---

# Suggested execution order

### Days 1–2

COLMAP/camera pipeline + reprojection test.

### Days 3–4

Finish PyTorch renderer and expected depth.

### Days 5–6

CUDA sanity check + clone/prune/split.

### Day 7

One successful end-to-end DTU baseline.

At this point, **you already have a respectable upgraded 3DGS project.**

### Days 8–9

Evaluation + four-run sparse-view pilot.

At this point, **you already have an experimental 3D vision project.**

### Days 10–12

Depth generation/alignment + depth loss.

### Days 13–14

RGB-only vs depth intervention.

At this point, if the result is interesting, **stop feature development and write it up.**

Additional scenes/seeds happen afterward.

---

# What counts as success

### Minimum success

You correctly implement:

**COLMAP → 3D Gaussians → projection → compositing → optimization → adaptive density control**

and perform a controlled sparse-view experiment.

Already resume-worthy.

### Strong success

You show a real interaction between sparse supervision and densification, then test a geometry-aware depth regularizer.

This is the target.

### Excellent success

You validate the result across several scenes and demonstrate improvement using independent geometric ground truth.

That's where it starts looking unusually strong for an undergraduate portfolio project.

---

# The rule I'd keep at the top of your README

> **Never implement the next phase until the previous phase produces a complete, interpretable result.**

That is what prevents this from becoming a half-finished research system.

Your previous roadmap had all the right technical ideas.  This version changes the execution philosophy: **4 runs before 40, one metric before five, one scene before three, one intervention before ablations.**

That's the scope I'd commit to.