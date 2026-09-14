---
title: "Books and Reading Guide"
subtitle: "A Study Roadmap for Acoustic Engine RPM Estimation by Digital Signal Processing"
author: "Adebanji Oluwatimileyin Adelowo"
date: "2026"
toc: true
toc-depth: 3
number-sections: true
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

\newpage

# How to Use This Guide

## What this document is

This is a **study roadmap**, not a bibliography. Its companion,
*Acoustic Engine RPM Estimation via Digital Signal Processing* (referred to
throughout as **the technical document**), records what the
`aircraft-engine-monitor` system does, derives the theory behind it, and reports
what happens when it is run against real engine audio. That document cites 24
primary sources --- mostly journal papers, several of them from the 1920s to the
1970s. Papers are the right citation for a specific result. They are the wrong
place to *learn* a subject.

This guide fills that gap. Every book here was selected because it teaches
something the `aircraft-engine-monitor` codebase actually depends on, and each
entry states exactly which chapters to read, why, and which part of the technical
document or which line of the code they explain.

A book is included only if it satisfies at least one of these tests:

1. It is the standard teaching text for a topic the technical document derives
   (sampling, DFT/FFT, windowing, filter design, decimation).
2. It covers a topic the project *needs* and the technical document only surveys
   (order tracking, statistical estimation theory, multirate cascade design).
3. It is the textbook counterpart to a paper the technical document cites, so the
   reader can acquire the background before attempting the paper.

Books that are merely famous, or that cover adjacent fields without bearing on
this system, are excluded.

## The seven problems this project actually poses

The reading is organised around the seven technical problems the system raises.
Every book below is mapped to at least one of them.

| # | Problem | Where it appears in the code | Technical doc |
|---|---|---|---|
| P1 | Sampling a continuous acoustic field without aliasing | `SAMPLE_RATE = 44100` | Section 3 |
| P2 | Computing and interpreting a finite spectrum | `fftpack.fft`, `np.hanning` | Section 4 |
| P3 | Filtering and reducing the sample rate by 88 | `Decimation`, `scipy.signal.decimate` | Section 5 |
| P4 | Estimating a frequency, and knowing how well | `np.argmax` over the 10--200 Hz band | Sections 6, 8 |
| P5 | Mapping frequency to shaft speed through harmonic order | `EVENTS_PER_CYCLE = 4` | Section 7 |
| P6 | Running it concurrently and in real time | six daemon threads, bounded queue | Section 9 |
| P7 | Validating the result | three-level test ladder | Sections 11--12 |

**P5 is the one that matters most.** The technical document shows, with measured
spectra, that the system's dominant failure --- a 21.4% order-flip rate producing
50% RPM errors, and an exact factor-of-two error on the in-flight recording --- is
entirely a P5 failure. A reader with limited time should read for P4 and P5 first
(Brandt Ch. 12, Randall Ch. 3, Stoica & Moses Ch. 2, Kay Ch. 3) and treat
everything else as background.

## Classification scheme

Each entry carries three labels.

**Level** --- *Undergraduate*, *Graduate*, or *Research monograph*: the
mathematical preparation assumed.

**Role** --- *Foundational* (needed to understand the system at all),
*Supplementary* (deepens or broadens), or *Advanced* (needed only to implement the
improvements proposed in Section 14 of the technical document).

**Priority** --- *Core* (read this), *Reference* (consult this), or *Optional*
(read if the topic interests you).

\newpage

# Tier 0: Foundations

These two books assume nothing beyond calculus and complex numbers, and they are
where a reader without a signals background should start.

## Signals and Systems

**Oppenheim, A. V., Willsky, A. S., with Nawab, S. H.** --- *Signals and Systems*,
2nd edition. Prentice Hall / Pearson, Upper Saddle River NJ, 1996.
ISBN 978-0-13-814757-0.
Publisher page: <https://www.pearson.com/en-us/subject-catalog/p/signals-and-systems/P200000003155/9780138147570>

**Level:** Undergraduate. **Role:** Foundational. **Priority:** Core (if new to
the field).

**Why it matters to this project.** The technical document opens Section 3 by
modelling sampling as impulse-train modulation, then uses the
multiplication-convolution duality to derive spectral replication. That single
argument is the load-bearing step in the whole sampling theorem, and it is
developed here from scratch. The book's distinctive feature --- treating
continuous-time and discrete-time in parallel, chapter by chapter --- is exactly
the right structure for this project, where a *continuous* acoustic pressure field
is turned into a *discrete* sequence and the relationship between the two must be
understood precisely.

**Topics it supplies that appear in the technical document:** linearity and
time-invariance; convolution; the continuous-time and discrete-time Fourier
transforms; the multiplication-convolution duality; the sampling theorem;
aliasing; discrete-time processing of continuous-time signals; the effect of a
finite observation window.

**Chapters to study:**

- **Ch. 1--2** --- signals, systems, LTI properties, convolution. Skim if familiar.
- **Ch. 3--4** --- Fourier series and the continuous-time Fourier transform.
  Establishes the transform pairs used throughout Section 3 of the technical doc.
- **Ch. 5** --- the discrete-time Fourier transform. This is the $X(e^{j\omega})$
  of Section 4.1.
- **Ch. 7 --- essential.** Sampling. The impulse-train model, spectral
  replication, the sampling theorem, aliasing, and reconstruction. Section 3 of
  the technical document is a compressed retelling of this chapter applied to
  44,100 Hz microphone audio. Read it in full, including the treatment of
  discrete-time processing of continuous-time signals, which is precisely what
  the recorder-to-analyser path does.

**Connection to implementation.** Ch. 7 explains why `SAMPLE_RATE = 44100` is
harmless and why the 88-fold rate reduction in `Decimation` is *not* harmless
unless preceded by a filter.

**Relationship to the cited papers.** This is the textbook route into Nyquist
(1928) and Shannon (1949). Both papers are short and are worth reading afterwards,
but neither is a tutorial; Ch. 7 is.

## Understanding Digital Signal Processing

**Lyons, R. G.** --- *Understanding Digital Signal Processing*, 3rd edition.
Prentice Hall / Pearson, Upper Saddle River NJ, 2010. ISBN 978-0-13-702741-5.
984 pp.
Publisher page: <https://www.pearson.com/en-us/subject-catalog/p/Lyons-Understanding-Digital-Signal-Processing-3rd-Edition/P200000000443>

**Level:** Undergraduate / practitioner. **Role:** Foundational.
**Priority:** Core.

**Why it matters to this project.** Lyons is the most practical DSP book in
common use: it explains what actually happens to real data rather than what the
theory permits. For someone whose goal is to *fix* the peak-picker in
`rpi_test_final.py` rather than to pass an exam, this is the most immediately
useful book on the list. Its treatment of the picket-fence (scalloping) effect and
of FFT peak interpolation addresses, directly and concretely, the failure
mechanism identified in Section 4.4 of the technical document --- that the 0.23 dB
margin between engine orders 2 and 4 is smaller than the worst-case scalloping
loss of even a Hann window.

**Topics it supplies:** the DFT and its properties; DFT leakage and windowing;
the picket-fence effect; FFT structure and implementation; FIR and IIR design;
decimation and interpolation; multistage decimation; averaging; sample-rate
conversion; frequency estimation by interpolated FFT.

**Chapters to study:**

- **Ch. 2** --- periodic sampling and aliasing, with the clearest set of
  frequency-folding diagrams available anywhere.
- **Ch. 3 --- essential.** The DFT: leakage, windows, the picket-fence effect,
  zero-padding, and DFT resolution. Maps directly onto Section 4 of the technical
  document. The distinction Lyons draws between *resolution* (set by window
  length) and *the appearance of resolution* (set by zero-padding) is one that the
  technical document relies on but does not belabour.
- **Ch. 4** --- the FFT: radix-2 structure and the practical consequences of
  input length.
- **Ch. 5--6** --- FIR and IIR filters, including the Chebyshev response that
  `scipy.signal.decimate` actually uses.
- **Ch. 10 --- essential for P3.** Sample-rate conversion: decimation,
  interpolation, and **multistage** decimation. This is the chapter that justifies
  the technical document's recommendation (Section 14) to replace the
  out-of-specification $M = 88$ single stage with a $2 \times 4 \times 11$
  cascade, and it explains why the cascade is *cheaper* as well as better
  conditioned.
- **Ch. 13** --- a large collection of practical techniques, including
  **interpolated-FFT frequency estimation**, which is the concrete recipe for the
  sub-bin peak interpolation recommended in Section 14.

**Connection to implementation.** Ch. 10 and Ch. 13 between them contain the
implementations of two of the seven efficiency-and-robustness items in Section 14
of the technical document.

\newpage

# Tier 1: Core Digital Signal Processing

## Discrete-Time Signal Processing

**Oppenheim, A. V. and Schafer, R. W.** --- *Discrete-Time Signal Processing*,
3rd edition. Prentice Hall / Pearson, Upper Saddle River NJ, 2010.
ISBN 978-0-13-198842-2.
Publisher page: <https://www.pearson.com/en-us/subject-catalog/p/Oppenheim-Oppenheim-Discret-Signal-Process-c-3-3rd-Edition/P200000003226/9780131988422>

**Level:** Advanced undergraduate / graduate. **Role:** Foundational.
**Priority:** Core --- this is the single most important book on the list.

**Why it matters to this project.** This is the citation anchor of the technical
document. It is referenced as the standard authority for Sections 3, 4, and 5 ---
sampling, the DFT, and filtering and decimation --- which together are the entire
DSP content of the system. Where the technical document states a result without
derivation, the derivation is here.

**Topics it supplies:** discrete-time signals and LTI systems; the z-transform;
sampling of continuous-time signals and the reconstruction theorem; changing the
sampling rate by an integer factor (the exact derivation the technical document
compresses in Section 5.1); transform analysis of LTI systems; the discrete
Fourier transform and its relationship to the DTFT; circular versus linear
convolution; efficient DFT computation; Fourier analysis of signals using the DFT,
including windowing and the time-dependent (short-time) Fourier transform; FIR and
IIR filter design including the Butterworth and Chebyshev families.

**Chapters to study:**

- **Ch. 2** --- discrete-time signals and systems; the DTFT.
- **Ch. 3** --- the z-transform. Needed to read the filter-design chapters.
- **Ch. 4 --- essential.** Sampling of continuous-time signals. Its Section 4.6,
  *Changing the Sampling Rate Using Discrete-Time Processing*, contains the exact
  downsampling relation
  $X_d(e^{j\omega}) = \frac{1}{M}\sum_{i=0}^{M-1} X(e^{j(\omega - 2\pi i)/M})$
  quoted in Section 5.1 of the technical document, together with the derivation of
  the anti-aliasing condition that decimation must satisfy.
- **Ch. 7 --- essential for P3.** Filter design. Butterworth and Chebyshev
  approximations, including the maximally-flat condition that Section 5.2 of the
  technical document restates in modern form, and the equiripple Chebyshev type I
  response that `scipy.signal.decimate` actually instantiates as
  `cheby1(8, 0.05, 0.8/q)`.
- **Ch. 8 --- essential for P2.** The discrete Fourier transform. The DFT as
  sampled DTFT, Hermitian symmetry for real inputs (why the code writes
  `fft_data[:N//2]`), and the linear-versus-circular convolution distinction.
- **Ch. 9** --- computation of the DFT. Decimation-in-time and
  decimation-in-frequency FFT algorithms, and the generalisation to non-power-of-2
  lengths, which matters here because the decimated window is
  $2506 = 2 \times 7 \times 179$.
- **Ch. 10 --- essential for P2 and P4.** Fourier analysis of signals using the
  DFT. Windowing, resolution, the time-dependent Fourier transform (STFT), and
  spectrum analysis of sinusoidal signals. This is the theoretical home of the
  entire peak-finder.

**Connection to implementation.** Ch. 4.6 justifies `Decimation`; Ch. 7 explains
the filter inside `scipy.signal.decimate`; Ch. 8--10 justify
`FrequencyPeakFinder`.

**Relationship to the cited papers.** Ch. 9 is the textbook treatment of Cooley
and Tukey (1965); Ch. 10 is the textbook treatment of Harris (1978); Ch. 4 is the
textbook treatment of Nyquist (1928) and Shannon (1949). Read the relevant chapter
before the corresponding paper in every case.

## Digital Signal Processing: Principles, Algorithms, and Applications

**Proakis, J. G. and Manolakis, D. G.** --- *Digital Signal Processing:
Principles, Algorithms, and Applications*, 4th edition. Pearson / Prentice Hall,
Upper Saddle River NJ, 2006. ISBN 978-0-13-187374-2. 1104 pp.
Publisher page: <https://www.pearson.com/en-us/subject-catalog/p/digital-signal-processing/P200000003224>

**Level:** Advanced undergraduate / graduate. **Role:** Foundational.
**Priority:** Reference.

**Why it matters to this project.** Proakis and Manolakis is the natural companion
to Oppenheim and Schafer: it covers most of the same ground but is more
algorithm-oriented and, crucially for this project, devotes far more space to
**multirate signal processing** and to **power spectrum estimation** --- the two
topics where Oppenheim and Schafer is thinnest and where this project has its
sharpest needs (P3 and P4). Where the two books disagree in emphasis, this one is
usually closer to what an implementer needs.

**Topics it supplies:** discrete-time signals and systems; the z-transform; the
DFT and its efficient computation; implementation of discrete-time systems; FIR
and IIR filter design; **multirate digital signal processing** (decimation,
interpolation, rational-factor sample-rate conversion, polyphase structures,
multistage implementations); **power spectrum estimation** (periodogram, Bartlett,
Welch, Blackman-Tukey, and parametric methods).

**Chapters to study:**

- **Ch. 6--7** --- sampling, reconstruction, and the DFT, with a thorough
  treatment of DFT-based frequency analysis.
- **Ch. 8** --- efficient DFT computation.
- **Ch. 10** --- digital filter design, including the Butterworth and Chebyshev
  analogue prototypes and their bilinear transformation.
- **Ch. 11 --- essential for P3.** Multirate digital signal processing. Decimation
  by an integer factor, the anti-aliasing requirement, polyphase decomposition,
  and **multistage implementation of sampling-rate conversion**. This chapter is
  the direct theoretical justification for replacing the single-stage $M = 88$
  decimation.
- **Ch. 14 --- essential for P4.** Power spectrum estimation. The periodogram and
  its variance problem, Bartlett and Welch averaging, windowing effects, and the
  resolution-variance trade-off. This is the correct framework for the
  question the technical document raises in Section 12 --- how much of the
  chunk-to-chunk scatter is real speed variation and how much is estimator
  variance.

**Relationship to the cited papers.** Ch. 11 is the textbook treatment of
Crochiere and Rabiner (1981).

## Digital Signal Processing: An Introduction

**Sundararajan, D.** --- *Digital Signal Processing: An Introduction*. Springer,
Cham, 2021. ISBN 978-3-030-62367-8 (hardcover); 978-3-030-62370-8 (softcover);
978-3-030-62368-5 (eBook).
DOI: [10.1007/978-3-030-62368-5](https://doi.org/10.1007/978-3-030-62368-5)
(A revised Springer edition appeared in 2024, ISBN 978-3-031-56739-1.)

**Level:** Senior undergraduate / first-year graduate.
**Role:** Foundational. **Priority:** Core.

**Why it matters to this project.** This is the most efficient single book on the
list: it is concise, modern, and it covers **every** DSP topic this project uses
--- DTFT, DFT, fast DFT algorithms, FIR and IIR design, multirate DSP, and power
spectral density estimation --- in one volume of manageable length, with Matlab
support material. Where Oppenheim and Schafer is the reference to consult and
Lyons is the book to reach for at the bench, Sundararajan is the book to read
cover to cover in order to hold the whole subject in mind at once.

Two of its chapters map one-to-one onto this project's weakest points.

**Topics it supplies:** discrete signals and systems; the DFT and its inverse;
the DTFT; fast algorithms for computing the DFT; the z-transform; FIR and IIR
filter design; **multirate digital signal processing** (downsamplers, upsamplers,
decimation, interpolation); **power spectral density estimation**; an introduction
to the discrete wavelet transform.

**Chapters to study:**

- **The DFT chapter** --- derivation of the DFT and its inverse, worked examples.
  Section 4.1 of the technical document in longer form.
- **The Fast DFT Algorithms chapter** --- practically efficient algorithms for
  real- and complex-valued data based on the classical divide-and-conquer
  strategy. Note especially the treatment of *real-valued* data: the technical
  document identifies the use of a full complex FFT on real microphone audio as an
  available factor-of-two saving (Sections 4.4 and 14.7), and this chapter gives
  the algorithms that realise it.
- **Multirate Digital Signal Processing (Ch. 8) --- essential for P3.** The
  downsampler and upsampler as first-class components, and the principles of
  changing sampling rate to suit the task. This is `Decimation` from first
  principles.
  DOI: [10.1007/978-3-030-62368-5\_8](https://doi.org/10.1007/978-3-030-62368-5_8)
- **The Power Spectral Density chapter --- essential for P4.** Estimating the PSD
  of *periodic signals buried in noise* observed only over a finite interval,
  using the DFT with sectionalising, windowing and averaging, or via the
  autocorrelation. This is a precise formal statement of this project's actual
  measurement problem: the engine orders are periodic components buried in cabin
  noise, observed in five-second sections.
- **The filter design chapters** --- detailed coverage of FIR and IIR design,
  which is where the Chebyshev anti-aliasing filter belongs.

**Connection to implementation.** The multirate chapter explains `Decimation`; the
fast-DFT chapter explains how to make `FrequencyPeakFinder` twice as fast; the PSD
chapter reframes the whole estimator correctly.

## A Course in Digital Signal Processing

**Porat, B.** --- *A Course in Digital Signal Processing*. John Wiley and Sons,
New York, 1996 (frequently catalogued as 1997). ISBN 978-0-471-14961-3. 632 pp.
Publisher page: <https://www.wiley.com/en-us/A+Course+in+Digital+Signal+Processing-p-9780471149613>

**Level:** Advanced undergraduate / graduate. **Role:** Foundational.
**Priority:** Reference.

**Why it matters to this project.** Porat is unusual among DSP textbooks in giving
**practical spectral analysis** a full chapter of its own, covering the use of
windows for spectral analysis, the analysis of sinusoidal signals, and the effect
of noise. That is exactly the triple this project lives on: the signal is a sum of
sinusoidal components (engine orders), it is windowed (Hann), and it is embedded
in noise. Most books treat these three topics in three separate places; Porat
treats them together, which is how they are actually encountered.

It also carries a chapter on the **analysis and modelling of random signals**,
giving the statistical vocabulary that the technical document's discussion of
estimator variance requires.

**Topics it supplies:** review of frequency-domain analysis; sampling and
reconstruction; the DFT; the FFT; **practical spectral analysis** (windows,
sinusoidal signal analysis, noise); z-transforms and difference equations;
introduction to digital filters; FIR filters; IIR filters; digital filter
realisation and implementation; **multirate signal processing**; analysis and
modelling of random signals; DSP applications.

**Chapters to study:**

- **Sampling and Reconstruction** --- P1.
- **The Discrete Fourier Transform** and **The Fast Fourier Transform** --- P2.
- **Practical Spectral Analysis --- essential for P2 and P4.** Read this before
  attempting Harris (1978): it establishes why one needs a window catalogue before
  presenting one.
- **Multirate Signal Processing** --- P3.
- **Analysis and Modeling of Random Signals** --- P4, and the vocabulary for
  Section 12's variance discussion.

**Relationship to the cited papers.** The *Practical Spectral Analysis* chapter is
the ideal preparation for Harris (1978) and for Rife and Boorstyn (1974).

\newpage

# Tier 2: Multirate Signal Processing

The system reduces its sample rate by a factor of 88 in a single stage --- nearly
seven times SciPy's documented recommended maximum of 13. Section 5.5 of the
technical document shows the deployed configuration is empirically sound but
fragile, and Section 14 recommends a multistage cascade. These three books are why.

## Multirate Digital Signal Processing

**Crochiere, R. E. and Rabiner, L. R.** --- *Multirate Digital Signal Processing*.
Prentice-Hall, Englewood Cliffs NJ, 1983. (Prentice-Hall Signal Processing
Series.)

**Level:** Graduate. **Role:** Foundational for P3. **Priority:** Reference.

**Why it matters to this project.** This is the monograph the technical document
cites for decimation theory (Section 5.1), by the authors of the tutorial review
it also cites. It is the origin text for the systematic treatment of
integer-factor decimation, the aliasing condition, and --- most relevantly here
--- the design of **multistage** decimators, including the optimisation of how to
split a large factor across stages.

**Topics it supplies:** sampling-rate conversion by integer and rational factors;
the anti-aliasing requirement and filter specification for decimation; polyphase
structures; **multistage decimator design and the choice of stage factors**;
computational-cost analysis of multirate structures.

**Sections to study:** the chapters on integer-factor decimation and on multistage
implementations. The cost analysis showing that a cascade requires fewer
multiplications per input sample than a single stage is the quantitative
justification for the technical document's recommendation to replace $M = 88$ with
$2 \times 4 \times 11$.

**Relationship to the cited papers.** This book and Crochiere and Rabiner's 1981
*Proceedings of the IEEE* tutorial review are companions; the review is the
20-page summary, the book the full treatment. If only one is available, the review
is sufficient for this project's needs.

## Multirate Systems and Filter Banks

**Vaidyanathan, P. P.** --- *Multirate Systems and Filter Banks*. Prentice-Hall,
Englewood Cliffs NJ, 1993. ISBN 978-0-13-605718-5.

**Level:** Graduate / research monograph. **Role:** Advanced.
**Priority:** Optional.

**Why it matters to this project.** Vaidyanathan is the definitive modern
treatment of multirate theory, and it is the right book if the reader wants the
*structural* view: polyphase decomposition, noble identities, and perfect
reconstruction. For this project specifically, its value is in the polyphase
material --- a polyphase decimator computes only the samples that survive
downsampling, so the anti-aliasing filter costs $1/M$ of what the direct form
costs. At $M = 88$ that is a very large factor, and it is the single biggest
available efficiency gain in the decimation stage after multistaging.

It is also the reference to reach for if the tacholess order-tracking work
proposed in Section 14 leads toward filter-bank or wavelet-based time-frequency
decomposition.

**Chapters to study:** the chapters on multirate fundamentals and **polyphase
representations**; the noble identities. The filter-bank and wavelet material
beyond that is not needed for this project unless the ridge-tracking direction is
pursued.

**Caution:** this is a demanding book. Read Proakis Ch. 11 or Lyons Ch. 10 first.

## Multirate Signal Processing for Communication Systems

**harris, f. j. (Fredric J. Harris)** --- *Multirate Signal Processing for
Communication Systems*. Prentice Hall PTR, Upper Saddle River NJ, 2004.
ISBN 978-0-13-146511-4. (A 2nd edition is published by River Publishers /
Taylor and Francis,
DOI: [10.1201/9781003338888](https://doi.org/10.1201/9781003338888).)

**Level:** Graduate / practitioner. **Role:** Supplementary.
**Priority:** Optional.

**Why it matters to this project.** This is by the same author as the 1978
windowing paper cited throughout Section 4 of the technical document, but it is an
entirely different and much later work --- an applied multirate design book, not a
window catalogue. It is included because it is the most *design-oriented* of the
three multirate books: it is organised around how to build cascade and
multiple-stage filter structures that meet a specification, which is precisely the
task the technical document leaves open in Section 14.3.

**Topics it supplies:** resampling fundamentals; window-method and equiripple
filter design for resampling; **polyphase FIR filters and channelizers (Ch. 6)**;
**polyphase interpolators and arbitrary sample-rate change (Ch. 7)**; cascade and
multiple-stage filter structures.

**Chapters to study:** Ch. 6 and Ch. 7 for polyphase structures; the chapters on
cascaded and multistage designs for the $88 = 2 \times 4 \times 11$ problem.

**Relationship to the cited papers.** Reading harris (2004) alongside harris
(1978) is instructive: the same author's window analysis reappears as a *design
tool* for the resampling filters, showing how the leakage and sidelobe figures
tabulated in 1978 are used in practice.

\newpage

# Tier 3: Spectral Analysis and Estimation Theory

This tier addresses P4 --- estimating a frequency and knowing how well. The
technical document's central negative result is an estimation failure, and these
two books provide the framework in which that failure is properly described.

## Spectral Analysis of Signals

**Stoica, P. and Moses, R. L.** --- *Spectral Analysis of Signals*. Pearson
Prentice Hall, Upper Saddle River NJ, 2005. ISBN 978-0-13-113956-5. 452 pp.
(Expanded from *Introduction to Spectral Analysis*, Prentice-Hall, 1997.)
Author's copy: <https://user.it.uu.se/~ps/SAS-new.pdf>

**Level:** Graduate. **Role:** Foundational for P4. **Priority:** Core.

**Why it matters to this project.** The system's estimator is a periodogram peak
--- the most basic method in this book --- and this book is where its properties,
limitations, and alternatives are laid out rigorously. Two of its themes bear
directly on the technical document's findings.

First, the **bias-variance and resolution trade-offs** of nonparametric spectral
estimation give the correct language for Section 12's observation that the
chunk-to-chunk scatter (15.7 rpm) is five times the bin quantisation (3.0 rpm) and
must therefore have a different cause.

Second, its treatment of **parametric and subspace methods** for line spectra
(Prony, MUSIC, ESPRIT, and related high-resolution techniques) is the natural
next step for a signal that genuinely *is* a sum of sinusoids in noise --- which
an engine order spectrum genuinely is. These methods estimate the number and
frequencies of the components jointly, which is structurally the right answer to
the order-assignment problem that defeats the argmax.

**Topics it supplies:** nonparametric methods (periodogram, Blackman-Tukey,
windowing, Bartlett and Welch averaging, refined methods); parametric methods for
rational spectra (AR, ARMA); **parametric methods for line spectra**; **subspace
methods (MUSIC, ESPRIT, min-norm)**; filter-bank methods (Capon); spatial
analysis.

**Chapters to study:**

- **Ch. 1** --- basic definitions; energy and power spectral density.
- **Ch. 2 --- essential.** Nonparametric methods. The periodogram, its bias and
  variance, windowing, and the averaged and smoothed variants. Read the discussion
  of resolution alongside Section 4.2 of the technical document.
- **Ch. 4 --- essential for the improvements.** Parametric methods for line
  spectra: the sinusoids-in-noise model, nonlinear least squares, and the
  high-order Yule-Walker approach.
- **Ch. 5** --- subspace methods. MUSIC and ESPRIT. Ambitious for a Pi Zero at
  full spectrum length, but entirely feasible on a short decimated record.

**Connection to implementation.** Ch. 2 describes what `FrequencyPeakFinder`
already does; Ch. 4--5 describe what should replace it if the harmonic-template
approach of Section 14.1 proves insufficient.

## Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory

**Kay, S. M.** --- *Fundamentals of Statistical Signal Processing, Volume I:
Estimation Theory*. Prentice Hall PTR, Upper Saddle River NJ, 1993.
ISBN 978-0-13-345711-7.
Publisher page: <https://www.pearson.com/en-us/subject-catalog/p/fundamentals-of-statistical-processing-estimation-theory-volume-1/P200000009271/9780133457117>

**Level:** Graduate. **Role:** Foundational for P4. **Priority:** Core.

**Why it matters to this project.** Section 6.2 of the technical document quotes
the Cramer-Rao lower bound for single-tone frequency estimation and its $N^{-3}$
variance behaviour, attributing it to Rife and Boorstyn (1974). Kay is where that
bound is *derived*, along with the whole apparatus needed to reason about it:
what an estimator is, what bias and variance mean, when a bound is attainable, and
what the maximum-likelihood estimator does. The technical document's corrected
discussion of the **threshold effect** --- the abrupt departure from the bound when
the periodogram maximum lands on the wrong peak --- also belongs to this framework.

Kay's own worked example of sinusoidal parameter estimation appears repeatedly
through the book, so the reader gets the frequency-estimation problem developed
from several directions.

**Topics it supplies:** the estimation problem; minimum variance unbiased
estimation; the **Cramer-Rao lower bound**; linear models; best linear unbiased
estimators; **maximum likelihood estimation** and its asymptotic properties;
least squares; the method of moments; Bayesian estimation; the **Kalman filter**.

**Chapters to study:**

- **Ch. 1--2** --- the estimation problem, unbiasedness, minimum variance.
- **Ch. 3 --- essential.** The Cramer-Rao lower bound. Kay's running example of
  frequency estimation for a sinusoid in noise is the exact problem of Section
  6.2. This chapter is the prerequisite for reading Rife and Boorstyn (1974) with
  understanding.
- **Ch. 7 --- essential.** Maximum likelihood estimation. Establishes that the
  periodogram maximiser is the ML estimator for a single tone in Gaussian noise,
  which is the precise statement the technical document had to correct --- the
  *coarse DFT peak* is only a bin-quantised approximation to it.
- **Ch. 13 --- essential for Section 14.3.** The Kalman filter. State-space
  models, prediction, and the recursive update. This is the mathematics behind the
  constant-acceleration RPM tracker proposed in Section 14.3 of the technical
  document, and behind the Vold-Kalman order tracking filter cited there.

**Connection to implementation.** Ch. 13 is the direct prerequisite for
implementing the recommended
$[\Omega_k, \dot{\Omega}_k]^\top$ state-space smoother that would reject every
order-flip outlier measured in Section 12.

**Relationship to the cited papers.** Ch. 3 for Rife and Boorstyn (1974);
Ch. 13 for Vold and Leuridan (1995).

\newpage

# Tier 4: Rotating Machinery, Order Analysis, and Condition Monitoring

This is the application domain, and it is where the project's actual failure
lives. **A reader with time for only two books on this entire list should read
Brandt Ch. 12 and Randall Ch. 3.**

## Noise and Vibration Analysis: Signal Analysis and Experimental Procedures

**Brandt, A.** --- *Noise and Vibration Analysis: Signal Analysis and
Experimental Procedures*. John Wiley and Sons, Chichester, 2011.
ISBN 978-0-470-74644-8.
DOI: [10.1002/9781118962176](https://doi.org/10.1002/9781118962176)
Companion site: <https://www.wiley.com/go/brandt> (Matlab/ABRAVIBE toolbox)

**Level:** Advanced undergraduate / practitioner. **Role:** Foundational for P5.
**Priority:** Core --- the best-matched single book on this list.

**Why it matters to this project.** Brandt is the book this project should have
been written with open on the desk. It is a signal-processing text written
specifically for noise and vibration engineers, so it covers the *same DSP* as the
Tier 1 books but consistently in terms of the *same application*: rotating
machinery, measured with microphones and accelerometers, analysed with the DFT.
Every practical pitfall the technical document rediscovered empirically ---
leakage, the picket-fence effect, window choice, order versus frequency --- is
treated here in the vocabulary of the domain.

**Chapter 12 is the single most relevant chapter in this entire guide.** It treats
rotating machinery analysis directly: tachometer signals, **order analysis**,
**order tracking**, and **synchronous (angle-domain) sampling**. It is the
textbook answer to P5, which is the problem `EVENTS_PER_CYCLE = 4` fails to solve.

**Topics it supplies:** noise and vibration fundamentals; transducers
(accelerometers, microphones) and their calibration --- directly relevant to the
`Sensors/` ADXL345 interface; dynamic signals and systems; time-data analysis;
statistics and random processes; **frequency analysis of signals: the DFT, the
FFT, leakage, the picket-fence effect, time windows** (Ch. 8--10); measurement
and analysis systems (Ch. 11); **rotating machinery analysis, order analysis, order
tracking, synchronous sampling (Ch. 12)**; experimental modal analysis.

**Chapters to study:**

- **Ch. 2--3** --- transducers and measurement chains. Read if the microphone or
  the accelerometer path is to be improved.
- **Ch. 8--10 --- essential for P2.** Frequency analysis: the DFT and FFT,
  leakage, the picket-fence effect, and time windows. The picket-fence
  (scalloping) treatment is the domain-specific version of the correction made in
  Section 4.4 of the technical document, where the 0.23 dB inter-order margin is
  shown to be smaller than the worst-case scalloping loss of a Hann window.
- **Ch. 11** --- measurement and analysis systems; specifications for a good test
  system. Useful context for the Level-3 hardware test.
- **Ch. 12 --- essential, read first.** Rotating machinery analysis. Tachometers,
  order analysis, order tracking, synchronous sampling. This chapter explains what
  an order *is*, why order spectra are the correct representation for a
  variable-speed machine, and how order tracking is performed with and without a
  tachometer. It is the direct background for Sections 6 and 7 of the technical
  document.

**Connection to implementation.** Ch. 12 is the theory that `EVENTS_PER_CYCLE`
approximates with a single constant. Reading it makes immediately clear why a
fixed order index cannot work across operating points, and what the standard
alternatives are.

**Relationship to the cited papers.** Ch. 12 is the textbook route into Vold and
Leuridan (1995) and into the tacholess order-tracking reviews by Lu et al. (2019)
and Peeters et al. (2019).

## Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications

**Randall, R. B.** --- *Vibration-based Condition Monitoring: Industrial,
Aerospace and Automotive Applications*. John Wiley and Sons, Chichester, 2011.
ISBN 978-0-470-74785-8.
DOI: [10.1002/9780470977668](https://doi.org/10.1002/9780470977668)
Companion site: <https://www.wiley.com/go/randall> (exercises, data sets, Matlab code)

**Level:** Graduate / practitioner. **Role:** Foundational for P5.
**Priority:** Core.

**Why it matters to this project.** This is the monograph the technical document
cites as the standard reference for the machine-condition-monitoring perspective
(Section 6). Where Brandt teaches the measurement and analysis, Randall teaches
the **diagnosis**: what the order spectrum of a machine *means*, how a
reciprocating engine's signature differs from a gearbox's or a bearing's, and what
each spectral family indicates about condition.

Crucially for this project, Randall devotes explicit attention to **reciprocating
machines**, including the half-order and firing-order structure that Section 7 of
the technical document derives from kinematics and then confirms in measured data.
The measured order table for `1000_rpm.wav` --- dominant even orders 2, 4, 6, 8
built on a half-order family at 8.368 Hz, with moderate odd half-orders indicating
cylinder imbalance --- is a textbook example of exactly what Randall describes.

**Topics it supplies:** vibration signals from rotating and reciprocating
machines; signal processing for condition monitoring (spectrum analysis,
**cepstrum analysis**, **envelope/demodulation analysis**, time-frequency
analysis, **order tracking**); fault detection in gears, bearings, and
**reciprocating machines**; separation of signal sources; **cyclostationary**
methods; diagnostic case studies including aerospace and automotive.

**Chapters to study:**

- **Ch. 2** --- vibration signals from machines. Deterministic and random
  components; the periodic families a machine produces.
- **Ch. 3 --- essential for P5.** Signal processing techniques. Spectrum analysis,
  **cepstrum analysis**, envelope analysis, and order tracking. The **cepstrum**
  material deserves particular attention: the cepstrum collapses a uniformly
  spaced harmonic family to a single peak at the reciprocal of its spacing, which
  is exactly the quantity this project needs and exactly the technique the
  technical document proposes in Section 14.2 as an independent cross-check on the
  harmonic-template estimator.
- **The chapter on reciprocating machines --- essential.** Firing order,
  half-order components, cylinder-to-cylinder variation, and the interpretation of
  the resulting spectra. This is the domain authority for the entire order-model
  argument of Section 7.
- **The bearing and gear diagnostic chapters** --- not needed for RPM estimation,
  but they are the destination if the project extends from *speed measurement* to
  *condition monitoring*, which is its stated long-term motivation (Section 1.1).

**Relationship to the cited papers.** This book is by the author of Randall and
Antoni (2011), the bearing-diagnostics tutorial cited in Section 6.5, and it
develops the cyclostationary framework of Antoni (2009) cited in the same section.
Read the book chapters before either paper.

\newpage

# Tier 5: Concurrency and Real-Time Embedded Systems

This tier addresses P6. The technical document is explicit that this part of the
system is competent engineering rather than research, so the reading here is
deliberately narrow: one book that covers the exact patterns used, and one that
supplies the real-time context.

## Foundations of Multithreaded, Parallel, and Distributed Programming

**Andrews, G. R.** --- *Foundations of Multithreaded, Parallel, and Distributed
Programming*. Addison-Wesley, Reading MA, 2000. ISBN 978-0-201-35752-3.
Author's companion site: <https://raptor.cs.arizona.edu/~greg/mpdbook>

**Level:** Advanced undergraduate / graduate. **Role:** Foundational for P6.
**Priority:** Core.

**Why it matters to this project.** The architecture in Section 9 of the technical
document is a bounded-buffer producer-consumer system with two mutexes, and this
book covers exactly that, in exactly that vocabulary, with the same worked
problems. The match is unusually close:

- **Ch. 4, Semaphores** --- includes sections on *Critical Sections for Mutual
  Exclusion*, *Producers and Consumers*, *Bounded Buffers for Resource Counting*,
  and a **Pthreads case study**. The `analyze_queue = queue.Queue(maxsize=10)`
  between `recorder` and `analyzer` is the bounded-buffer problem verbatim, and
  `uart_lock` and `init_lock` are the mutual-exclusion problem verbatim.
- **Ch. 5, Monitors** --- introduces condition variables through the bounded
  buffer, readers/writers, and an **interval timer**. Python's `queue.Queue` is
  implemented as a monitor over a deque with condition variables, so this chapter
  describes the actual mechanism the code relies on. It is also the chapter that
  supplies the tool for the improvement the technical document suggests in
  Section 9.3: a `collections.deque(maxlen=10)` guarded by a `Condition` would
  express the drop-oldest overflow policy directly, rather than by the current
  non-atomic `get_nowait()`-then-`put()` sequence.

**Topics it supplies:** processes and synchronisation; locks and barriers;
**semaphores**; **monitors and condition variables**; the producer-consumer and
bounded-buffer problems; readers-writers; deadlock; message passing; performance.

**Chapters to study:** Ch. 1--2 for the concurrency model and the correctness
conditions; **Ch. 4** and **Ch. 5** in full; the deadlock material for the
check-then-act hazard identified in Section 9.4 of the technical document.

**Relationship to the cited papers.** This is the textbook route into Dijkstra
(1968), *Cooperating Sequential Processes*, which the technical document cites as
the origin of the semaphore formulation of the bounded-buffer problem. Dijkstra's
paper is readable but terse; Andrews Ch. 4 is the guided version.

## Real-Time Systems: Design Principles for Distributed Embedded Applications

**Kopetz, H.** --- *Real-Time Systems: Design Principles for Distributed Embedded
Applications*, 2nd edition. Springer, New York, 2011. ISBN 978-1-4419-8236-0.
(A later Springer edition is available at
DOI: [10.1007/978-3-031-11992-7](https://doi.org/10.1007/978-3-031-11992-7).)

**Level:** Graduate / practitioner. **Role:** Supplementary. **Priority:**
Optional.

**Why it matters to this project --- with a caveat.** Kopetz is about *hard*
real-time systems, in which missing a deadline is a failure. This project is
**soft** real-time: the constraint identified in Section 2.1 of the technical
document is throughput (keep up with the microphone) rather than deadline
guarantees, and the drop-oldest queue policy is an explicit decision to degrade
rather than to guarantee. So Kopetz should be read as *context and contrast*, not
as a prescription. Read this way it is genuinely valuable, because it makes the
soft/hard distinction sharp and shows what this system deliberately is not doing.

Two specific topics do transfer directly:

- **Global time and clock synchronisation.** The system sends a Time-sync packet
  (`START_BYTE_SYNC = 0xAB`) every 60 seconds so the ESP32 can relate its own
  clock to the Pi's, and every Data packet carries an absolute millisecond
  timestamp. Kopetz's treatment of global time, clock drift, and internal versus
  external synchronisation is the theory behind that design, and explains what the
  60-second interval is actually buying.
- **Temporal accuracy of real-time data.** Kopetz formalises the idea that an
  observation has a validity interval. The technical document makes exactly this
  point in Section 10.5: a transmitted RPM value covers the interval
  `[record_timestamp, record_timestamp + 5s]` and should be plotted as such.

**Chapters to study:** the chapters on the real-time system model and temporal
accuracy; **global time and clock synchronisation**; the treatment of real-time
communication protocols, for perspective on the UART framing design of Section 9.5.

\newpage

# Tier 6: Supplementary --- Fourier Theory and Audio Practice

## The Fourier Transform and Its Applications

**Bracewell, R. N.** --- *The Fourier Transform and Its Applications*, 3rd
edition. McGraw-Hill, New York, 2000. ISBN 978-0-07-303938-1. 624 pp.

**Level:** Undergraduate / graduate. **Role:** Supplementary.
**Priority:** Optional.

**Why it matters to this project.** Section 3 of the technical document leans on
several continuous-domain Fourier facts: that the transform of an impulse train is
an impulse train, that multiplication in time is convolution in frequency, and
that the transform of a rectangular window is a sinc. Bracewell is the classic
source for all of these, and treats the impulse-train ("shah" or comb) function
with more care than any DSP textbook does --- which matters, because the entire
sampling argument rests on it.

Its extensive tables of transform pairs and its treatment of convolution make it a
useful desk reference even for a reader who never studies it systematically.

**Chapters to study:** the chapters on the impulse symbol and the shah function;
convolution; the sampling and interpolation material; the tables of transform
pairs.

**Relationship to the cited papers.** Preparation for the continuous-domain half
of Shannon (1949).

## Fundamentals of Music Processing: Using Python and Jupyter Notebooks

**Müller, M.** --- *Fundamentals of Music Processing: Using Python and Jupyter
Notebooks*, 2nd edition. Springer, Cham, 2021. ISBN 978-3-030-69807-2.
(1st edition: *Fundamentals of Music Processing: Audio, Analysis, Algorithms,
Applications*, Springer, 2015,
DOI: [10.1007/978-3-319-21945-5](https://doi.org/10.1007/978-3-319-21945-5).)
Companion site: <https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html>
(open FMP notebooks); `libfmp` Python package.

**Level:** Undergraduate / graduate. **Role:** Supplementary.
**Priority:** Optional.

**Why it matters to this project --- and its limits.** The subject matter is music,
not engines, and it is included for one specific reason: it is the best available
treatment of **applied audio spectral analysis in Python**, and it is the only book
on this list whose stack matches this project's exactly (NumPy, SciPy, Jupyter,
with runnable notebooks for every figure).

The transferable material is real. Chapter 2 develops the **STFT and the
spectrogram** carefully, including window choice, hop size, and the
resolution trade-off --- which is precisely the machinery the technical document
proposes in Section 14.4 for ridge-based tacholess order tracking, and precisely
what the unused `STFT` class in
`audio_src/FeatureEngineering/Spectrograms/Fourier.py` wraps. Chapter 3's
**log-frequency and chroma** representations are, structurally, order-analysis by
another name: a mapping from linear frequency onto a scale where harmonically
related components align. Chapter 6's **onset detection** is spectral-flux
analysis of impulsive events, which is what combustion pulses are.

Do not read it for the engine content --- there is none. Read it if the goal is to
prototype STFT-based tracking quickly in the same language the system is written
in.

**Chapters to study:** **Ch. 2** (Fourier analysis of signals; STFT and
spectrogram) --- essential if pursuing Section 14.4; Ch. 3 (log-frequency
representations); Ch. 6 (onset detection, spectral flux, novelty functions).

\newpage

# Recommended Reading Order

## The short path (P4 and P5 only)

For a reader who wants to fix the system's actual defect and nothing else. Roughly
two weeks of evenings.

1. **Brandt, Ch. 12** --- order analysis and order tracking. What an order is, and
   why a fixed order index cannot work.
2. **Randall, Ch. 3** --- signal processing for condition monitoring, especially
   **cepstrum analysis**.
3. **Randall, reciprocating machines chapter** --- firing order and half-order
   structure. Confirms the order model of Section 7.
4. **Lyons, Ch. 13** --- interpolated-FFT frequency estimation. Removes the
   scalloping bias.
5. **Stoica & Moses, Ch. 2** --- what a periodogram peak actually estimates.

At this point the reader can implement the harmonic-template estimator of
Section 14.1 and the cepstral cross-check of Section 14.2 with confidence.

## The full path

**Stage 1 --- Foundations (if new to signals).**
Oppenheim, Willsky & Nawab Ch. 1--5, then **Ch. 7 (sampling)**.
Lyons Ch. 2--4 in parallel, for the practical view.

**Stage 2 --- Core DSP.**
Sundararajan, cover to cover --- the most efficient single pass over the whole
subject. Then Oppenheim & Schafer **Ch. 4, 7, 8, 9, 10** as the authoritative
treatment of the same material. Consult Porat's *Practical Spectral Analysis*
chapter here.

**Stage 3 --- Read the primary papers.**
Now, and not before, read Nyquist (1928), Shannon (1949), Cooley & Tukey (1965),
and Harris (1978). Each is short; each is far easier with Stage 2 behind you.

**Stage 4 --- Multirate.**
Proakis & Manolakis **Ch. 11**, then Lyons **Ch. 10** for the multistage design
procedure. Crochiere & Rabiner (1981 review, or the 1983 book) for the origin
treatment. Vaidyanathan's polyphase chapters and harris (2004) Ch. 6--7 if
implementing the cascade.

**Stage 5 --- Estimation theory.**
Kay **Ch. 1--3** (Cramer-Rao bound), then **Ch. 7** (maximum likelihood). Read
Rife & Boorstyn (1974) at this point. Then Stoica & Moses **Ch. 2 and Ch. 4**.

**Stage 6 --- The application domain.**
Brandt **Ch. 8--10, then Ch. 12**. Randall **Ch. 2--3 and the reciprocating
machines chapter**. Then the papers: Vold & Leuridan (1995), Lu et al. (2019),
Peeters et al. (2019), Randall & Antoni (2011), Antoni (2009), Shan et al. (2020).

**Stage 7 --- Tracking.**
Kay **Ch. 13** (Kalman filter). This closes the loop back to Vold & Leuridan and
enables the tracker proposed in Section 14.3.

**Stage 8 --- Systems.**
Andrews **Ch. 4--5**, then Dijkstra (1968). Kopetz selectively, for the global-time
material behind the sync packet.

\newpage

# Cross-Reference Tables

## Technical document section to book chapter

| Technical doc section | Topic | Primary reading | Secondary |
|---|---|---|---|
| 3. Sampling and aliasing | Nyquist, aliasing, decimation rationale | Oppenheim/Willsky Ch. 7; Oppenheim & Schafer Ch. 4 | Lyons Ch. 2; Bracewell (shah function) |
| 4.1--4.2 DFT, resolution | DFT, $\Delta f = 1/T_{\text{win}}$ | Oppenheim & Schafer Ch. 8 | Sundararajan (DFT ch.); Porat (DFT ch.) |
| 4.3 FFT | Cooley-Tukey, radix-2, composite $N$ | Oppenheim & Schafer Ch. 9 | Sundararajan (Fast DFT ch.); Lyons Ch. 4 |
| 4.4 Leakage, windows | Hann window, sidelobes, scalloping | Oppenheim & Schafer Ch. 10; Porat (Practical Spectral Analysis) | Brandt Ch. 9--10; Lyons Ch. 3 |
| 4.5 STFT | Spectrogram, hop, overlap | Oppenheim & Schafer Ch. 10.3 | Müller Ch. 2 |
| 5.1 Decimation theory | Aliasing condition, filter-then-downsample | Oppenheim & Schafer Ch. 4.6; Proakis Ch. 11 | Crochiere & Rabiner; Lyons Ch. 10 |
| 5.2 Butterworth | Maximally flat response, poles | Oppenheim & Schafer Ch. 7 | Proakis Ch. 10 |
| 5.4--5.5 Chebyshev, $M=88$ | Anti-alias design, multistage | Lyons Ch. 10; Proakis Ch. 11 | Vaidyanathan (polyphase); harris (2004) |
| 6. Order estimation survey | Orders, order tracking, cepstrum | **Brandt Ch. 12**; **Randall Ch. 3** | Stoica & Moses Ch. 4--5 |
| 6.2 CRLB, threshold effect | Estimator variance, ML | **Kay Ch. 3, Ch. 7** | Stoica & Moses Ch. 4 |
| 7. Harmonic RPM model | Firing order, half-orders | **Randall (reciprocating ch.)**; Brandt Ch. 12 | --- |
| 8. Peak detection | Argmax, interpolation, tracking | Lyons Ch. 13; Stoica & Moses Ch. 2 | Kay Ch. 3 |
| 9.1--9.4 Threads, queue, locks | Producer-consumer, mutexes, monitors | **Andrews Ch. 4--5** | Kopetz (context) |
| 9.5--9.8 Protocol, CRC, ACK, sync | Framing, global time | Kopetz (global time) | --- |
| 10. Implementation | Whole pipeline | Sundararajan; Lyons | Brandt Ch. 8--11 |
| 11--12. Validation, results | PSD estimation, variance | Proakis Ch. 14; Stoica & Moses Ch. 2 | Sundararajan (PSD ch.) |
| 13. Limitations | Order ambiguity | Brandt Ch. 12; Randall Ch. 3 | --- |
| 14.1--14.2 Template, cepstrum | Harmonic matching, cepstrum | **Randall Ch. 3 (cepstrum)** | Stoica & Moses Ch. 4 |
| 14.3 Kalman tracking | State-space smoothing | **Kay Ch. 13** | --- |
| 14.4 Tacholess tracking | STFT ridge following | Brandt Ch. 12; Müller Ch. 2 | Vaidyanathan |
| 14.6 Sensor fusion | Accelerometers, cross-correlation | Brandt Ch. 2--3 | Randall Ch. 2 |
| 14.7 Efficiency | rfft, multistage, interpolation | Sundararajan (Fast DFT); Lyons Ch. 10, 13 | harris (2004) |

## Cited paper to background reading

Every paper cited in the technical document, with the book chapter that prepares
for it. Read the book chapter first in each case.

| Cited paper (technical doc) | Read this first |
|---|---|
| Nyquist (1928); Shannon (1949) | Oppenheim/Willsky Ch. 7; Bracewell (sampling) |
| Oppenheim & Schafer (2010) | (is itself the book) |
| Cooley & Tukey (1965) | Oppenheim & Schafer Ch. 9; Sundararajan (Fast DFT) |
| Harris (1978) | Porat (Practical Spectral Analysis); Oppenheim & Schafer Ch. 10 |
| Butterworth (1930) | Oppenheim & Schafer Ch. 7 |
| Crochiere & Rabiner (1981) | Proakis Ch. 11; Lyons Ch. 10 |
| Crochiere & Rabiner (1983) | (is itself the book) |
| Vold & Leuridan (1995) | Brandt Ch. 12; Kay Ch. 13 |
| Randall (2011) | (is itself the book) |
| Randall & Antoni (2011) | Randall Ch. 3 (envelope analysis) |
| Antoni (2009) | Randall (cyclostationarity ch.) |
| Lu et al. (2019); Peeters et al. (2019) | Brandt Ch. 12; Kay Ch. 13 |
| Shan et al. (2020) | Brandt Ch. 2--3; Randall Ch. 2 |
| Rife & Boorstyn (1974) | **Kay Ch. 3 and Ch. 7** |
| Peterson & Brown (1961); Koopman & Chakravarty (2004) | Andrews (background only); see note below |
| Dijkstra (1968) | **Andrews Ch. 4** |
| NumPy (2020); SciPy (2020) | --- (software papers; no background needed) |
| SciPy `decimate` documentation | Proakis Ch. 11; Lyons Ch. 10 |
| Rotax 912 ULS technical data; Tecnam P92 specification | --- (manufacturer datasheets) |

All 24 sources cited in the technical document appear above.

**Note on the CRC references.** No book on this list covers cyclic codes, and none
is included, because the technical document's Section 9.6 already derives CRC-8
over GF(2) completely --- the generator polynomial, the long-division loop, and
all four detection guarantees --- and the two cited papers are short and
self-contained. A reader wanting more should go to a coding-theory text; adding one
to this guide would fail the relevance test stated in Section 1.

\newpage

# Coverage Check

This guide is complete with respect to the technical document and the code if
every DSP or systems construct in `rpi_test_final.py` and `audio_src/` has a
teaching source. The check:

| Code construct | File / class | Covered by |
|---|---|---|
| `sd.rec` at 44,100 Hz | `recorder` | Oppenheim/Willsky Ch. 7; Lyons Ch. 2 |
| `scipy.signal.decimate(q=88)` | `Decimation` | Proakis Ch. 11; Lyons Ch. 10; Crochiere & Rabiner |
| Chebyshev-I anti-alias filter | (inside SciPy) | Oppenheim & Schafer Ch. 7; Proakis Ch. 10 |
| `np.hanning` | `FrequencyPeakFinder` | Oppenheim & Schafer Ch. 10; Porat; Brandt Ch. 9 |
| `fftpack.fft` / `fftfreq` | `FrequencyPeakFinder` | Oppenheim & Schafer Ch. 8--9; Sundararajan |
| `np.argmax` over 10--200 Hz | `FrequencyPeakFinder` | Stoica & Moses Ch. 2; Kay Ch. 3; Lyons Ch. 13 |
| `EVENTS_PER_CYCLE = 4` | `RPM` | **Brandt Ch. 12; Randall (reciprocating ch.)** |
| `engine_status = rpm > 100` | `RPM` | Kay Ch. 1--2 (detection framing) |
| `queue.Queue(maxsize=10)` | module level | **Andrews Ch. 4--5** |
| `threading.Lock` (x2) | `uart_lock`, `init_lock` | **Andrews Ch. 4** |
| Six daemon threads | `main` | Andrews Ch. 1--2 |
| Framed packet + ACK + timestamps | `send_uart_packet` | Kopetz (global time, RT communication) |
| CRC-8, polynomial `0x07` | `calculate_crc8` | (derived in technical doc Section 9.7) |
| `scipy.signal.resample` | `Decimate.Resample` | Proakis Ch. 11; Vaidyanathan |
| `BandPassFilter` (Butterworth) | `Butterworth.py` | Oppenheim & Schafer Ch. 7 |
| `filtfilt` (zero-phase) | `Butterworth.py` | Oppenheim & Schafer Ch. 7; Lyons Ch. 5--6 |
| `TunableFilter` (adaptive BPF) | `Other.py` | Brandt Ch. 12 (tracking filters); Kay Ch. 13 |
| `STFT` wrapper | `Fourier.py` | Oppenheim & Schafer Ch. 10.3; **Müller Ch. 2** |
| `RFFT` (real-input, normalised) | `Fourier.py` | Sundararajan (Fast DFT, real data) |
| TFLite classifier + 70-window vote | `EngineAnalysis.EngineState` | Kay Ch. 1--2 (estimator/detector framing) |
| ADXL345 vibration interface | `Sensors/` | **Brandt Ch. 2--3**; Randall Ch. 2 |
| Three-level test ladder | `test_*.py` | Brandt Ch. 11 (test-system specification) |

Every construct has a source. The three entries in bold type are the ones where
the guide's reading would have *prevented* a defect documented in Section 13 of
the technical document: Brandt Ch. 12 and Randall's reciprocating-machines chapter
would have flagged the fixed-order assumption, and Andrews Ch. 4--5 would have
supplied the monitor-based drop-oldest queue that the current non-atomic
`get_nowait()`-then-`put()` sequence approximates.

## Summary of the list

| # | Book | Tier | Role | Priority |
|---:|---|---|---|---|
| 1 | Oppenheim, Willsky & Nawab --- *Signals and Systems* | 0 | Foundational | Core |
| 2 | Lyons --- *Understanding Digital Signal Processing* | 0 | Foundational | Core |
| 3 | Oppenheim & Schafer --- *Discrete-Time Signal Processing* | 1 | Foundational | Core |
| 4 | Proakis & Manolakis --- *Digital Signal Processing* | 1 | Foundational | Reference |
| 5 | Sundararajan --- *Digital Signal Processing: An Introduction* | 1 | Foundational | Core |
| 6 | Porat --- *A Course in Digital Signal Processing* | 1 | Foundational | Reference |
| 7 | Crochiere & Rabiner --- *Multirate Digital Signal Processing* | 2 | Foundational | Reference |
| 8 | Vaidyanathan --- *Multirate Systems and Filter Banks* | 2 | Advanced | Optional |
| 9 | harris --- *Multirate Signal Processing for Communication Systems* | 2 | Supplementary | Optional |
| 10 | Stoica & Moses --- *Spectral Analysis of Signals* | 3 | Foundational | Core |
| 11 | Kay --- *Statistical Signal Processing I: Estimation Theory* | 3 | Foundational | Core |
| 12 | Brandt --- *Noise and Vibration Analysis* | 4 | Foundational | **Core** |
| 13 | Randall --- *Vibration-based Condition Monitoring* | 4 | Foundational | **Core** |
| 14 | Andrews --- *Foundations of Multithreaded, Parallel, Distributed Programming* | 5 | Foundational | Core |
| 15 | Kopetz --- *Real-Time Systems* | 5 | Supplementary | Optional |
| 16 | Bracewell --- *The Fourier Transform and Its Applications* | 6 | Supplementary | Optional |
| 17 | Müller --- *Fundamentals of Music Processing* | 6 | Supplementary | Optional |

**Seventeen books.** Two of them --- Brandt and Randall --- address the problem
that the technical document identifies as the system's actual defect. If the
reading budget is one book, it is Brandt; if two, Brandt and Randall.
