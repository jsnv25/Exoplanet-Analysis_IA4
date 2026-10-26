<div align="center">

# Statistical Analysis of NASA Confirmed Exoplanets

**Which stars host which planets, where the habitable zone really is, and what the data can and cannot tell us.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regression%20%26%20clustering-F7931E?logo=scikitlearn&logoColor=white)
![Project](https://img.shields.io/badge/Data%20Analytics-Mini%20Project%20(IA4)-2b5c8f)

[**Project page**](https://jsnv25.github.io/Exoplanet-Analysis_IA4/) · [**Report (PDF)**](report.pdf) · [**Notebook**](Exoplanet_Analysis_Final.ipynb) · [**Script**](Exoplanet_Analysis_Final.py)

</div>

---

## Overview

This project analyses a NASA exoplanet catalogue of **6,150 confirmed planets** orbiting **4,582 host stars** (31 columns, discoveries from 1992 to 2026). It follows a complete data analytics workflow and adds an astronomy-specific question: *do the planets the dataset calls "habitable" really sit in a habitable zone?*

| Stage | What was done |
|---|---|
| Data quality | Duplicates, missing values, skewness and outliers checked; "Unknown" labels converted to missing values |
| Exploratory analysis | Distributions, discovery trends, mass–radius relation, correlation matrix, the radius valley |
| Star-type grouping | Planets grouped by host-star class (M, K, G, F, A/B/O) and compared on size, orbit, temperature and distance |
| Habitable zones | Dataset flag compared with a zone computed from each star's luminosity; Earth-like candidates ranked |
| Hypothesis testing | Six tests at α = 0.05: chi-square, Welch t-test, one-way and two-way ANOVA |
| Regression | Multiple linear regression predicting planet radius, with and without planet mass |
| Clustering | KMeans (elbow and silhouette) with PCA visualisation |

## Key findings

- **Red dwarfs host smaller planets.** Median planet radius is 1.9 Earth radii for M-type hosts, 2.6 for K, 2.8 for G and 3.4 for F. 52% of M-dwarf planets are under 2 Earth radii, against 27% for Sun-like G stars.
- **The dataset's habitable-zone flag is generous.** It marks 403 planets, and the flag is exactly "equilibrium temperature between 200 and 400 K". A zone computed from stellar luminosity gives **294 planets** (optimistic) or **172** (conservative). Only 109 planets are in both, and 286 flagged planets receive too much energy to qualify.
- **Small planets in the habitable zone favour red dwarfs.** 14.2% of small (0.5–2 Earth radii) M-dwarf planets sit in the zone, against 2.2% for K and 0.4% for G.
- **Newer discoveries are smaller.** Planets found from 2015 onwards have a geometric mean radius of 3.53 Earth radii, against 4.56 before, a survey effect from Kepler and TESS rather than a change in nature.
- **Giant planets prefer metal-rich stars.** Mean host [Fe/H] is +0.071 for giants and −0.028 for rocky planets, in line with the planet–metallicity trend reported in [1] and [2].
- **43 Earth-like candidates** (0.5–2 Earth radii, inside the optimistic zone, not flagged controversial). The top three by Earth Similarity Index are **Wolf 1069 b (0.97), TRAPPIST-1 e (0.95) and TOI-700 d (0.95)**. The closest is Proxima Cen b, 1.30 parsecs away.

<p align="center">
  <img src="figures/fig5_habitable_zones.png" alt="Planets against the habitable zone" width="95%">
</p>

## Hypothesis tests (α = 0.05)

| # | Test | Question | Result | Decision |
|---|---|---|---|---|
| H1 | Chi-square | Is planet type independent of host-star type? | χ²(15) = 482.9, p = 2.4e-93 | Reject H0 |
| H2 | Welch t-test | Is host metallicity equal for giant and rocky planets? | t = 14.37, p = 2.2e-45 | Reject H0 |
| H3 | One-way ANOVA | Is mean log radius equal across F, G, K, M hosts? | F(3, 5761) = 58.4, p = 3.3e-37 | Reject H0 |
| H4 | Welch t-test | Is mean log radius equal before and after 2015? | t = 9.44, p = 7.0e-21 | Reject H0 |
| H5 | Chi-square | Is habitable-zone membership independent of star type? | χ²(3) = 75.4, p = 2.9e-16 | Reject H0 |
| H6 | Two-way ANOVA | Star type × discovery era on log radius | Interaction F(3, 5756) = 4.46, p = 0.004 | Reject H0 |

Radius is log-transformed before testing because it is strongly right-skewed. H1–H4 were repeated without the 49 planets flagged controversial, and every decision stayed the same.

## Regression and clustering

**Multiple linear regression** (target: log10 planet radius, 80/20 split, 5,494 planets, seed 42):

| Model | R² | RMSE | MAE |
|---|---|---|---|
| A: with planet mass | 0.844 | 0.155 | 0.106 |
| B: without planet mass | 0.396 | 0.306 | 0.233 |

Most of Model A's fit is the mass–radius relation itself (see [3] and [4] on that relationship). Mass is present for 99.5% of planets, which suggests many values are estimates derived from radius, so Model B is the more honest view of how much the orbit and host star explain.

**KMeans clustering** (4,500 planets with complete data, six standardised features): k = 3 has the highest silhouette score (0.373), and PCA keeps 73.8% of the variance in two components. The groups are small planets around cool dwarfs (728), hot giant planets (907) and typical close-in small planets (2,865).

<p align="center">
  <img src="figures/fig9_clustering.png" alt="Elbow, silhouette and KMeans clusters" width="95%">
</p>

## Repository structure

```
Exoplanet-Analysis_IA4/
├── index.html                      # project page (GitHub Pages)
├── README.md
├── report.pdf                      # the project report
├── Exoplanet_Analysis_Final.py     # full analysis as a script
├── Exoplanet_Analysis_Final.ipynb  # the same analysis as a notebook
├── requirements.txt
└── figures/                        # 9 charts written by the script
```

## Quick start

```bash
git clone https://github.com/jsnv25/Exoplanet-Analysis_IA4.git
cd Exoplanet-Analysis_IA4
pip install -r requirements.txt
```

1. Put `nasa_exoplanet_intelligence.csv` in the same folder. **The dataset is not included in this repository.**
2. Run `python Exoplanet_Analysis_Final.py`, or open the notebook in Jupyter or Google Colab and run all cells.

It takes about 15 seconds and writes the figures to `figures/` and a cleaned dataset to `exoplanets_cleaned.csv` (all original columns plus 23 derived ones; nothing is dropped or imputed). Results are reproducible because every random step uses seed 42.

## Data

The input comes from NASA's exoplanet catalogue [6] and has 6,150 rows and 31 columns: planet properties (radius, mass, orbital period, equilibrium temperature, eccentricity), host-star properties (temperature, radius, mass, age, metallicity), discovery details (method, facility, year), distance and sky position, plus pre-computed labels such as `planet_type`, `star_type` and `habitable_zone_flag`.

Columns added by the script include `Star_Class`, `Era`, log-transformed variables, stellar luminosity, the energy a planet receives (`insolation`), the habitable-zone limits and flags, and the Earth Similarity Index (`ESI`).

## Methods in brief

- **Habitable zone.** The inner and outer limits come from Kopparapu et al. [5], computed per star from its temperature and luminosity (luminosity = radius² × (temperature / 5772 K)⁴). Where the orbit size was missing it was derived from Kepler's third law. The "conservative" zone runs from the moist-greenhouse to the maximum-greenhouse limit; the "optimistic" zone from recent Venus to early Mars.
- **Earth Similarity Index.** Combines radius, density, escape velocity and equilibrium temperature into one score where 1.0 means Earth. Planets with physically impossible densities (above 25 or below 0.05 g/cm³) get no score.
- **Missing values.** Not imputed. Each test and model uses the rows where its variables exist, which avoids leaking information from test data into training.

## Limitations

- The catalogue shows what surveys can detect, not what exists. Counts per star type are not occurrence rates, and red dwarfs are over-represented among nearby discoveries.
- Many planets lack a measured mass, and Teq assumes a fixed reflectivity, so ESI is a rough ranking and not a measurement.
- Being inside the habitable zone is not the same as being habitable. It says nothing about atmosphere, magnetic field or stellar flares, which matter a lot for red dwarfs.
- The formulas in [5] are calibrated for stars between 2,600 and 7,200 K, and the script extends them slightly to 2,500–7,500 K.
- 2026 is a partial year in the data.

## Future work

Correct for detection bias using survey completeness, add stellar flare and tidal-locking information for M dwarfs, try tree-based or gradient-boosting regressors with cross-validation, and build an interactive dashboard.

## References

[1] L. A. Buchhave et al., “An abundance of small exoplanets around stars with a wide range of metallicities,” *Nature*, vol. 486, pp. 375–377, 2012. doi: [10.1038/nature11121](https://doi.org/10.1038/nature11121).

[2] A. Osborn and D. Bayliss, “Investigating the planet–metallicity correlation for hot Jupiters,” *MNRAS*, vol. 491, no. 3, pp. 4481–4487, 2020. doi: [10.1093/mnras/stz3207](https://doi.org/10.1093/mnras/stz3207).

[3] C. Mordasini et al., “Characterization of exoplanets from their formation – II. The planetary mass-radius relationship,” *A&A*, vol. 547, A112, 2012. doi: [10.1051/0004-6361/201118464](https://doi.org/10.1051/0004-6361/201118464).

[4] S. Müller, J. Baron, R. Helled, F. Bouchy, and L. Parc, “The mass–radius relation of exoplanets revisited,” *A&A*, vol. 686, A296, 2024. doi: [10.1051/0004-6361/202348690](https://doi.org/10.1051/0004-6361/202348690).

[5] R. K. Kopparapu et al., “Habitable zones around main-sequence stars: New estimates,” *ApJ*, vol. 765, no. 2, article 131, 2013. doi: [10.1088/0004-637X/765/2/131](https://doi.org/10.1088/0004-637X/765/2/131).

[6] NASA Exoplanet Archive, “Acknowledging the NASA Exoplanet Archive in Publications,” https://exoplanet.ipac.caltech.edu/docs/acknowledge.html (accessed 25 Oct. 2026).

## Author

**JSNV** ([@jsnv25](https://github.com/jsnv25)), Data Analytics mini project (IA4).

Code and report are shared for learning and reference. The exoplanet data comes from NASA's public exoplanet catalogue [6]; check the original source for its terms before redistributing it.
