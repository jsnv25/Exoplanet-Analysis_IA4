# Statistical Analysis of NASA Confirmed Exoplanets

This project analyzes a supplied catalog of 6,150 confirmed exoplanets across 31 fields. It combines data-quality checks, exploratory analysis, statistical hypothesis tests, regression, KMeans clustering, PCA, and a physics-based estimate of the habitable zone around each host star.

The habitable-zone calculation identifies **294 planets in the optimistic zone and 172 in the conservative zone**, among 5,788 planets with the stellar and orbital information required for that calculation. The conservative zone is contained within the optimistic zone, so the two counts overlap; they should not be added together.

## Contents

- `NASA_Exoplanets_IA4.ipynb` — analysis notebook.
- `nasa_exoplanet_intelligence.csv` — input data file expected by the notebook.
- `README.md` — project overview and instructions.
- `exoplanets_cleaned.csv` — generated cleaned dataset.
- `Report/Team_8.pdf` — Report on NASA Exoplanets Archive.
- `figures/` — generated analysis charts.

The last two items are created when the notebook runs. Keep the input CSV in the notebook's working directory, or update `CSV_PATH` in the setup cell.

## Habitable-zone results

The notebook compares the dataset's `habitable_zone_flag` with a separate, stellar-flux-based calculation using the habitable-zone limits from Kopparapu et al. (2013). These are two different screening rules:

| Measure | Count | Share of 5,788 eligible planets |
| --- | ---: | ---: |
| Optimistic flux-based habitable zone | **294** | **5.08%** |
| Conservative flux-based habitable zone | **172** | **2.97%** |

The analysis considered a planet eligible for the flux calculation when its host-star temperature was between 2,500 K and 7,500 K and the required luminosity and semi-major-axis values were available. The stellar luminosity is estimated from stellar radius and temperature. If the catalog has no semi-major axis, the notebook estimates it with Kepler's third law when the required stellar mass and orbital period are available. This yielded 5,788 eligible planets out of 6,150.

The optimistic limits use the recent-Venus inner boundary and early-Mars outer boundary. The conservative limits use the moist-greenhouse inner boundary and maximum-greenhouse outer boundary. A planet inside either calculated band is a **candidate for further study**, not evidence that it has liquid surface water or life; the calculation does not establish its atmosphere, climate, or surface conditions.

For comparison, the catalog's own temperature-based flag marks **403 planets** whose equilibrium temperatures are between 200 K and 400 K. Of the 5,788 flux-eligible planets, 400 have that temperature flag: 109 are also in the optimistic flux-based zone and 291 are outside it. Three of the 403 temperature-flagged planets lack the inputs for the flux comparison. Among the eligible temperature-flagged planets outside the optimistic zone, 286 are inside the optimistic inner boundary and therefore too hot under the flux-based calculation; the remaining five are outside the optimistic band for another reason. The two rules should not be treated as interchangeable.

An additional screening step selects non-controversial planets between 0.5 and 2 Earth radii that fall in the optimistic flux-based zone. It shortlists **43 candidates**. This is a screening result, not a claim of confirmed habitability.

## What the notebook does

1. **Checks the data.** Reports the table shape, duplicate rows and names, missing values, skew, IQR outliers, unknown categories, controversial records, and very long orbital periods. In the supplied snapshot, the table has 6,150 rows, 31 columns, no duplicate rows or planet names, and 6,346 pre-existing missing cells (3.33% of all input cells).
2. **Prepares analysis fields.** Converts `Unknown` category labels to missing values, derives discovery era and star/planet groups, and creates log10 versions of positive measurements. It also estimates stellar luminosity, orbital distance where possible, insolation, habitable-zone boundaries, density, escape velocity, and an Earth Similarity Index. These steps preserve all 6,150 rows; they do not drop the source records.
3. **Keeps temperature predictions separate.** A linear regression using host-star temperature, semi-major axis, and stellar radius predicts equilibrium temperatures for eligible rows where the catalog value is missing. In this data snapshot, 1,262 of the 1,563 missing temperatures have all three predictors available; the other 301 cannot be predicted by this model. Predictions are stored in `predicted_equilibrium_temp_k`; the original `equilibrium_temp_k` values are preserved.
4. **Explores distributions and relationships.** Calculates summary statistics, discovery-method and planet-type counts, mass-radius correlation, and comparisons by host-star type and discovery era. The notebook saves nine figures in `figures/`.
5. **Tests six hypotheses.** Uses chi-square tests, Welch's t-tests, one-way ANOVA, and two-way ANOVA at a significance level of 0.05. The report finds statistically significant differences for all six tests; because the data are observational, these results describe associations and do not establish causation.
6. **Models planet radius.** Compares a multiple linear regression model with log mass against one without log mass, using a reproducible 80/20 train/test split.
7. **Groups planets exploratorily.** Standardizes six features, evaluates KMeans solutions for k = 2 through 8, chooses the highest silhouette score, and uses PCA to visualize the selected groups.

## Selected findings

- Log10 planet mass and radius are strongly correlated in the supplied data (Pearson *r* = 0.918, *n* = 6,076).
- Transit accounts for 73.4% of discoveries in the analyzed snapshot.
- In the radius-prediction comparison, Model A (with log mass) achieves test R² = 0.844, RMSE = 0.155, and MAE = 0.106. Model B (without log mass) achieves R² = 0.396, RMSE = 0.306, and MAE = 0.233. Much of Model A's performance reflects the mass-radius relationship; the report notes that some archive masses may themselves be estimates based on radius.
- The report retains k = 3 as an exploratory clustering solution. Its silhouette score is 0.373, close to 0.372 for k = 2, so the clusters should not be read as definitive planet classes.
- Repeating the first four hypothesis tests without the 49 controversial records does not change their reject/fail-to-reject decisions.

## Data quality and missing values

The table below reports **missing values in the original 6,150 × 31 input file, before replacing the literal string `Unknown` with missing values**. Percentages use 6,150 rows as the denominator and are rounded to one decimal place. There are 6,346 null cells overall (3.33% of all 190,650 input cells).

| Input field | Missing | Percent |
| --- | ---: | ---: |
| `disc_year` | 1 | 0.0% |
| `orbital_period_days` | 334 | 5.4% |
| `planet_radius_earth` | 50 | 0.8% |
| `planet_mass_earth` | 31 | 0.5% |
| `equilibrium_temp_k` | 1,563 | 25.4% |
| `orbital_eccentricity` | 938 | 15.3% |
| `semi_major_axis_au` | 316 | 5.1% |
| `star_temp_k` | 290 | 4.7% |
| `star_radius_sun` | 314 | 5.1% |
| `star_mass_sun` | 8 | 0.1% |
| `star_age_gyr` | 1,311 | 21.3% |
| `star_surface_gravity` | 318 | 5.2% |
| `star_metallicity` | 550 | 8.9% |
| `dist_from_earth_pc` | 27 | 0.4% |
| `star_vmag` | 295 | 4.8% |

The remaining 16 input fields have no nulls: `planet_name`, `host_star`, `n_stars`, `n_planets`, `discovery_method`, `disc_facility`, `ra`, `dec`, `controversial_flag`, `planet_type`, `habitable_zone_flag`, `multi_planet_system`, `is_recent_discovery`, `dist_category`, `star_type`, and `orbital_period_cat`.

The cleaning step also converts literal `Unknown` entries to nulls in `planet_type` (50), `star_type` (290), `dist_category` (27), and `orbital_period_cat` (334). These are not included in the 6,346 pre-existing null-cell total above. The analysis does not fill those categories or generally impute missing input values. The equilibrium-temperature regression is the sole exception: its estimates go into a separate derived column, leaving the original measurements unchanged.

### Cleaning and derived values

| Operation | Result |
| --- | --- |
| Duplicate rows / duplicate planet names | 0 / 0 |
| `Unknown` values recoded to missing | 50 planet types; 290 star types; 27 distance categories; 334 orbital-period categories |
| Controversial records | 49 flagged; retained in the main analysis, then excluded in the H1–H4 sensitivity comparison |
| Very long orbits | 5 planets with periods over 1,000,000 days; excluded from clustering only |
| Temperature log transform | `planet_radius_earth` skew 1.74 → 0.26; `planet_mass_earth` 4.39 → 0.66; `orbital_period_days` 76.15 → 1.36 |
| IQR upper-fence observations | Radius: 6 (0.1%); mass: 1,009 (16.5%); orbital period: 1,014 (17.4%). These are flagged for inspection, not deleted. |
| Temperature regression coverage | 1,262 rows predicted; 301 rows remain unpredicted because one or more predictors are missing |
| Output size from the updated notebook | 6,150 rows and 55 columns: the 31 input fields plus 24 derived fields |

Derived columns include discovery era and decade; star and planet groupings; six log10 measures; stellar luminosity, semi-major axis, and insolation; four habitable-zone limits and two zone flags; density, escape velocity, ESI; and `predicted_equilibrium_temp_k`.

### Descriptive statistics

Statistics below are computed on the available, non-missing values for each field. Standard deviation uses the sample standard deviation. Large means and standard deviations for orbital period reflect the long-period tail.

| Variable | n | Mean | Median | Std. dev. | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Planet radius (Earth radii) | 6,100 | 5.84 | 2.84 | 5.44 | 0.31 | 87.21 |
| Planet mass (Earth masses) | 6,119 | 406.79 | 9.27 | 1,142.87 | 0.02 | 9,534.85 |
| Orbital period (days) | 5,816 | 73,594.21 | 11.13 | 5,273,718.15 | 0.09 | 402,000,000 |
| Equilibrium temperature (K) | 4,587 | 912.90 | 818.00 | 464.27 | 34 | 4,050 |
| Host-star temperature (K) | 5,860 | 5,398.16 | 5,546.00 | 1,747.46 | 415 | 57,000 |
| Host-star mass (Solar masses) | 6,142 | 0.94 | 0.94 | 0.41 | 0.01 | 10.94 |
| Host-star metallicity ([Fe/H]) | 5,600 | 0.02 | 0.02 | 0.19 | -1.00 | 0.60 |
| Distance (parsecs) | 6,123 | 711.67 | 376.66 | 1,212.14 | 1.30 | 8,500 |

Other composition summaries: discovery method is Transit 73.4%, Radial Velocity 19.2%, Microlensing 4.5%, and Imaging 1.5%. Planet-type counts are Mini-Neptune 2,148; Gas Giant 1,734; Super-Earth 1,185; Neptune-like 479; Super-Jupiter 324; and Sub-Earth 230. The archive records 4,582 host stars, and 42.6% of planets are in multi-planet systems.

## Hypothesis tests

All tests use α = 0.05. Reported p-values are unadjusted; the notebook does not apply a multiple-testing correction. “Reject H₀” indicates evidence against the stated null under the test assumptions, not a causal conclusion.

| Test | Null hypothesis and sample | Test statistic | Result |
| --- | --- | --- | --- |
| H1: Chi-square | Planet type is independent of F/G/K/M host-star class; n = 5,765 | χ²(15) = 482.9; p = 2.37 × 10⁻⁹³; minimum expected cell = 16.6 | Reject H₀. Planet-type mix differs by host-star class. |
| H2: Welch t-test | Mean host-star [Fe/H] is equal for giant and rocky planets; n = 1,733 giant and 1,356 rocky | t(3,074) = 14.37; p = 2.24 × 10⁻⁴⁵ | Reject H₀. Mean [Fe/H] is 0.071 for giant hosts and -0.028 for rocky-planet hosts. |
| H3: One-way ANOVA | Mean log10(radius) is equal across F/G/K/M hosts; n = 1,073 / 2,653 / 1,617 / 422 | F(3, 5,761) = 58.44; p = 3.30 × 10⁻³⁷ | Reject H₀. Median radii are 3.40 / 2.80 / 2.58 / 1.90 Earth radii for F/G/K/M. Group SDs of log10(radius) are 0.425 / 0.378 / 0.405 / 0.377. |
| H4: Welch t-test | Mean log10(radius) is equal before 2015 and from 2015 onward; n = 1,752 / 4,347 | t(3,030) = 9.44; p = 7.01 × 10⁻²¹ | Reject H₀. Geometric mean radius is 4.56 Earth radii before 2015 and 3.53 from 2015 onward. |
| H5: Chi-square | Optimistic flux-based HZ membership is independent of F/G/K/M host class | χ²(3) = 75.4; p = 2.93 × 10⁻¹⁶; minimum expected cell = 20.7 | Reject H₀. Optimistic-HZ rates are F 2.6%, G 4.6%, K 5.4%, and M 13.5%. |
| H6: Two-way ANOVA | Log10(radius) has no star-class effect, era effect, or interaction; n = 5,764 | Star class: F(3, 5,756) = 51.41, p = 8.82 × 10⁻³³; era: F(1, 5,756) = 111.01, p = 1.01 × 10⁻²⁵; interaction: F(3, 5,756) = 4.46, p = 3.93 × 10⁻³ | Reject each null component. Radius patterns differ by star class and era, with a statistically significant interaction. |

For H1 and H5, the smallest expected chi-square counts are both above 5. The H3 group standard deviations are similar, supporting the notebook's stated equal-variance rationale. The analysis reruns H1–H4 without the 49 controversial planets; the decisions remain unchanged:

| Test | All records p-value | Excluding controversial records p-value | Decision changes? |
| --- | ---: | ---: | --- |
| H1 | 2.37 × 10⁻⁹³ | 1.12 × 10⁻⁹² | No |
| H2 | 2.24 × 10⁻⁴⁵ | 4.00 × 10⁻⁴⁴ | No |
| H3 | 3.30 × 10⁻³⁷ | 7.10 × 10⁻³⁷ | No |
| H4 | 7.01 × 10⁻²¹ | 2.65 × 10⁻²¹ | No |

## Regression and clustering results

### Radius regression

The target is log10(planet radius). The notebook keeps 5,494 complete records from 6,100 planets with a radius, then uses a fixed-seed 80/20 split (4,395 training; 1,099 test records). Predictors are standardized using training data.

| Model | Predictors | Test R² | RMSE (log10 radius) | MAE (log10 radius) | Approx. multiplicative error from RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Model B predictors plus log10(mass) | 0.844 | 0.155 | 0.106 | ×1.43 |
| B | Orbit, host-star, metallicity, distance, star-group, and discovery-method fields; no mass | 0.396 | 0.306 | 0.233 | ×2.02 |

Mass raises test R² substantially, but the report notes that many catalog masses may be empirical estimates based on radius. Model A therefore partly restates the mass-radius relationship and should not be read as an independent physical explanation.

### KMeans and PCA

Clustering uses 4,500 complete cases after excluding orbital periods over 1,000,000 days. Six standardized inputs are used: log10(period), log10(radius), log10(mass), log10(equilibrium temperature), host-star temperature, and log10(stellar mass). The notebook evaluates k = 2 through 8 by silhouette score and chooses k = 3 (0.373), narrowly above k = 2 (0.372). The first two PCA axes explain 46.0% and 27.9% of variance (73.8% combined, as reported by the notebook).

Cluster labels are algorithmic identifiers; they do not imply official planet classes. Medians and the planet-type counts below describe the fitted clusters:

| Cluster | Planets | Median radius (R⊕) | Median mass (M⊕) | Median period (days) | Median Teq (K) | Median host Teff (K) | Median host mass (M☉) | Optimistic-HZ planets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 728 | 1.92 | 4.66 | 7.55 | 496.9 | 3,810 | 0.55 | 64 |
| 2 | 907 | 12.78 | 293.96 | 4.15 | 1,345 | 5,808 | 1.10 | 10 |
| 3 | 2,865 | 2.24 | 5.84 | 11.13 | 811 | 5,670 | 0.96 | 39 |

Planet-type counts within each cluster:

| Cluster | Sub-Earth | Super-Earth | Mini-Neptune | Neptune-like | Gas Giant | Super-Jupiter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 61 | 250 | 347 | 32 | 38 | 0 |
| 2 | 0 | 5 | 17 | 31 | 579 | 275 |
| 3 | 151 | 809 | 1,560 | 285 | 60 | 0 |

Silhouette scores for all tested k values: k=2, 0.372; k=3, 0.373; k=4, 0.335; k=5, 0.295; k=6, 0.299; k=7, 0.280; k=8, 0.268. The near-tie between k=2 and k=3 means the three-cluster interpretation is exploratory rather than a uniquely established grouping.

## Running the notebook

### Requirements

Python 3 and the following packages are used:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
jupyter
```

Install the packages with:

```bash
python -m pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### Steps

1. Place `NASA_Exoplanets_IA4.ipynb` and `nasa_exoplanet_intelligence.csv` in the same working directory.
2. Start Jupyter Notebook or JupyterLab in that directory.
3. Open the notebook and run the cells from top to bottom.
4. Find the cleaned data in `exoplanets_cleaned.csv` and the charts in `figures/` in the notebook's working directory.

The analysis uses a fixed random seed (`42`) for the train/test split and KMeans. Results can differ if the input data or analysis code changes.

## Important interpretation notes

- The habitable-zone counts depend on the supplied catalog and the stated temperature, luminosity, and orbit-availability filters. They are not counts of confirmed habitable worlds.
- Missing category values are not generally filled in. The separate equilibrium-temperature prediction column is provided for reference and does not overwrite observed values.
- A very small p-value indicates evidence against a test's null hypothesis under its assumptions; it does not measure practical importance. The large sample also makes small differences easier to detect.
- Planet mass is present for most records, but some values may be empirical estimates. This limits how independently the mass-inclusive radius model can be interpreted.
- Discovery methods have different selection effects. The observed sample is not a complete census of all planets in the Galaxy.

## References

1. R. K. Kopparapu et al., “Habitable Zones around Main-sequence Stars: New Estimates,” *The Astrophysical Journal*, 765(2), 131 (2013). [https://doi.org/10.1088/0004-637X/765/2/131](https://doi.org/10.1088/0004-637X/765/2/131)
2. NASA Exoplanet Archive, [Acknowledging the NASA Exoplanet Archive in Publications](https://exoplanet.ipac.caltech.edu/docs/acknowledge.html).

For the broader literature review and the full set of reported hypothesis-test results, see the accompanying project report.
