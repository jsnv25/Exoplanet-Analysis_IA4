# Statistical Analysis of NASA Confirmed Exoplanets

This project analyzes a supplied catalog of 6,150 confirmed exoplanets across 31 fields. It combines data-quality checks, exploratory analysis, statistical hypothesis tests, regression, KMeans clustering, PCA, and a physics-based estimate of the habitable zone around each host star.

The habitable-zone calculation identifies **294 planets in the optimistic zone and 172 in the conservative zone**, among 5,788 planets with the stellar and orbital information required for that calculation. The conservative zone is contained within the optimistic zone, so the two counts overlap; they should not be added together.

## Contents

- `NASA_Exoplanets_IA4.ipynb` — analysis notebook.
- `nasa_exoplanet_intelligence.csv` — input data file expected by the notebook.
- `README.md` — project overview and instructions.
- `exoplanets_cleaned.csv` — generated cleaned dataset.
- `figures/` — generated analysis charts.

The last two items are created when the notebook runs. Keep the input CSV in the notebook's working directory, or update `CSV_PATH` in the setup cell.

## Habitable-zone results

The notebook compares the dataset's `habitable_zone_flag` with a separate, stellar-flux-based calculation using the habitable-zone limits from Kopparapu et al. (2013). These are two different screening rules:

| Measure | Count | Share of 5,788 eligible planets |
| --- | ---: | ---: |
| Optimistic flux-based habitable zone | **294** | **5.08%** |
| Conservative flux-based habitable zone | **172** | **2.97%** |

The analysis considered a planet eligible for the flux calculation when its host-star temperature was between 2,500 K and 7,500 K and the required luminosity and semi-major-axis values were available. The stellar luminosity is estimated from stellar radius and temperature. If the catalog has no semi-major axis, the notebook estimates it with Kepler's third law when the required stellar mass and orbital period are available.

The optimistic limits use the recent-Venus inner boundary and early-Mars outer boundary. The conservative limits use the moist-greenhouse inner boundary and maximum-greenhouse outer boundary. A planet inside either calculated band is a **candidate for further study**, not evidence that it has liquid surface water or life; the calculation does not establish its atmosphere, climate, or surface conditions.

For comparison, the catalog's own temperature-based flag marks **403 planets** whose equilibrium temperatures are between 200 K and 400 K. Only **109** satisfy both that temperature rule and the optimistic flux-based rule. The report also finds that 286 temperature-flagged planets are inside the optimistic inner boundary, and therefore too hot under the flux-based calculation. The two rules should not be treated as interchangeable.

An additional screening step selects non-controversial planets between 0.5 and 2 Earth radii that fall in the optimistic flux-based zone. It shortlists **43 candidates**. This is a screening result, not a claim of confirmed habitability.

## What the notebook does

1. **Checks the data.** Reports the table shape, duplicate rows and names, missing values, skew, IQR outliers, unknown categories, controversial records, and very long orbital periods. In the supplied snapshot, the table has 6,150 rows, 31 columns, no duplicate rows, and 6,346 missing cells.
2. **Prepares analysis fields.** Converts `Unknown` category labels to missing values, derives discovery era and star/planet groups, and creates log10 versions of positive measurements. It also estimates stellar luminosity, orbital distance where possible, insolation, habitable-zone boundaries, density, escape velocity, and an Earth Similarity Index.
3. **Keeps temperature predictions separate.** A linear regression using host-star temperature, semi-major axis, and stellar radius predicts equilibrium temperatures for eligible rows where the catalog value is missing. Predictions are stored in `predicted_equilibrium_temp_k`; the original `equilibrium_temp_k` values are preserved.
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
