# `worms-doubleml`

**Who benefits from childhood deworming? Heterogeneous treatment effects on long-run human capital and labor market outcomes.**

---

## Abstract

School-based deworming programs have reached hundreds of millions of children across sub-Saharan Africa and South Asia, scaling up from a landmark randomized trial in Kenya. Whether benefits from these programs are concentrated among specific groups (younger children who were treated during more critical developmental windows, or girls who face steeper labor market barriers) has direct implications for program design. We apply double/debiased machine learning (DoubleML) and causal forest estimation to the Kenya Life Panel Survey follow-up of the Miguel and Kremer (2004) deworming trial, estimating long-run effects of childhood deworming on adult BMI, underweight prevalence, educational attainment, and employment, and characterizing heterogeneity by age at treatment and gender. The developmental-window hypothesis is not confirmed by formal inference: no estimated heterogeneity survives Romano-Wolf multiple testing correction (minimum adjusted p = 0.33). The null is not an attrition artifact (Lee bounds for BMI: [-0.10, +0.08]) and is not explained by methodological choices. The design was underpowered for plausible nutritional effect sizes at a 20-year horizon. Against this primary null, we document three directionally consistent but fragile signals that motivate further investigation at scale.

---

## Background and motivation

The Primary School Deworming Project (PSDP), conducted in western Kenya between 1998 and 2001, randomly assigned 75 primary schools to receive deworming at different times. Miguel and Kremer (2004) found substantial reductions in school absenteeism. Hamory et al. (2021) followed the same cohort two decades later and found significant gains in earnings and consumption. Evidence from this trial informed the Kenya National School-Based Deworming Programme, which scaled to millions of children from 2009, and programs like the Deworm the World Initiative (Evidence Action), which now operates across Kenya, India, Ethiopia, and Pakistan.

A natural next question for program designers at these scales is whether benefits are concentrated in ways that could guide targeting. Two hypotheses have theoretical grounding. First, the developmental-window hypothesis: children who were younger at the time of treatment were dewormed during more critical periods of physical and cognitive development, so effects on adult human capital should be larger for them. Second, a gender-differentiated labor market pathway: girls in rural Kenya face competing demands (household labor, early marriage pressure) that make the school attendance and eventually the employment margin more binding. If deworming eased the health constraint on attendance, the downstream effects on labor market participation may be larger for women.

These questions are exactly the kind that standard subgroup analysis handles poorly. Splitting a sample by age tercile or gender and comparing means pre-commits to a particular partition, is sensitive to the choice of cutpoint, and does not adjust for the correlation between moderators and other baseline covariates. Double/debiased machine learning with causal forest estimation offers a more principled approach. The causal forest estimates a conditional average treatment effect (CATE) for each individual as a flexible function of baseline covariates, without pre-committing to a partition. The best linear predictor (BLP) of the CATE then provides valid inference on pre-specified moderators, accounting for the uncertainty in the forest estimates themselves. Together, the two methods are strictly more informative than a table of interaction terms and, crucially, control the false discovery rate across the full set of heterogeneity tests via Romano-Wolf stepdown correction.

This repository applies that methodological apparatus to a question where the answer matters for operational program design. The honest answer, as it turns out, is a carefully characterized null.

---

## Data

The analysis combines two data sources.

**PSDP baseline (1998).** The original trial enrolled 75 schools and approximately 30,000 pupils. Schools were randomized into three groups: full treatment (deworming beginning in 1998), partial treatment (beginning in 1999), and comparison (no treatment during the study period). The baseline data provide school identifiers, treatment assignment, and limited pupil-level covariates.

**Kenya Life Panel Surveys (KLPS).** The KLPS tracked the same cohort into adulthood across two waves: KLPS-3 (2011-2014, approximately 10-15 years post-treatment) and KLPS-4 (2017-2022, approximately 17-22 years post-treatment). The analysis sample consists of the 7,527 PSDP-enrolled pupils with confirmed treatment assignment, excluding the Girl Sponsorship Programme subsample. Attrition to the tracked sample is roughly 33% for nutritional outcomes and employment and 44% for education.

**Treatment definition.** The primary analysis pools full-treatment and partial-treatment schools into a single treated group. At a 15-20 year horizon, the one-year difference in treatment timing has plausibly attenuated. The pooled estimand is a convex combination of what we call direct-early effects (full-treatment schools, dewormed from 1998), direct-late effects (partial-treatment schools, dewormed from 1999), and spillover-late effects (partial-treatment pupils who were untreated in 1998 but exposed to neighboring full-treatment schools). A formal test of equality between the full-treatment and partial-treatment ATEs fails to reject at the 5% level for all four outcomes (p = 0.20, 0.48, 0.055, 0.94 for BMI, underweight, education, and employment respectively). The restriction to full-treatment vs. comparison only is reported as a sensitivity check throughout.

**Outcomes.** We examine four outcomes organized by the causal chain we hypothesize: adult BMI and underweight prevalence (KLPS-3, nutritional status), years of schooling completed (KLPS-3, human capital), and employment status (KLPS-4, labor market). Earnings are excluded: the KLPS-4 E-Module has 75% attrition relative to the original PSDP cohort, producing a sample too selected for credible causal inference. Education has 44% missingness on the primary variable, non-differential by treatment (p = 0.85), and results are treated as suggestive rather than definitive.

**Moderators.** Two moderators are pre-specified: age at treatment in 1998 (primary, testing the developmental-window hypothesis) and gender (secondary, testing the labor market margin hypothesis). Age at treatment has 14-21% missingness in year of birth and is imputed via multiple imputation by chained equations (MICE) with 20 imputations, using baseline grade, school-level clustering, and parental education as predictors. A cross-wave consistency check revealed that parental education reports are unstable across KLPS-3 and KLPS-4 (Spearman rho: father 0.16, mother 0.29), so parental education is retained as a noisy control in outcome models but excluded from the imputation model. A complete-case sensitivity check restricting to respondents with non-missing year of birth is reported alongside the primary results.

---

## Methods

### Estimation framework

The analysis follows the double/debiased ML framework of Chernozhukov et al. (2018). For each outcome $Y$, treatment $D$, moderators $X$, and controls $W$, the partially linear regression model is:

$$Y_i = \theta D_i + g(W_i) + \varepsilon_i, \quad D_i = m(W_i) + v_i$$

Cross-fitting (splitting the sample into folds, estimating nuisance functions $g$ and $m$ on held-out folds, and forming orthogonal residuals) ensures that the treatment effect estimator $\hat{\theta}$ is root-n consistent and asymptotically normal even when the nuisance functions are estimated by high-dimensional machine learning methods. We use gradient-boosted trees (HistGradientBoosting) for both nuisance functions.

Cluster-aware cross-fitting is essential here. With 73 schools and unconditional intraclass correlation of 0.023, assigning pupils from the same school to both training and test folds leaks the school-level treatment signal into the nuisance model, biasing the orthogonal scores. We enforce school-level fold boundaries throughout via GroupKFold with five folds.

### Heterogeneity estimation

Treatment effect heterogeneity is estimated via a causal forest (CausalForestDML from EconML), which produces an individual-level CATE $\hat{\tau}(X_i)$ as a nonparametric function of the moderators. The forest uses 4,000 trees with minimum leaf size of 50 (approximately average school size), which prevents the forest from learning splits that effectively isolate individual pupils within schools after cluster-aware cross-fitting. Variance is estimated via bootstrap of little bags resampling at the school level.

The causal forest output is a high-dimensional object that is useful for visualization but difficult to act on directly. The best linear predictor (BLP) of the CATE projects the forest estimates onto the pre-specified moderators using doubly robust orthogonal scores (LinearDML):

$$E[\hat{\tau}(X_i) \mid \text{moderators}] = \alpha_0 + \alpha_1 \cdot \text{age}_{1998,i} + \alpha_2 \cdot \text{female}_i + \alpha_3 \cdot (\text{age}_{1998,i} \times \text{female}_i)$$

$\alpha_1$ tests the developmental-window hypothesis directly. $\alpha_2$ tests whether women and men differ in their long-run returns. $\alpha_3$ tests whether the age gradient itself differs by gender, motivated by the competing-demands argument.

This implementation approximates current best practice in Python, with three known limitations relative to the `grf` package in R. Trees are not fully honest: they do not use separate subsamples for determining splits and estimating leaf means (Wager and Athey, 2018; Athey et al., 2019). Variance estimates are approximate: the bootstrap-of-little-bags approximates the cluster-robust infinitesimal jackknife (Wager et al., 2014; Kleiner et al., 2014), but with 73 clusters, standard tests over-reject and confidence intervals for heterogeneity may undercover slightly (Cameron et al., 2008). BLP inference via LinearDML is asymptotically valid under Neyman orthogonality but finite-sample coverage is not guaranteed (Chernozhukov et al., 2018; 2017).

### Multiple testing correction

With two moderators tested across four outcomes, unadjusted p-values are not informative about the overall inference picture. We apply Romano-Wolf stepdown correction separately within each wave family: the KLPS-3 family covers BMI, underweight, and education (six tests); the KLPS-4 family covers employment (two tests). The bootstrap resamples at the school level. Because the education outcome has a smaller sample (n = 2,769) than BMI and underweight (n = 5,025), effective sample size for education varies across bootstrap replications, making the Romano-Wolf adjustment for education slightly conservative.

---

## Results

### Primary results: no evidence of heterogeneity

No estimated heterogeneity survives Romano-Wolf multiple testing correction. All BLP coefficients have Romano-Wolf adjusted p-values above 0.33. The minimum adjusted p-value across all moderator-outcome combinations is 0.33, for the gender interaction on employment. The developmental-window hypothesis (that younger children at the time of treatment benefited more) is not confirmed by formal inference for any outcome. This is the headline result and it holds across all specifications examined.

The average treatment effect estimates are themselves informative. BMI shows a pooled ATE of -0.24 points (p = 0.001), running counter to the expected direction of nutritional improvement. Underweight probability shows a pooled ATE of +1.5 percentage points (p = 0.014), also positive, meaning slightly more underweight in the treated group, which is consistent across all treatment contrasts and is a separately puzzling pattern. Employment shows no detectable effect (ATE = +0.010, p = 0.33). Education shows no detectable pooled effect (ATE = +0.002, p = 0.98).

### Robustness: the null is not a methodological artifact

Three features of the data and analysis could in principle produce a null even if the true effect on heterogeneity were positive. We address each in turn.

**Sample selection over 20 years.** With roughly 33% of the original cohort successfully tracked at the KLPS, the concern is specific: if deworming raised incomes and improved labor market outcomes, treated respondents may have been more mobile and therefore differentially difficult to locate, or alternatively easier to locate through formal employment records. Either pattern would mean the tracked sample differs by treatment status in ways that correlate with the outcomes we measure. The standard approach to this problem due to Lee (2009) constructs sharp nonparametric bounds on the treatment effect under the weakest possible identifying assumption: that treatment can only increase, or only decrease, the probability of being tracked. Lee bounds for BMI of [-0.10, +0.08] straddle zero under this assumption, confirming the null is not driven by differential tracking. We note that the monotonicity assumption itself may be violated if deworming-induced labor mobility made treated respondents differentially reachable; this cannot be ruled out but the bounds are informative conditional on the assumption.

**Age imputation.** Age at treatment is the primary moderator and is missing for roughly one in five respondents, imputed via MICE. If the imputation induced a spurious correlation between age and treatment assignment, the null on the age coefficient could reflect noise rather than an absence of signal. Restricting the analysis to the 4,232 respondents with observed year of birth who also have valid BMI data (4,209 total with observed year of birth in the KLPS-3 sample), all BLP age coefficients remain near zero and insignificant across all four outcomes. The null on age heterogeneity is not imputation-driven.

**Forest regularization.** The causal forest's minimum leaf size of 50 was chosen to prevent within-school overfitting, but it also constrains the forest's flexibility. If over-regularization were driving the null, a more flexible forest would find statistically significant heterogeneity. We re-ran the forest at minimum leaf sizes of 20 and 100. Mean CATE is stable across all leaf sizes, always close to the ATE. Directional patterns, including the BMI gender gap (female minus male CATE: -0.35 at leaf=20, -0.33 at leaf=50, and -0.21 at leaf=100) and the underweight age gradient (+0.019 to +0.020), are stable across the full regularization range. A leaf size of 20 provides maximum flexibility and still does not produce significant heterogeneity. The null is robust to the full range of forest regularization choices.

**The pooled BMI result is a treatment-timing artifact.** The pooled ATE on BMI of -0.24 (p = 0.001) was apparently significant, which is puzzling given the expected direction. When the analysis is restricted to the cleanest treatment contrast (full-treatment schools versus comparison schools only, excluding partial-treatment schools), the BMI ATE drops to -0.07 (p = 0.40), while the partial-treatment effect alone is -0.22 (p = 0.003). The significant pooled estimate of -0.24 conflates these distinct treatment timings, and it is driven almost entirely by the partial-treatment group. This inverts the natural expectation: if childhood deworming genuinely improves long-run nutritional status, the group that received two full years of treatment should show at least as large an effect as the group that received one year. The fact that the opposite is true is the clearest evidence that the pooled BMI result reflects the partial-treatment group's distinct exposure history rather than a genuine nutritional effect of deworming. These pupils were untreated in 1998 while exposed to spillovers from neighboring full-treatment schools, and received treatment only in 1999: a contaminated counterfactual that the pooled estimand cannot separate.

### Statistical power

The school-level minimum detectable effect for this design, with 48 treated schools, 25 comparison schools, average cluster size of 68, and unconditional ICC of 0.023, is 0.41 BMI points (0.13 standard deviations) at 80% power. Under more conservative ICC assumptions of 0.05 and 0.08, the MDE rises to 0.54 and 0.66 BMI points respectively. The 2:1 allocation of schools to treated versus comparison increases the MDE by approximately 15% relative to a balanced design with equal cluster counts per arm: under balanced allocation with 36-37 schools per arm and the same total sample, the MDE at conservative ICC of 0.05 would be approximately 0.45 BMI points rather than 0.54. The observed ATE of -0.24 BMI points falls below the MDE under all but the most favorable assumptions.

The prior expectation was a positive effect: childhood deworming reduces iron malabsorption and caloric competition from worm burden, which should improve nutritional status. Any true effect in the range of approximately [-0.54, +0.54] BMI points is statistically consistent with what we observe under conservative assumptions. The null result on BMI should therefore be read as uninformative about small true effects rather than as evidence against a nutritional pathway. The design was adequate for detecting the large labor market effects documented by Hamory et al. (2021), for which the relevant outcomes have higher signal-to-noise ratios, but not for detecting plausible nutritional effect sizes at a 20-year horizon.

### Directional signals: promising but fragile

Against the backdrop of the confirmed null, three patterns are directionally consistent with theory across multiple specifications, though none survives formal inference. We report them here as directions for future work rather than findings.

**Gender and employment under the cleanest treatment contrast.** When the analysis is restricted to full-treatment versus comparison schools, the BLP on employment shows a female coefficient of +0.41 (p = 0.03) and an age-times-female interaction of -0.03 (p = 0.03). The direction is consistent with the competing-demands hypothesis: childhood deworming increased employment more for women, concentrated among those who were youngest at treatment. The estimated differential treatment effect by gender is 41 percentage points: conditional on age, childhood deworming is estimated to increase employment by 41 percentage points more for women than for men. Both signals are absent in the pooled sample and rely on 50 schools and 3,406 employment observations, roughly two-thirds of the pooled sample, making these specification-dependent results. They do not warrant a policy recommendation on their own, but they suggest that a larger, better-powered study examining gender-differentiated labor market effects would be worth running.

**Underweight age gradient across all specifications.** The BLP coefficient on age at treatment for underweight prevalence is consistently negative across the pooled analysis, the full-treatment-only analysis, and the complete-case analysis (p = 0.12-0.13 in all three). A consistently negative coefficient means that children who were younger at treatment show lower adult underweight prevalence, the direction predicted by the developmental-window hypothesis for nutritional outcomes. This pattern never reaches conventional significance but its consistency across specifications that use different samples and make different imputation choices is the most stable directional signal in the project. A study with larger cluster counts would be well-positioned to test this hypothesis with adequate power.

**Education under the cleanest treatment contrast.** Full-treatment versus comparison schools shows a positive education ATE of +0.29 years (p = 0.015 uncorrected; note this is a subsample analysis not factored into the pre-specified Romano-Wolf correction, n = 1,909). This result is absent in the pooled sample and relies on a subsample of 50 schools with wide standard errors, so it should be treated as potentially reflecting sampling variability. We note it because it is in the expected direction and the magnitude (roughly one additional third of a year of schooling) is consistent with what the long-run returns literature would predict from an effective early health intervention.

---

## Limitations

The estimates throughout are intention-to-treat effects of school assignment to deworming, not effects of deworming receipt. Compliance varied across schools and the externality mechanism means comparison-group pupils were not entirely unexposed.

Without baseline health data for the comparison group (stool examination and hemoglobin data are available only for treatment schools), we cannot decompose the total effect into health-mediated versus education-mediated pathways. The age-at-treatment gradient is consistent with a critical-period mechanism operating through early nutritional improvement, but we cannot rule out alternative pathways operating through schooling quality, peer effects, or labor market entry timing.

The KLPS was designed primarily to measure long-run economic outcomes, not to serve as a heterogeneity study. Adult anthropometric outcomes were added to a survey with a different primary focus and the sample was not powered for them. A study designed from the outset to detect nutritional heterogeneity by age and gender would require substantially larger cluster counts than the 73 schools available here.

---

## Extensions

**Intergenerational effects.** The KLPS-4 Kids modules contain cognitive assessments, teacher evaluations, and birthweight data for the children of original PSDP respondents. These open a natural extension to intergenerational effects: did childhood deworming affect the next generation? The heterogeneity framework developed here is directly suited to that question: who among the original respondents transmitted the largest gains to the next generation, and whether the age-at-treatment gradient persists into the intergenerational channel.

**Bayesian hierarchical nuisance models.** The MixedLM sensitivity check in this analysis asks whether partial pooling across schools in the nuisance functions changes the substantive conclusions. A natural next step replaces REML random intercepts with a fully Bayesian hierarchical specification, propagating school-level uncertainty through the orthogonal scores and into the BLP standard errors. This connects to the partial pooling framework in the companion [bayesian-segmentation](https://github.com/ruschenpohler/bayesian-segmentation) repository, where hierarchical shrinkage is applied to a related heterogeneity problem in a consumer segmentation context.

---

## Reproducibility

All raw data files are verified against SHA-256 hashes before any analysis runs. Every dataset transformation is logged with a content-addressed hash. Every specification choice is recorded in `impl-log.jsonl` before the relevant analysis runs, providing a verifiable audit trail rather than a reconstruction from memory. Before fitting any estimator to the data, nuisance model hyperparameters were set and logged; no choices were made after seeing results.

The analysis is fully reproducible from the raw data files via `make all`. Dependencies are pinned in `pyproject.toml` and managed via `uv`.

**Data citation:** Miguel, Edward; Kremer, Michael, 2014, "Replication data for: Worms: Identifying Impacts on Education and Health in the Presence of Treatment Externalities", https://doi.org/10.7910/DVN/28038, Harvard Dataverse, V2.

---

## Related work

For the Bayesian hierarchical extension of the heterogeneity analysis developed here, and for a worked treatment of partial pooling across groups in a related context, see the Extensions section and the companion [bayesian-segmentation](https://github.com/ruschenpohler/bayesian-segmentation) repository.

A related project applying double/debiased ML to the Kenya Life Panel Survey to estimate long-run treatment effects with heterogeneity by age at treatment and gender is available at [REPO LINK].

---

## References

Aiken, A. M., Davey, C., Hargreaves, J. R., and Hayes, R. J. (2015). Re-analysis of health and educational impacts of a school-based deworming programme in western Kenya: a pure replication. *International Journal of Epidemiology*, 44(5), 1572-1580.

Athey, S., Tibshirani, J., and Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*, 47(2), 1148-1178.

Baird, S., Hicks, J. H., Kremer, M., and Miguel, E. (2016). Worms at work: Long-run impacts of a child health investment. *Quarterly Journal of Economics*, 131(4), 1637-1680.

Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-Based Improvements for Inference with Clustered Errors. *Review of Economics and Statistics*, 90(3), 414-427.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., and Newey, W. (2017). Double/Debiased/Neyman Machine Learning of Treatment Effects. *American Economic Review: Papers and Proceedings*, 107(5), 261-265.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

Davey, C., Aiken, A. M., Hayes, R. J., and Hargreaves, J. R. (2015). Re-analysis of health and educational impacts of a school-based deworming programme in western Kenya: a statistical replication of a cluster quasi-randomized stepped-wedge trial. *International Journal of Epidemiology*, 44(5), 1581-1592.

Hamory, J., Kremer, M., Miguel, E., Moulin, M., and Null, C. (2021). Twenty-year economic impacts of deworming. *Proceedings of the National Academy of Sciences*, 118(14), e2023185118.

Kleiner, A., Talwalkar, A., Sarkar, P., and Jordan, M. I. (2014). A scalable bootstrap for massive data. *Journal of the Royal Statistical Society: Series B*, 76(4), 795-816.

Lee, D. S. (2009). Training, Wages, and Sample Selection: Estimating Sharp Bounds on Treatment Effects. *Review of Economic Studies*, 76(3), 1071-1102.

Miguel, E., and Kremer, M. (2004). Worms: Identifying Impacts on Education and Health in the Presence of Treatment Externalities. *Econometrica*, 72(1), 159-217.

Semenova, V. (2025). Generalized Lee Bounds. *Journal of Econometrics*, 251, 106055.

Wager, S., and Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects Using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.

Wager, S., Hastie, T., and Efron, B. (2014). Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife. *Journal of Machine Learning Research*, 15, 1625-1651.
