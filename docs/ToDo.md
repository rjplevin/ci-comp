# To Do

## Analysis

### Setup

* Create configuration file with entries for versions 4.3, 4.4.1, and 5.1.2
  * perhaps `[ci-4.3]`, `[ci-4.4]` and `[ci-5.1]`?
  * Have the files for each GCAM  version in a subdirectory, e.g., `ci-comp/4.3/etc` and `ci-comp/5.1/mcs`

* Separate project.xml files to handle differences in names of queries used for CI calculation

* Decide whether to rely only on parameters common to all versions, or to use version-specific features.
  * Using common parameters simplifies comparison across versions
  * Could use `<Constant value="0"/>` distribution for models lacking a given parameter

### Scenarios

* Key question is how many biofuels to evaluate. If the focus is on uncertainty, just one
  fuel should suffice. 

* Slightly more interesting to evaluate 2 fuels since changes by version may vary.

### Monte Carlo Simulation

* Start with 1000 trials and verify that correlations have converged

* Combine all results into one data set to compare model choice against other parameters
  * Use the same set of variables, in the same order, for each MCS, so one set of trial data 
    can be used
  * Create a pseudo-input that is the GCAM version (could use actual version numbers)
  * Write a script to dump all results for each version of GCAM and combine them into
    a single pseudo-MCS (shared inputs, combined outvalues) that can be analyzed in explorer.

#### Variables

Start with subset of variables used in EPA analyses.

#### Model outputs

1. CI
2. Fuel market rebound
3. RF timeseries (one plot showing timeseries with error bands for each model version for each fuel)

### Other figures

* Show key results for static runs, e.g., showing where LUC and LUC emissions occur 
  in each version.
  * Maps showing difference between AEZ and basin versions might be interesting.


## Post acceptance

### Replication tools

* Create a script to download the required versions of GCAM

* Document how to run the full exercise


