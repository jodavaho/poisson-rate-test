# poisson-rate-test

## Purpose

A Rust library that provides a methods for comparing the rates of poisson data and conducing hypothesis tests about that data.

Specifically, two types of tests are provided as of 1.0, Rate-to-rate comparisons (2 events), and ratio-to-ration comparisons (4 events).

## Rate-to-rate

This tests the hypothesis that the number of events A and the number of events B in a given set of data have a rate of the form `r_a / r_b >= R`, for a constant R against the null hypothesis that the two events occur with the same rate.

### Example: Testing the rate of events against a hypothesis

```rust
use poisson_ratio_test::two_tailed_rates_equal;
//make some data that sure looks like it occurs with rate = 0.5;
let data = vec![0,1,1,0]; //note, 0,2,0,0 would be the same (2/4).
let n1 = data.len() as f64;
let sum1 = data.iter().sum::<usize>() as f64;
//are these rates equal to my hypothesized rate of 0.5?
let expected_n = n1;
let expected_sum = 0.5 * n1;
let p = two_tailed_rates_equal(sum1, n1, expected_sum, expected_n);
assert!(p>0.99); //<--confidently yes
```

### Example, comparing the rate of events under a new condition

```rust
use claim::{assert_lt,assert_gt};
use poisson_ratio_test::{one_tailed_ratio,two_tailed_rates_equal};
//say we made a change, and observed the new rates 
let occurances_observed = vec![0,0,1,0];
//and here's the "usual" data
let occurances_usual = vec![1,1,5,3,3,8];
//need the basic n/sum statistics
let n1 = occurances_observed.len() as f64;
let n2 = occurances_usual.len() as f64;
let sum1 = occurances_observed.iter().sum::<usize>() as f64;
let sum2 = occurances_usual.iter().sum::<usize>() as f64;
//is rate of observed > rate usual?
let p = one_tailed_ratio(sum1, n1, sum2, n2, 1.0);
assert_lt!(p,0.01); //<--confidently no

//Maybe just check both tails to be sure (this tests r observed / r baseline != 1)
let p = two_tailed_rates_equal(sum1, n1, sum2, n2);
assert_lt!(p,0.01); //<--confidently no
```
### Example, more data helps:

Here's a long example, see more in [the docs](https://docs.rs/poisson-rate-test/)

```rust
use claim::{assert_lt,assert_gt};
use poisson_ratio_test::{one_tailed_ratio,two_tailed_rates_equal};

//create data where rate1 == 1/2 * rate2
let occurances_one = vec![1,0,1,0,1,0];
let occurances_two = vec![1,1,1,1,0,2];
let n1 = occurances_one.len() as f64;
let n2 = occurances_two.len() as f64;
let sum1 = occurances_one.iter().sum::<usize>() as f64;
let sum2 = occurances_two.iter().sum::<usize>() as f64;

//test hypothesis that r1/r2 > 1/2
let p = one_tailed_ratio(sum1, n1, sum2, n2, 0.5);
assert_eq!(p, 0.50); //<-- nope
//let's test the neighbordhood around that
let p = one_tailed_ratio(sum1, n1, sum2, n2, 0.49999 );
assert_gt!(p, 0.49); //<-- still nope

//Two sided test. What is the likelihood of seeing the data we got
//given that r1/r2 == 1/2?
let p_half = one_tailed_ratio(sum1, n1, sum2, n2, 0.49999);
//other side
let p_double = one_tailed_ratio(sum2, n2, sum1, n1, 2.0001);
//just about 1.0!
assert_gt!(2.0*p_half.min(p_double),0.99);

//we *know* they are not equal, but can we prove it in general?
let mut p_double = two_tailed_rates_equal(sum2, n2, sum1, n1);
//note: p_double is in [.15,.25]
assert_lt!(p_double,0.25);//<--looking  unlikely... maybe more data is required
assert_gt!(p_double,0.15);//<--looking  unlikely... maybe more data is required

//get more of the same data
let trial2_one = vec![1,0,1,0,1,0,1,0,1,0,1,0,1,0];
let trial2_two = vec![1,1,1,1,0,2,0,2,1,1,0,2,1,1];
let t2n1 = trial2_one.len() as f64;
let t2n2 = trial2_two.len() as f64;
let t2sum1 = trial2_one.iter().sum::<usize>() as f64;
let t2sum2 = trial2_two.iter().sum::<usize>() as f64;
p_double = two_tailed_rates_equal(t2sum2, t2n2, t2sum1, t2n1);
assert_lt!(p_double,0.05);//<--That did the trick
```

# Comparing ratio of events

Suppose there's two events, a and b. And we have two groups (base and
treatment). We changed something in treatment, and want to know if that change
affected the ratio of a/b. So, we count up a and b for both baseline and
treatment.  note the p -vals are estimated from simulation, so they might
change a little (as in 0.01 or so) between different runs. Pass in a higher
sample count to stabilize, at the expense of cpu cost.

## Example: Comparing a new weapon in Hunt Showdown

This is how it's done in [kda-tools](https://github.com/jodavaho/kda-tools) 

```rust
use poisson_rate_test::bootstrap::param::ratio_events_greater_pval;
use claim::{assert_lt,assert_gt};
//57 matches, 50 kills, 27 deaths without Caldwell Conversion pistol (baseline)
let normal_matches = 57;
let normal_kills = 50;
let normal_deaths = 27;
//10 matches, 4 kills, 9 deaths with Caldell Conversion pistol (treatment)
let cc_matches=10;
let cc_kills=4;
let cc_deaths=9;

let p_cc_treatment_greater= bootstrap::param::ratio_events_greater_pval(
    normal_kills,normal_deaths, normal_matches,
    cc_kills,cc_deaths, cc_matches,
).unwrap() ;
assert_gt!(p_cc_treatment_greater,0.90); //Hell no that's not greater (cc_kills/cc_deaths) is much less than normal_kills/normal_deaths
let p_cc_treatment_less = bootstrap::param::ratio_events_greater_pval(
    cc_kills,cc_deaths, cc_matches,
    normal_kills,normal_deaths, normal_matches,
).unwrap() ;  
assert_lt!(p_cc_treatment_less,0.05); //very high significance / very low p-value
```

```rust
use poisson_rate_test::boostrap::param::ratio_events_equal_pval_n;
use claim::{assert_lt,assert_gt};
let base_a = vec![0,0,1,0];
let base_b = vec![1,0,1,1];
let treat_a = vec![1,1,1,2];
let treat_b = vec![1,1,1,1];
//Did treatment increase ratio of a/b?
let p = bootstrap::param::ratio_events_equal_pval_n(
    base_a.iter().sum::<usize>(),
    base_b.iter().sum::<usize>(),
    base_a.len() as usize,
    treat_a.iter().sum::<usize>(),
    treat_b.iter().sum::<usize>(),
    treat_a.len() as usize,
    10000
);
assert_lt!(p.unwrap(),0.15); //<--tentatively yes
assert_gt!(p.unwrap(),0.05);

//just need more data, right?
let base_a = vec![0,0,1,0, 1,0,0,0];
let base_b = vec![1,0,1,1, 0,1,1,1];
let treat_a = vec![1,1,1,2, 1,2,1,1];
let treat_b = vec![1,1,1,1, 1,1,1,1];
//Did treatment increase ratio of a/b?
let p = bootstrap::param::ratio_events_equal_pval_n(
    base_a.iter().sum::<usize>(),
    base_b.iter().sum::<usize>(),
    base_a.len() as usize,
    treat_a.iter().sum::<usize>(),
    treat_b.iter().sum::<usize>(),
    treat_a.len() as usize,
    10000
);
assert_lt!(p.unwrap(),0.05); //<--confidently yes 
assert_gt!(p.unwrap(),0.01);
```

## Ratio to ratio

This tests the hypothesis that two events occur with different ratios in two datasets `r1_a/r2_b >= r2_a/r2_b` against the null hypothesis that they are equal. 

## Why

A test statistic of interst in games is the ratio of events (such as Kills /
Deaths for various loadouts), or rates of kills / match with and without items. 

I use it in [kda-tools](https://github.com/jodavaho/kda-tools) for hypothesis testing loadouts in Hunt Showdown.
