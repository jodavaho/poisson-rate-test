# poisson-rate-test
Rust repo that provides a robust poisson-rate hypothesis test, returning p -values for the probability that two observed poisson data sets are different.

The intended use of this small library is to test whether two events are occuring at the same rate.

Another test statistic of interst in games is the ratio of events (such as Kills / Deaths). For this, I provide a bootstrap method to estimate p-values in an A/B test of the ratio of two events.

I use it in [kda-tools](https://github.com/jodavaho/kda-tools) for hypothesis testing loadouts in video games.

Here's a few examples, see more in [the docs](https://docs.rs/poisson-rate-test/)

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

```rust
fn test_two_diff_bootstrap_parametric(){
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

    }
```
