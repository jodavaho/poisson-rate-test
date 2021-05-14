use statrs::distribution::{Discrete as _, Poisson as statrs_Poisson};
use statrs::distribution::Univariate as _;
use statrs::function::gamma::{gamma_li, gamma};

/// Return the version string for the current version of the library
pub fn version()->String{
    return env!("CARGO_PKG_VERSION").to_string();
}

pub mod bootstrap{

    pub mod param{

        use rand_distr::Poisson as rand_Poisson;
        use rand_distr::Distribution;
        use rayon::prelude::{ IntoParallelIterator, ParallelIterator};

        /// Hypothesis test R1/R2 under condition 1 > R1/R2 without constraint 
        /// and return the p-value.  where R1 = #events of type 1 / #trials and
        /// R2 = #events of type 2 / #trials 
        /// 
        /// We assume two populations, one of which had some difference, and
        /// we're trying to see if it affected the ratio of the occurances of
        /// these two events.
        /// 
        /// We're using boostrap methods (computationally more expensive, but
        /// not bad)
        /// 
        /// One caveat: If # events_1 and # events_2 under treatment are both 0
        /// (neither events occurred under treatment), that is considered a
        /// viable event and is compared to the occurances of 0/0 in baseline
        /// samples.
        /// 
        /// 
        /// *Example*
        /// ```rust
        /// use claim::{assert_lt,assert_gt};
        /// let base_a = vec![1,1,1,1, 1,1,2,1];
        /// let base_b = vec![1,0,1,1, 0,1,1,1];
        /// let treat_a = vec![1,1,1,1, 1,1,2,1];
        /// let treat_b = vec![1,0,0,0, 0,0,0,0];
        /// //Did treatment increase ratio of a/b?
        /// let p = poisson_rate_test::bootstrap::param::ratio_events_greater_pval(
        ///     base_a.iter().sum::<usize>(),
        ///     base_b.iter().sum::<usize>(),
        ///     base_a.len() as usize,
        ///     treat_a.iter().sum::<usize>(),
        ///     treat_b.iter().sum::<usize>(),
        ///     treat_a.len() as usize,
        /// );
        /// assert_lt!(p.unwrap(),0.05); //<--confidently yes
        /// assert_gt!(p.unwrap(),0.001);
        /// ```
        pub fn ratio_events_greater_pval(
            num_events_one_baseline:usize,
            num_events_two_baseline:usize,
            num_baseline_group:usize,
            num_events_one_treatment:usize,
            num_events_two_treatment:usize,
            num_treatment_group:usize,
        ) -> Result<f64, &'static str>
        {
            return  ratio_events_greater_pval_n(num_events_one_baseline,
            num_events_two_baseline, num_baseline_group,
            num_events_one_treatment, num_events_two_treatment,
            num_treatment_group, 1000);
        }

        /// Hypothesis test R1/R2 under condition 1 > R1/R2 without constraint 
        /// and return the p-value.  where R1 = #events of type 1 / #trials and
        /// R2 = #events of type 2 / #trials 
        /// 
        /// We assume two populations, one of which had some difference, and
        /// we're trying to see if it affected the ratio of the occurances of
        /// these two events.
        /// 
        /// We're using boostrap methods (computationally more expensive, but
        /// not bad)
        /// 
        /// One caveat: If # events_1 and # events_2 under treatment are both 0
        /// (neither events occurred under treatment), that is considered a
        /// viable event and is compared to the occurances of 0/0 in baseline
        /// samples.
        pub fn ratio_events_greater_pval_n(
            num_events_one_baseline:usize,
            num_events_two_baseline:usize,
            num_baseline_group:usize,
            num_events_one_treatment:usize,
            num_events_two_treatment:usize,
            num_treatment_group:usize,
            num_samples:usize,
        ) -> Result<f64, &'static str>
        {
            if num_events_one_treatment == 0 && num_events_two_treatment == 0 {
                return Err("Err creating test statistic: Treatment event ratio 0/0 is uncomparable")
            }
            if num_events_one_baseline == 0 {
                return Err("Event one does not occur in baseline, assumption of poisson distribution is violated");
            }
            if num_events_two_baseline == 0 {
                return Err("Event two does not occur in baseline, assumption of poisson distribution is violated");
            }
            let t_stat:f64 = match (num_events_one_treatment,num_events_two_treatment)
            {
             // (0,0)=> num_events_one_treatment as f64 /num_events_two_treatment as f64  
             //       - num_events_one_baseline as f64  /num_events_two_baseline  as f64,
                (_,_)=> num_events_one_treatment as f64 /num_events_two_treatment as f64  
                      - num_events_one_baseline as f64  /num_events_two_baseline  as f64,
            };
            //we're going to try to handle infinities ... 
            //generate baseline distribution of size num_baseline_group + num_treatment_group
            //seperate and calculate # events 1 and 2 for both populations
            //
            let rate_one_baseline:f64 = num_events_one_baseline as f64 / num_baseline_group as f64;
            let rate_two_baseline:f64 = num_events_two_baseline  as f64/ num_baseline_group as f64;
            let p_one = rand_Poisson::new(rate_one_baseline).unwrap();
            let p_two = rand_Poisson::new(rate_two_baseline).unwrap();
            let p_val = (0..num_samples).into_par_iter().map(|_| 
                {
                let occ_b_one:f64 = p_one.sample_iter(&mut rand::thread_rng()).take(num_baseline_group).sum();
                let occ_b_two:f64 = p_two.sample_iter(&mut rand::thread_rng()).take(num_baseline_group).sum();
                let occ_t_one:f64 = p_one.sample_iter(&mut rand::thread_rng()).take(num_treatment_group).sum();
                let occ_t_two:f64 = p_two.sample_iter(&mut rand::thread_rng()).take(num_treatment_group).sum();
                let ti = occ_t_one / occ_t_two - occ_b_one/occ_b_two;
                if occ_t_two == 0.0 {
                    return 1.0/(num_samples as f64)
                }else if ti>t_stat{
                    return 1.0/(num_samples as f64)
                } 0.0
            }
            ).sum();
            Ok(p_val)
        }
    }

}

///Returns the p-value of the two-tailed hypothesis test r1/r2 != 1.0 Or, tests
///the equality of the rates of two poisson processes, by their observed
///parameters (sum of events and # trials)
/// 
/// **Example: Testing the rate of one set of data**
/// ```rust
/// use poisson_rate_test::two_tailed_rates_equal;
/// let data = vec![0,1,1,0];
/// let n1 = data.len() as f64;
/// let sum1 = data.iter().sum::<usize>() as f64;
/// //are these rates equal to my hypothesized rate of 0.5?
/// let expected_n = n1;
/// let expected_sum = 0.5 * n1;
/// let p = two_tailed_rates_equal(sum1, n1, expected_sum, expected_n);
/// assert!(p>0.99); //<--confidently yes
/// ```
pub fn two_tailed_rates_equal(
    num_events_one:f64,
    t_one:f64,
    num_events_two:f64,
    t_two:f64)-> f64 {
        let p_1 = one_tailed_ratio(num_events_one, t_one, num_events_two, t_two, 1.0);
        let p_2 = one_tailed_ratio(num_events_one, t_one, num_events_two, t_two, 1.0);
        return 2.0 * (p_1.min(p_2));
    }


/// Conducts a likelihood ratio test (LHR) to determine if the two rates, r1 and
/// r2, satisfy R>=r1/r2 for a ratio R.
/// 
/// The returned value is the probability of observing these two data sets given
/// that r1/r2=R
/// 
/// See:
/// >Gu, Ng, Tuang, Schucany 2008 "Testing the Ratio of Two Poisson Rates"`
/// 
/// **Example: Testing two unequal rate events**
/// ```rust
/// use poisson_rate_test::one_tailed_ratio;
/// 
/// let occurances_observed = vec![0,0,1,0];
/// let occurances_other = vec![1,1,5,3,3,9];
/// let n1 = occurances_observed.len() as f64;
/// let n2 = occurances_other.len() as f64;
/// let sum1 = occurances_observed.iter().sum::<usize>() as f64;
/// let sum2 = occurances_other.iter().sum::<usize>() as f64;
/// //are these rates the same (e.g. 1.0 is the ratio of rates)?
/// let p = one_tailed_ratio(sum1, n1, sum2, n2, 1.0);
/// assert!(p<0.01); //<--confidently no
/// ```
/// 
/// 
pub fn one_tailed_ratio(
    num_events_one:f64,
    t_one:f64,
    num_events_two:f64,
    t_two:f64,
    h0_rate_ratio:f64) -> f64
    {

        assert!(num_events_one>0.0 || num_events_two>0.0,"We cannot test without some events occurring (parameter 1 and 3 were 0)");
        //by Gu 2008 Testing Ratio of Two Poisson Rates
        //use magic factor R/d, w/ d=t0/t1 (the # trials, I guess ... )
        //so r0 is "rate of kills with gear" and r1 is "rate of kills w/o gear"
        // so t0 is # of games w/ gear, and t1 is #games without gear
        //calculate the expected rates under the null hypothesis
        //Generally, here, using the funny constants, but in the case R=1 and t1=t0, it's simpler.
        //magic constant #1, d = t1/t0
        let magic_d = t_two / t_one ;
        //magic constant #2, g = R/d
        let magic_g = h0_rate_ratio / magic_d ;
        //now using magic constants, calculate the "hypothesis constarained rates"
        //i.e., the expected rates given H0 is true
        if cfg!(debug_assertions){
            eprintln!("magic d: {} and g: {}", magic_d, magic_g);
        }
        let hypothesized_rate_one = (num_events_one + num_events_two) / (t_one * (1.0+1.0/magic_g));
        let hypothesized_rate_two = (num_events_one + num_events_two)/ (t_two * (1.0+magic_g));
        debug_assert!(hypothesized_rate_one>0.0);
        //debug_assert!(hypothesized_rate_two>0.0);

        let mut p_val = 0.0;

        let obs_rate_one = num_events_one  / t_one ;
        let obs_rate_two = num_events_two   / t_two ;
        if cfg!(debug_assertions){
            eprintln!("hyp rate 1 : {}, hyp rate 2 : {}",hypothesized_rate_one ,hypothesized_rate_two );
            eprintln!("obs 1 : {},  obs 2 : {}",obs_rate_one,obs_rate_two);
        }
        if obs_rate_one == 0.0
        {
            //specific case of probability 0 | non-group rate and only t_one trials
            p_val = statrs_Poisson::new(obs_rate_two * t_one ).unwrap().pmf(0);
        } else if t_one>0.0 && t_two > 0.0{

            //OK so if you follow through magic_g, under the case R=1, t0 == t1, it all cancels out nicely. 
            let maximum_likelihood_h0:f64 = 
                statrs_Poisson::new(hypothesized_rate_one * t_one  ).unwrap().pmf(num_events_one as u64)
                * statrs_Poisson::new(hypothesized_rate_two * t_two ).unwrap().pmf(num_events_two as u64);
            let maximum_likelihood_unconstrained:f64 = 
                statrs_Poisson::new(obs_rate_one * t_one ).unwrap().pmf(num_events_one as u64)
                * statrs_Poisson::new(obs_rate_two * t_two ).unwrap().pmf(num_events_two as u64);
            let lhr =  maximum_likelihood_h0 / maximum_likelihood_unconstrained;
            if lhr == 1.0{
                p_val = 0.5;//chi-square-cdf(x-->0) --> 0
            } else {
                let test_statistic:f64 = -2.0*(maximum_likelihood_h0 / maximum_likelihood_unconstrained).ln();
                //of course this doesn't work:
                //let p_val = ( 1-ChiSquared::new(1.0).unwrap().checked_inverse_cdf(test_statistic) );
                p_val =  0.5 * (1.0- gamma_li(0.5,test_statistic ) / gamma(0.5) );
            }
        }
        p_val
    }

    #[deprecated]
    #[allow(dead_code)]
    /**
     * **Deprecated**
     * 
     * This method, for R1 = X1/T1 and R2 = X2/T2, checks p(X>X1 | T1=t1,
     * X2=x2, T2=t2). It does not check p(R1>R2), which is what I thought it
     * did.  The problem is the use of the assumption that T1 is known. 
     * 
     * It's kept here for completeness, but never use it.
     */
fn one_tailed_n(
    num_events_one:f64,
    t_one:f64,
    num_events_two:f64,
    t_two:f64,
    h0_rate_ratio:f64) -> f64
    {

        assert!(num_events_one>0.0 || num_events_two>0.0,"We cannot test without some events occurring (parameter 1 and 3 were 0)");
        let mut p_val = 0.0;

        let obs_rate_one = num_events_one  / t_one ;
        let obs_rate_two = num_events_two   / t_two ;
        if obs_rate_one == 0.0
        {
            //specific case of probability 0 | non-group rate and only t_one trials
            p_val = statrs_Poisson::new( h0_rate_ratio * obs_rate_two * t_one ).unwrap().pmf(0);
        } else if t_one>0.0 && t_two > 0.0{

            p_val = 1.0-statrs_Poisson::new(h0_rate_ratio * obs_rate_two * t_one).unwrap().cdf(num_events_one);
        }
        p_val
    }

#[cfg(test)]
mod tests{
use super::*;
use claim::{assert_lt,assert_gt};

    #[test]
    fn test_ones_side(){
        let p = one_tailed_ratio(1.0,1.0,1.0,1.0,1.0);
        //one sided test!
        assert_eq!(p,0.5);
    }

    #[test]
    fn test_two_sides_null_hypothesis_true(){

        //create data where rate1 == 1/2 * rate2
        let occurances_one = vec![1,1,1,1,1,1];
        let occurances_two = vec![2,2,2,2,2,2];
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

        //Two sided test
        let p_half = one_tailed_ratio(sum1, n1, sum2, n2, 0.49999);
        //other side
        let p_double = one_tailed_ratio(sum2, n2, sum1, n1, 2.0001);
        assert_gt!(2.0*p_half.min(p_double),0.99);
    }

    #[test]
    fn test_two_diff(){
        let occurances_observed = vec![0,0,1,0];
        let occurances_other = vec![1,1,5,3,3,8];
        let n1 = occurances_observed.len() as f64;
        let n2 = occurances_observed.len() as f64;
        let sum1 = occurances_observed.iter().sum::<usize>() as f64;
        let sum2 = occurances_other.iter().sum::<usize>() as f64;
        //are these rates the same?
        let p = one_tailed_ratio(sum1, n1, sum2, n2, 1.0);
        assert_lt!(p,0.01); //<--confidently no
    }

    #[test]
    fn test_two_diff_bootstrap_parametric(){
        let base_a = vec![0,0,1,0];
        let base_b = vec![1,0,1,1];
        let treat_a = vec![1,1,1,2];
        let treat_b = vec![1,1,1,1];
        //Did treatment increase ratio of a/b?
        let p = bootstrap::param::ratio_events_greater_pval_n(
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

        let base_a = vec![0,0,1,0, 1,0,0,0];
        let base_b = vec![1,0,1,1, 0,1,1,1];
        let treat_a = vec![1,1,1,2, 1,2,1,1];
        let treat_b = vec![1,1,1,1, 1,1,1,1];
        //Did treatment increase ratio of a/b?
        let p = bootstrap::param::ratio_events_greater_pval_n(
            base_a.iter().sum::<usize>(),
            base_b.iter().sum::<usize>(),
            base_a.len() as usize,
            treat_a.iter().sum::<usize>(),
            treat_b.iter().sum::<usize>(),
            treat_a.len() as usize,
            10000
        );
        assert_lt!(p.unwrap(),0.051); //<--confidently yes
        assert_gt!(p.unwrap(),0.01);

    }

    #[test]
    fn test_denom_decrease_boostrap_parametric(){
        let base_a = vec![1,1,1,1];
        let base_b = vec![1,0,1,1];
        let treat_a = vec![1,1,1,1];
        let treat_b = vec![1,0,0,0];
        //Did treatment increase ratio of a/b?
        let p = bootstrap::param::ratio_events_greater_pval(
            base_a.iter().sum::<usize>(),
            base_b.iter().sum::<usize>(),
            base_a.len() as usize,
            treat_a.iter().sum::<usize>(),
            treat_b.iter().sum::<usize>(),
            treat_a.len() as usize,
        );
        assert_lt!(p.unwrap(),0.15); //<--tentatively yes
        assert_gt!(p.unwrap(),0.05);

        let base_a = vec![1,1,1,1, 1,1,2,1];
        let base_b = vec![1,0,1,1, 0,1,1,1];
        let treat_a = vec![1,1,1,1, 1,1,2,1];
        let treat_b = vec![1,0,0,0, 0,0,0,0];
        //Did treatment increase ratio of a/b?
        let p = bootstrap::param::ratio_events_greater_pval(
            base_a.iter().sum::<usize>(),
            base_b.iter().sum::<usize>(),
            base_a.len() as usize,
            treat_a.iter().sum::<usize>(),
            treat_b.iter().sum::<usize>(),
            treat_a.len() as usize,
        );
        assert_lt!(p.unwrap(),0.05); //<--confidently yes
        assert_gt!(p.unwrap(),0.001);

        //gather more data
    }

    #[test]
    fn test_two_same(){
        let occurances_observed = vec![1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
        let occurances_other = vec![1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
        let n1 = occurances_observed.len() as f64;
        let n2 = occurances_observed.len() as f64;
        let sum1 = occurances_observed.iter().sum::<usize>() as f64;
        let sum2 = occurances_other.iter().sum::<usize>() as f64;
        let p_left  = one_tailed_ratio(sum1, n1, sum2, n2, 1.0);
        let p_right = one_tailed_ratio(sum2, n2, sum1, n1, 1.0);
        assert_eq!(1.0,2.0*p_left.min(p_right));
    }

    #[test]
    fn test_by_rate(){

        let data = vec![0,1,1,0];
        let n1 = data.len() as f64;
        let sum1 = data.iter().sum::<usize>() as f64;
        //are these rates equal to my hypothesized rate of 0.5?
        let expected_n = n1;
        let expected_sum = 0.5 * n1;
        let p = two_tailed_rates_equal(sum1, n1, expected_sum, expected_n);
        assert!(p>0.99); //<--confidently yes
    }

    #[test]
    fn test_readme_example(){
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

    }

    #[test]
    fn test_ones_side_compare(){
        let p = one_tailed_ratio(1.0,1.0,1.0,1.0,1.0);
        //one sided test!
        assert_eq!(p,0.5);
    }

    #[test]
    fn test_p_indep_of_magnitude(){
        use statrs::assert_almost_eq;
        let p_one = bootstrap::param::ratio_events_greater_pval_n(
            200,200,
            100,
            10,1,
            1,
            5000
        );  
        let p_ten = bootstrap::param::ratio_events_greater_pval_n(
            200,200,
            100,
            10000,1000,
            1,
            5000
        );  
        assert_almost_eq!(p_one.unwrap() , p_ten.unwrap(),  0.01 );
    }

    #[test]
    fn stress_test_never_fails(){
        let p = bootstrap::param::ratio_events_greater_pval_n(
            150,150,
            100,
            10,10,
            1,
            50000
        ) ;  
        //useless test
        assert_gt!(p.unwrap(),0.0);
    }

    #[test]
    fn test_n_does_change_likelihood(){
        let p_small = bootstrap::param::ratio_events_greater_pval_n(
            10,10,
            10,
            10,1,
            1,
            5000
        ) ;  
        let p_large = bootstrap::param::ratio_events_greater_pval_n(
            10,10,
            10,
            1000,100,
            100,
            5000
        ) ;  
        //small n is just more likely to produce outliers. The p-value will be
        //larger despite a standout performance vs one that is repeated a lot
        assert_gt!(p_small.unwrap(),0.30);
        assert_lt!(p_large.unwrap(),0.05); 

        let p_small = bootstrap::param::ratio_events_greater_pval_n(
            10,10,
            10,
            10,1,
            1,
            5000
        ) ;  
        let p_large = bootstrap::param::ratio_events_greater_pval_n(
            1000,1000,
            1000,
            10,1,
            1,
            5000
        ) ;  
        // However, the baseline doesn't matter as much. Outliers unsupported by
        // evidence are easy to reject
        assert_gt!(p_small.unwrap(),0.30);
        assert_gt!(p_large.unwrap(),0.30);
    }
    #[test]
    fn test_rayon_example(){
        use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
        let p = (0..25usize).into_par_iter()
                        .zip(0..25usize)
                        .filter(|&(x, y)| x % 5 == 0 || y % 5 == 0)
                        .map(|(x, y)| x * y)
                        .sum::<usize>();

        let s = (0..25usize).zip(0..25)
                        .filter(|&(x, y)| x % 5 == 0 || y % 5 == 0)
                        .map(|(x, y)| x * y)
                        .sum();

        assert_eq!(p, s);
    }

    #[test]
    fn jp_caldwell_conversion_data(){
        //57 matches, 50 kills, 27 deaths without (baseline)
        //10 matches, 4 kills, 9 deaths with (treatment)
        let p_cc= bootstrap::param::ratio_events_greater_pval(
            50,27, 47,
            4,9, 10,
        ) ;  
        eprintln!("{}",p_cc.unwrap());
        assert!(false);
    }
}
