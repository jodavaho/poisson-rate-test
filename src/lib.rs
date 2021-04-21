use statrs::distribution::Discrete;
use statrs::distribution::Poisson;
use statrs::function::gamma::{gamma_li, gamma};

pub fn version()->String{
    "0.1.0".to_string()
}
pub fn poisson_lhr_test(
    sum_metric_group:f64,
    n_group:usize,
    sum_metric_non_group:f64,
    n_non_group:usize,
    h0_rate_ratio:f64) -> f64
    {

        //by Gu 2008 Testing Ratio of Two Poisson Rates
        //use magic factor R/d, w/ d=t0/t1 (the # trials, I guess ... )
        //so r0 is "rate of kills with gear" and r1 is "rate of kills w/o gear"
        // so t0 is # of games w/ gear, and t1 is #games without gear
        //calculate the expected rates under the null hypothesis
        //Generally, here, using the funny constants, but in the case R=1 and t1=t0, it's simpler.
        //magic constant #1, d = t1/t0
        let magic_d = n_non_group as f64 / n_group as f64;
        //magic constant #2, g = R/d
        let magic_g = h0_rate_ratio as f64 / magic_d as f64;
        //now using magic constants, calculate the "hypothesis constarained rates"
        //i.e., the expected rates given H0 is true
        if cfg!(debug_assertions){
            eprintln!("magic d: {} and g: {}", magic_d, magic_g);
        }
        let hypothesized_rate_group = (sum_metric_group + sum_metric_non_group) as f64/ (n_group as f64 * (1.0+1.0/magic_g));
        let hypothesized_rate_non_group = (sum_metric_group + sum_metric_non_group) as f64 / (n_non_group as f64 * (1.0+magic_g));
        debug_assert!(hypothesized_rate_group>0.0);
        //debug_assert!(hypothesized_rate_non_group>0.0);
        if cfg!(debug_assertions){
            eprintln!("hyp rate w/ : {}, hyp rate w/o : {}",hypothesized_rate_group ,hypothesized_rate_non_group );
            eprintln!("metric w/ : {},  metric w/o : {}",sum_metric_group,sum_metric_non_group);
        }

        let mut p_val = 0.0;

        let obs_rate_group = sum_metric_group as f64 / n_group as f64;
        let obs_rate_non_group = sum_metric_non_group as f64  / n_non_group as f64;
        if obs_rate_group == 0.0
        {
            //specific case of probability 0 | non-group rate and only n_group trials
            p_val = Poisson::new(obs_rate_non_group * n_group as f64).unwrap().pmf(0);
        } else if n_group>0 && n_non_group > 0{

            //OK so if you follow through magic_g, under the case R=1, t0 == t1, it all cancels out nicely. 
            let maximum_likelihood_h0:f64 = 
                Poisson::new(hypothesized_rate_group * n_group  as f64).unwrap().pmf(sum_metric_group as u64)
                * Poisson::new(hypothesized_rate_non_group * n_non_group as f64).unwrap().pmf(sum_metric_non_group as u64);
            let maximum_likelihood_unconstrained:f64 = 
                Poisson::new(obs_rate_group * n_group as f64).unwrap().pmf(sum_metric_group as u64)
                * Poisson::new(obs_rate_non_group * n_non_group as f64).unwrap().pmf(sum_metric_non_group as u64);
            let test_statistic:f64 = -2.0*(maximum_likelihood_h0 / maximum_likelihood_unconstrained).ln();
            //of course this doesn't work:
            //let p_val = ( 1-ChiSquared::new(1.0).unwrap().checked_inverse_cdf(test_statistic) );
            p_val =  1.0- gamma_li(0.5,test_statistic as f64) / gamma(0.5) ;
        }
        p_val
    }