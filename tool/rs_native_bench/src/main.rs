// Pure-Rust timing of the rshgf nodalised engine (no Python in the loop),
// on the same inputs/specs as the hgf.dart and pyhgf benchmarks.
use rshgf::model::network::Network;
use std::collections::HashMap;
use std::time::Instant;

fn p(kv: &[(&str, f64)]) -> Option<HashMap<String, f64>> {
    Some(kv.iter().map(|(k, v)| (k.to_string(), *v)).collect())
}

fn continuous(levels: usize, omega1: f64) -> Network {
    let mut n = Network::new("unbounded");
    n.add_nodes("continuous-state", 1, None, None, None, None, None, p(&[("precision", 1e4)]));
    n.add_nodes("continuous-state", 1, None, Some(vec![0].into()), None, None, None,
        p(&[("mean", 1.04), ("tonic_volatility", omega1)]));
    n.add_nodes("continuous-state", 1, None, None, None, Some(vec![1].into()), None,
        p(&[("tonic_volatility", -2.0)]));
    if levels >= 3 {
        n.add_nodes("continuous-state", 1, None, None, None, Some(vec![2].into()), None,
            p(&[("tonic_volatility", -2.0)]));
    }
    n.set_update_sequence();
    n
}

fn binary() -> Network {
    let mut n = Network::new("unbounded");
    n.add_nodes("binary-state", 1, None, None, None, None, None, None);
    n.add_nodes("continuous-state", 1, None, Some(vec![0].into()), None, None, None,
        p(&[("tonic_volatility", -3.0)]));
    n.add_nodes("continuous-state", 1, None, None, None, Some(vec![1].into()), None,
        p(&[("tonic_volatility", -2.0), ("mean", 1.0)]));
    n.set_update_sequence();
    n
}

fn read(path: &str) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path).unwrap().lines()
        .map(|l| vec![l.trim().parse::<f64>().unwrap()]).collect()
}

fn median_ms<F: FnMut()>(repeat: usize, mut f: F) -> f64 {
    for _ in 0..3 { f(); }
    let mut t: Vec<f64> = (0..repeat).map(|_| {
        let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1e3
    }).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

fn main() {
    let dir = std::env::args().nth(1).expect("data dir");
    let repeat: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(7);
    let cont = read(&format!("{dir}/continuous.txt"));
    let bin = read(&format!("{dir}/binary.txt"));
    let usd = read(&format!("{dir}/usdchf.txt"));
    let omegas: Vec<f64> = read(&format!("{dir}/omegas.txt")).into_iter().map(|v| v[0]).collect();
    let mut out = Vec::new();
    for (name, data, mk) in [
        ("continuous_3level_usdchf_614", &usd, 0),
        ("continuous_3level_10k", &cont, 0),
        ("binary_3level_10k", &bin, 1),
    ] {
        let build = || if mk == 0 { continuous(3, -8.0) } else { binary() };
        let traj = median_ms(repeat, || { let mut n = build(); n.input_data(data.clone(), None, None, true); });
        let none = median_ms(repeat, || { let mut n = build(); n.input_data(data.clone(), None, None, false); });
        out.push(format!("{{\"name\":\"{name}/rshgf_native\",\"unit\":\"ms\",\"value\":{traj}}}"));
        out.push(format!("{{\"name\":\"{name}/rshgf_native_no_traj\",\"unit\":\"ms\",\"value\":{none}}}"));
    }
    let short: Vec<Vec<f64>> = cont[..1000].to_vec();
    let batch = median_ms(repeat.max(3) / 2, || {
        for &w in &omegas { let mut n = continuous(2, w); n.input_data(short.clone(), None, None, false); }
    });
    out.push(format!("{{\"name\":\"batch_1000x1000_continuous_2level/rshgf_native_loop\",\"unit\":\"ms\",\"value\":{batch}}}"));
    println!("[{}]", out.join(","));
}
