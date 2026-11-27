#include "cache.hpp"
#include "cli.hpp"
namespace fsb {
namespace {
FitOptions fit_options(const Options &opt, Timing &tm) {
  FitOptions o;
  o.memory_mb = number(opt.get("memory-mb", "8192"));
  o.solver = opt.get("solver", "pcg");
  o.varcomp = opt.get("varcomp", "ai-reml");
  o.trace = opt.get("trace", "exact");
  require(o.solver == "pcg" || o.solver == "direct", "unknown solver");
  require(o.varcomp == "ai-reml" || o.varcomp == "em-reml" ||
              o.varcomp == "profile-reml" || o.varcomp == "fixed",
          "unknown varcomp");
  require(o.trace == "exact" || o.trace == "hutchinson",
          "trace backend unavailable: " + o.trace);
  require(!opt.has("trace") || o.varcomp == "ai-reml" || o.varcomp == "em-reml",
          "--trace applies to AI/EM");
  require(
      (o.varcomp == "fixed") == opt.has("lambda"),
      "--varcomp fixed requires --lambda; other estimators estimate lambda");
  if (opt.has("lambda")) {
    o.lambda = number(opt.get("lambda"));
    require(o.lambda >= 0, "lambda must be nonnegative");
  }
  o.rtol = number(opt.get("pcg-rtol", "1e-11"));
  o.atol = number(opt.get("pcg-atol", "1e-13"));
  require(o.rtol > 0 && o.atol >= 0, "invalid PCG tolerances");
  o.max_pcg = integer(opt.get("pcg-maxiter", "2000"), 1);
  o.max_reml = integer(
      opt.get("reml-maxiter", o.varcomp == "em-reml" ? "20000" : "200"), 1);
  o.score_tol = number(opt.get("reml-tol", "1e-8"));
  require(o.score_tol > 0, "invalid score tolerance");
  o.preconditioner = opt.get("preconditioner", "jacobi");
  require(o.preconditioner == "jacobi" || o.preconditioner == "identity" ||
              o.preconditioner == "nystrom",
          "unknown preconditioner");
  o.rank = integer(opt.get("pcg-rank", "16"), 1);
  require(!opt.has("pcg-rank") || o.preconditioner == "nystrom",
          "--pcg-rank requires nystrom");
  if (o.solver == "direct")
    for (auto key : {"pcg-rtol", "pcg-atol", "pcg-maxiter", "preconditioner"})
      require(!opt.has(key), string("--") + key + " requires PCG");
  if (o.varcomp == "fixed" || o.varcomp == "profile-reml")
    for (auto key : {"reml-maxiter", "reml-tol", "init-vu", "init-ve"})
      require(!opt.has(key), string("--") + key + " requires AI/EM");
  require(opt.has("init-vu") == opt.has("init-ve"),
          "supply both initial variances");
  if (opt.has("init-vu")) {
    o.initial_g = number(opt.get("init-vu"));
    o.initial_e = number(opt.get("init-ve"));
    require(o.initial_g > 0 && o.initial_e > 0,
            "positive initial variances required");
  }
  if (o.trace == "hutchinson") {
    require(o.solver == "pcg", "Hutchinson requires PCG");
    string backend = opt.get("probe-solver", "pcg");
    require(backend == "pcg" || backend == "lanczos", "unknown probe solver");
    o.probe_lanczos = backend == "lanczos";
    o.trace_rank = integer(opt.get("trace-rank", "0"), 0);
    auto orth = opt.get("slq-reorth", "full");
    require(orth == "full" || orth == "local",
            "invalid SLQ reorthogonalization");
    o.full_reorth = orth == "full";
    o.probes = integer(opt.get("probes", "16"), 2);
    o.max_probes = integer(opt.get("max-probes", "256"), o.probes);
    o.seed = integer(opt.get("probe-seed", "42"), 0);
    o.slq_depth = integer(opt.get("slq-depth", "12"), 2);
    o.batch = integer(opt.get("probe-batch", "8"), 1);
    o.slq_batch = integer(opt.get("slq-batch", "1"), 1);
    require(
        o.slq_batch <= o.batch && (!o.full_reorth || o.slq_batch == 1),
        "--slq-batch must fit probe batch; panels require --slq-reorth local");
    o.mc_tol = number(opt.get("mc-tol", ".025"));
    o.slq_tol = number(opt.get("slq-tol", "1e-5"));
    require(o.mc_tol > 0 && o.slq_tol > 0, "invalid stochastic tolerances");
  } else
    for (auto key : {"slq-batch", "trace-rank", "slq-reorth", "probe-solver",
                     "probes", "max-probes", "probe-seed", "slq-depth",
                     "probe-batch", "mc-tol", "slq-tol"})
      require(!opt.has(key), string("--") + key + " requires hutchinson");
  if (opt.has("max-backtracks"))
    require(o.varcomp == "ai-reml" || o.trace == "hutchinson",
            "--max-backtracks requires AI or stochastic updating");
  o.max_backtracks = integer(
      opt.get("max-backtracks", o.trace == "hutchinson" ? "39" : "54"), 0);
  o.log_every = integer(opt.get("log-every", "10"), 1);
  o.record_every = o.log_every;
  if (tm.log) {
    o.trace_rhs = tm.log->trace_enabled();
    if (tm.log->info_enabled())
      tm.log->info("solver=" + o.solver + " varcomp=" + o.varcomp +
                   " trace=" + o.trace);
    if (tm.log->debug_enabled())
      o.event = [log = tm.log](const string &s) {
        if (log->trace_enabled())
          log->trace(s);
        else
          log->debug(s);
      };
  }
  return o;
}
size_t feature_panel(const FitOptions &o, size_t n) {
  size_t width = 1;
  if (o.trace == "hutchinson")
    width = std::max(o.batch, size_t(o.trace_rank));
  else if (o.varcomp != "fixed" || o.solver == "direct")
    width = 32;
  if (o.preconditioner == "nystrom")
    width = std::max(width, size_t(o.rank));
  return std::min(width, n);
}
void write_audits(const fs::path &dir, const std::vector<FitAudit> &audits,
                  const FitOptions &o, const string &prefix) {
  write_file(dir / (prefix + "pcg_summary.tsv"), [&](std::ostream &f) {
    f << "context\tsolver\tvarcomp\tn_train\trhs\tmatvec\tmax_iterations\tmax_"
         "true_residual_over_tolerance\tprojected_residual_over_"
         "tolerance\torthogonality_over_tolerance\tprobes\tslq_matvec\ttheta_"
         "evaluations\ttrace_se_g\ttrace_se_e\ttrace_identity_error\tlogdet_"
         "se\tpaired_lik_se\tslq_depth_delta_per_n\tlogratio_se\tprec_"
         "seconds\toperator_s\testimation_s\tgls_s\tpredict_s\tvector_"
         "workspace_mib\tinitial_Vu\tinitial_"
         "Ve\tpreconditioner\tprec_rank\tzero_rhs\tresidual_restarts\tcentered_"
         "solves\trtol\tatol\tprobe_seed\tprobe_solver\ttrace_"
         "stride\tupdates\tcross_rhs\tstatus\trequested_Vu\trequested_Ve\n";
    for (auto &a : audits)
      f << a.context << '\t' << o.solver << '\t' << o.varcomp << '\t' << a.n
        << '\t' << a.rhs << '\t' << a.matvec << '\t' << a.max_iterations << '\t'
        << a.residual_ratio << '\t' << a.projected_residual << '\t'
        << a.orthogonality << '\t' << a.probes << '\t' << a.slq_matvec << '\t'
        << a.trials << '\t' << std::scalbn(a.trace_se_g, -2 * a.scale_exponent)
        << '\t' << std::scalbn(a.trace_se_e, -2 * a.scale_exponent) << '\t'
        << a.trace_identity_error << '\t' << a.logdet_se << '\t' << a.paired_se
        << '\t' << a.depth_error << '\t' << a.logratio_se << '\t'
        << a.prec_seconds << '\t' << a.operator_seconds << '\t'
        << a.estimation_seconds << '\t' << a.gls_seconds << '\t'
        << a.predict_seconds << '\t' << a.workspace_mib << '\t' << a.initial_vu
        << '\t' << a.initial_ve << '\t' << o.preconditioner << '\t'
        << a.prec_rank << '\t' << a.zero_rhs << '\t' << a.residual_restarts
        << '\t' << a.centered_solves << '\t' << o.rtol << '\t' << o.atol << '\t'
        << o.seed << '\t' << (o.probe_lanczos ? "lanczos+pcg-check" : "pcg")
        << '\t' << o.record_every << '\t' << a.updates << '\t' << a.cross_rhs
        << '\t' << a.status << '\t' << o.initial_g << '\t' << o.initial_e
        << '\n';
  });
  write_file(dir / (prefix + "reml_trace.tsv"), [&](std::ostream &f) {
    f << "context\testimator\ttrace\titeration\tVu\tVe\tcontrast_loglik\tscore_"
         "g\tscore_e\tstep\tbacktracks\tAI_rcond\tprobes\tpaired_se\tdepth_"
         "error_per_n\tAI_rank\tstatus\tpaired_face_gain\tface_matvec\n";
    for (auto &a : audits)
      for (auto &r : a.steps)
        f << a.context << '\t' << o.varcomp << '\t' << o.trace << '\t'
          << r.iteration << '\t' << std::scalbn(r.g, 2 * a.scale_exponent)
          << '\t' << std::scalbn(r.e, 2 * a.scale_exponent) << '\t'
          << (r.lik -
              double(a.n - a.fixed_rank) * a.scale_exponent * std::log(2.))
          << '\t' << std::scalbn(r.sg, -2 * a.scale_exponent) << '\t'
          << std::scalbn(r.se, -2 * a.scale_exponent) << '\t' << r.step << '\t'
          << r.backtracks << '\t' << r.condition << '\t' << r.probes << '\t'
          << r.paired_se << '\t' << r.depth_error << '\t'
          << (r.information_rank < 0 ? "NA"
                                     : std::to_string(r.information_rank))
          << '\t' << r.status << '\t' << r.face_gain << '\t' << r.face_matvec
          << '\n';
  });
}
const char *policy_name(PdPolicy policy) {
  return policy == PdPolicy::strict ? "strict" : "nearPD";
}
PdPolicy read_policy(const Options &opt) {
  auto value = opt.get("pd-policy", "strict");
  if (value == "strict")
    return PdPolicy::strict;
  if (value == "nearPD" || value == "nearpd")
    return PdPolicy::near_pd;
  throw std::runtime_error("unknown --pd-policy: " + value +
                           " (expected strict or nearPD)");
}
void write_pd(const fs::path &dir, const Fusion &fusion, PdPolicy policy) {
  write_file(dir / "pd.tsv", [&](std::ostream &f) {
    f << "block\tpolicy\taction\tjitter\titerations\tconverged\tmin_eigen_raw"
         "\tmin_eigen_used\tnormF\trelative_normF\tmax_abs_change\trel_"
         "tol\tscope\n";
    auto optional_value = [&](double x) {
      if (std::isfinite(x))
        f << x;
      else
        f << "NA";
    };
    for (const auto &d : fusion.pd_blocks) {
      f << d.block << '\t' << policy_name(policy) << '\t'
        << (d.projected ? "nearPD" : "jitter") << "\t1e-6\t" << d.iterations
        << '\t' << (d.projected ? "1" : "NA") << '\t';
      optional_value(d.min_raw);
      f << '\t';
      optional_value(d.min_used);
      f << '\t' << d.norm_f << '\t' << d.relative_norm_f << '\t'
        << d.max_abs_change << '\t';
      optional_value(d.rel_tol);
      f << "\tfull\n";
    }
  });
}
Ids names(const Options &opt, string key) {
  Ids cols;
  if (opt.has(key)) {
    std::istringstream ss(opt.get(key));
    string s;
    while (std::getline(ss, s, ','))
      cols.push_back(s);
    require(!cols.empty(), "empty column list");
    unique_ids(cols);
  }
  return cols;
}
void save_ids(const fs::path &p, const Ids &ids) {
  write_file(p, [&](std::ostream &f) {
    f << "id\n";
    for (auto &id : ids)
      f << field(id) << '\n';
  });
}
Kernel raw_features(const fs::path &p, bool snp, bool fpkm, double cap,
                    Features &qc, Timing &tm) {
  auto features = read_features(p, cap);
  tm.mark(snp ? "read_snp" : "read_omics");
  qc = snp ? genomic(std::move(features.values))
           : omics(std::move(features.values), fpkm);
  return {std::move(features.ids), std::move(qc.k)};
}
void build(const Options &opt, const fs::path &dir, double cap, Timing &tm) {
  auto p = read_table(opt.get("ped"));
  for (auto &s : p.header)
    for (char &c : s)
      c = std::tolower(static_cast<unsigned char>(c));
  unique_ids(p.header);
  auto a = pedigree(p.column("id"), p.column("sire"), p.column("dam"), cap);
  tm.mark("pedigree");
  Features gq, oq;
  auto type = opt.get("omics-type", "generic");
  require(type == "generic" || type == "fpkm", "unknown omics type");
  auto g = raw_features(opt.get("snp"), true, false, cap, gq, tm);
  tm.mark("genomic");
  tm.times.emplace_back("G_preprocess",
                        tm.times.back().second - gq.gram_seconds);
  tm.times.emplace_back("G_gram", gq.gram_seconds);
  auto o = raw_features(opt.get("omics"), false, type == "fpkm", cap, oq, tm);
  tm.mark("omics");
  tm.times.emplace_back("O_preprocess",
                        tm.times.back().second - oq.gram_seconds);
  tm.times.emplace_back("O_gram", oq.gram_seconds);
  write_file(dir / "memory.tsv", [&](std::ostream &f) {
    f << "estimated_dense_mib\tlimit_mib\n"
      << std::max(
             {budget(a.ids.size(), 0, cap),
              budget(g.ids.size(), gq.kept + gq.filtered, cap),
              budget(o.ids.size(), oq.kept + oq.filtered + oq.constant, cap)})
      << '\t' << cap << '\n';
  });
  write_file(dir / "qc.tsv", [&](std::ostream &q) {
    q << "layer\tsamples\tretained_features\tfiltered_features\tconstant_"
         "removed\timputed_cells\n";
    q << "A\t" << a.ids.size() << "\tNA\tNA\tNA\tNA\n";
    for (auto item :
         {std::make_tuple("G", &g, &gq), std::make_tuple("O", &o, &oq)}) {
      auto [name, k, v] = item;
      q << name << '\t' << k->ids.size() << '\t' << v->kept << '\t'
        << v->filtered << '\t' << v->constant << '\t' << v->imputed << '\n';
    }
  });
  for (auto item : {std::make_pair("A", &a), std::make_pair("G", &g),
                    std::make_pair("O", &o)}) {
    auto [name, k] = item;
    write_matrix(dir / (string(name) + ".tsv"), k->ids, k->values);
    save_ids(dir / (string(name) + ".ids.tsv"), k->ids);
  }
  Layers layers{std::move(a), std::move(g), std::move(o)};
  string mode;
  try {
    mode = layers.mode();
  } catch (const std::runtime_error &) {
    mode = "unsupported";
  }
  write_file(dir / "coverage.tsv", [&](std::ostream &f) {
    f << "mode\tcounts\n" << mode << '\t' << layers.counts() << '\n';
  });
  tm.mark("write");
}
struct Phenotype {
  Ids ids, effects, levels;
  vec y, truth;
  mat x;
  uvec ref;
};
Phenotype phenotype(const Options &opt, const Table &t, double cap) {
  Phenotype p;
  p.ids = t.column(opt.get("id", "ID"));
  unique_ids(p.ids);
  size_t n = p.ids.size(), trait = t.col(opt.get("trait")),
         scorecol = t.col(opt.get("score-col", opt.get("trait")));
  p.y.set_size(n);
  p.truth.set_size(n);
  for (size_t i = 0; i < n; ++i) {
    auto &r = t.rows[i];
    p.y[i] = missing(r[trait]) ? NAN : number(r[trait]);
    p.truth[i] = std::isfinite(p.y[i]) ? number(r[scorecol]) : NAN;
  }
  p.ref = arma::find_finite(p.y);
  require(!p.ref.empty(), "no reference phenotypes");
  p.effects = {"(Intercept)"};
  auto covariates = names(opt, "covar");
  std::vector<std::pair<size_t, string>> indicators;
  for (auto &name : covariates) {
    t.col(name);
    p.effects.push_back(name);
  }
  for (auto &name : names(opt, "factor")) {
    auto values = t.column(name);
    std::set<string> levels(values.begin(), values.end());
    require(!levels.empty() && !missing(*levels.begin()) && !levels.count("NA"),
            "missing factor: " + name);
    p.levels.push_back(name + "=" + *levels.begin());
    size_t column = t.col(name);
    for (auto it = std::next(levels.begin()); it != levels.end(); ++it) {
      indicators.emplace_back(column, *it);
      p.effects.push_back(name + "=" + *it);
    }
  }
  require(p.ref.n_elem > p.effects.size(),
          "training design rank deficient or no residual degrees of freedom");
  require(double(n) * p.effects.size() * 16. / 1048576. < cap,
          "fixed design exceeds --memory-mb");
  p.x.ones(n, p.effects.size());
  size_t column = 1;
  for (auto &name : covariates) {
    auto source = t.col(name);
    for (size_t i = 0; i < n; ++i)
      p.x(i, column) = number(t.rows[i][source]);
    ++column;
  }
  for (auto &[source, level] : indicators) {
    for (size_t i = 0; i < n; ++i)
      p.x(i, column) = t.rows[i][source] == level;
    ++column;
  }
  unique_ids(p.effects);
  return p;
}
Labels labels_from_file(const fs::path &path, const Ids &ids, bool &temporal) {
  auto t = read_table(path);
  size_t id = t.col("id"), r = t.col("repeat"), f = t.col("fold");
  std::map<string, size_t> idx;
  for (size_t i = 0; i < ids.size(); ++i)
    idx[ids[i]] = i;
  Labels labels;
  for (auto &row : t.rows) {
    int rep = integer(row[r], 1), fold = integer(row[f], 0);
    auto &v = labels[rep];
    if (v.empty())
      v.resize(ids.size(), -1);
    require(idx.count(row[id]), "fold ID outside reference: " + row[id]);
    auto i = idx.at(row[id]);
    require(v[i] == -1, "duplicate fold entry: " + row[id]);
    v[i] = fold;
    if (!fold)
      temporal = true;
  }
  for (auto &[r, v] : labels)
    require(std::find(v.begin(), v.end(), -1) == v.end(),
            "missing reference in folds file");
  return labels;
}
struct Tuning {
  Search search;
  Ids fold_columns;
  string metric, coverage, domain, score_policy;
  ScoreRule rule;
  size_t scorable, budget;
};
Tuning tune(const Options &opt, const Table &t, const Phenotype &p,
            const Layers &layers, const fs::path &dir, PdPolicy policy,
            double cap, Timing &tm, const FitOptions &opts, Cache *cache) {
  Ids refs;
  for (auto i : p.ref)
    refs.push_back(p.ids[i]);
  double worker_cap = cap;
  Layers l;
  if (cache)
    l = {{refs, {}},
         {intersect(refs, layers.g.ids), {}},
         {intersect(refs, layers.o.ids), {}}};
  else
    l = layers.subset(refs);
  const string mode = layers.mode();
  const bool identifiable =
      !l.o.ids.empty() &&
      (mode == "A" || (l.o.ids.size() < l.g.ids.size() &&
                       (mode == "B" || l.g.ids.size() < l.a.ids.size())));
  if (!identifiable)
    throw TuningFailure("reference coverage degenerates: " + l.counts() +
                        "; supply fixed --alpha and --beta");
  string metric = opt.get("metric", "cor");
  bool time = false;
  Labels labels;
  if (opt.has("folds-file"))
    labels = labels_from_file(opt.get("folds-file"), refs, time);
  else if (opt.has("year") || opt.has("ngen")) {
    time = true;
    string key = opt.has("year") ? "year" : "ngen";
    auto col = t.column(key);
    double cutoff = number(opt.get(key));
    for (auto i : p.ref)
      labels[1].push_back(number(col[i]) > cutoff);
  } else
    labels = random_labels(refs, integer(opt.get("folds", "5"), 2),
                           integer(opt.get("repeats", "2"), 1),
                           integer(opt.get("seed", "42"), 0));
  write_file(dir / "folds.tsv", [&](std::ostream &table) {
    table << "repeat\tfold\tid\n";
    for (auto &[r, v] : labels)
      for (size_t i = 0; i < refs.size(); ++i)
        table << r << '\t' << v[i] << '\t' << field(refs[i]) << '\n';
  });
  mat x = p.x.rows(p.ref);
  vec y = p.y.elem(p.ref), truth = p.truth.elem(p.ref);
  auto folds = split(labels, time, truth, metric);
  write_file(dir / "fold_counts.tsv", [&](std::ostream &summary) {
    summary << "repeat\tfold\ttraining\tvalidation\treason\n";
    for (auto &f : folds)
      summary << f.repeat << '\t' << f.fold << '\t' << f.train.n_elem << '\t'
              << f.valid.n_elem << '\t' << f.reason << '\n';
  });
  if (opt.get("parallel", "blas") == "weights") {
    // Dense shared layers/conditionals, design, output reserve; each private
    // task includes full reference K, nearPD/LAPACK/REML scratch and
    // BLAS/thread space.
    double n = refs.size();
    double shared = (8. * (layers.a.values.n_elem + layers.g.values.n_elem +
                           layers.o.values.n_elem) +
                     64. * n * n + 24. * p.x.n_elem) /
                        1048576. +
                    1;
    double per = (192. * n * n + 24. * n * p.x.n_cols) / 1048576. + 9;
    size_t ntrain = 0;
    for (const auto &fold : folds)
      ntrain = std::max(ntrain, size_t(fold.train.n_elem));
    size_t panel = feature_panel(opts, ntrain);
    double base = 0;
    if (cache) {
      shared = (256. * layers.a.ids.size() + 32. * cache->g.columns +
                24. * p.x.n_elem) /
                   1048576. +
               1;
      size_t m = std::max(cache->g.columns, cache->o.columns);
      base = solver_workspace(ntrain, n, p.x.n_cols, opts) +
             320. * layers.a.ids.size() + 128. * n + 64. * m;
      auto min =
          feature_workspace(cache->g.columns, cache->o.columns, ntrain, n,
                            panel, cache->streaming, cache->resident, 0);
      per = (base + min.minimum_bytes) / 1048576.;
    } else if (opts.trace == "hutchinson" || opts.preconditioner == "nystrom") {
      per = std::max(
          per, (64. * n * n + solver_workspace(ntrain, n, p.x.n_cols, opts)) /
                       1048576. +
                   9);
    }
    require(cap >= shared + per,
            "--memory-mb insufficient for one candidate workspace");
    int requested = std::min(15, integer(opt.get("threads", "1"), 1));
    tm.weight_workers = static_cast<size_t>(
        std::min(double(requested), std::floor((cap - shared) / per)));
    worker_cap = (cap - shared) / tm.weight_workers;
    size_t block = 0;
    if (cache) {
      double condition = 0;
      if (mode == "B")
        condition = 96. * n * l.o.ids.size() +
                    128. * l.o.ids.size() * l.o.ids.size() +
                    256. * layers.a.ids.size();
      if (mode == "C")
        condition = 192. * n * n + 320. * layers.a.ids.size() +
                    128. * l.g.ids.size() * l.g.ids.size();
      if (!opt.has("strict-simplex"))
        condition =
            std::max(condition, 192. * n * n + 320. * layers.a.ids.size());
      while (tm.weight_workers > 1 &&
             worker_cap * 1048576. < condition + base) {
        --tm.weight_workers;
        worker_cap = (cap - shared) / tm.weight_workers;
      }
      auto plan = feature_workspace(cache->g.columns, cache->o.columns, ntrain,
                                    n, panel, cache->streaming, cache->resident,
                                    worker_cap * 1048576. - base);
      block = plan.block;
      per = (base + plan.planned_bytes) / 1048576.;
    }
    write_file(dir / "worker_memory.tsv", [&](std::ostream &f) {
      f << "shared_mib\tper_worker_mib\tworkers\testimated_mib\tmax_train\trhs_"
           "panel\tfeature_block_rows\n"
        << shared << '\t' << per << '\t' << tm.weight_workers << '\t'
        << shared + tm.weight_workers * per << '\t' << ntrain << '\t' << panel
        << '\t' << block << '\n';
    });
  }
  std::optional<FusionPlan> reference;
  if (!cache)
    reference.emplace(l);
  FitOptions fold_options = opts;
  fold_options.ids = &refs;
  // At most 15 grid + 40 refinement points. Bound retained history separately
  // from the numerical worker buffers; full debug events remain streamable.
  double trace_budget = std::max(1., cap * 1048576. * .04);
  fold_options.record_every = std::max(
      double(opts.log_every), std::ceil(55. * folds.size() * opts.max_reml *
                                        sizeof(Iteration) / trace_budget));
  fold_options.memory_mb = worker_cap;
  string score_policy = opt.get("score-policy", "relative");
  auto rule = scoring_rule(metric, truth, y, score_policy);
  const size_t scorable =
      std::count_if(folds.begin(), folds.end(),
                    [](const Fold &f) { return f.reason.empty(); });
  size_t budget = integer(opt.get("retry-probes", "0"), 0), remaining = budget;
  auto evaluate = [&](double a, double b, const FitOptions &fold_options) {
    if (cache) {
      Evaluation ev;
      ev.scores.resize(folds.size(), NAN);
      try {
        auto prepared = cache->prepare(refs, a, b, policy, worker_cap);
        record_pd(ev, prepared.diagnostics);
        assess_folds(ev, folds, truth, metric, a, b,
                     [&](const Fold &f, FitAudit &audit) {
                       auto op = prepared.restrict(
                           refs, f.train, worker_cap,
                           solver_workspace(f.train.n_elem, refs.size(),
                                            p.x.n_cols, fold_options) /
                               1048576.,
                           feature_panel(fold_options, f.train.n_elem));
                       return fit_operator(op, x.rows(f.train), x,
                                           y.elem(f.train), fold_options,
                                           audit);
                     });
      } catch (const CertificationUnavailable &ex) {
        ev.reason = ex.what();
        ev.outcome = "uncertified";
      } catch (const std::runtime_error &ex) {
        ev.reason = ex.what();
        ev.outcome = "invalid";
      }
      return ev;
    }
    return evaluate_weight(*reference, a, b, x, y, truth, folds, metric, policy,
                           &fold_options);
  };
  auto retry = [&](double a, double b, Evaluation &ev) {
    if (opts.trace != "hutchinson")
      return;
    size_t ceiling = opts.max_probes;
    string history;
    size_t start = 0;
    for (;;) {
      ev.probe_limit = ceiling;
      if (!history.empty())
        history += " | ";
      history += std::to_string(ceiling) + ":" +
                 (ev.reason.empty() ? "ok" : ev.reason);
      size_t rhs = 0, mv = 0, probes = 0;
      for (size_t j = start; j < ev.fits.size(); ++j) {
        rhs += ev.fits[j].rhs;
        mv += ev.fits[j].matvec;
        probes += ev.fits[j].probes;
      }
      start = ev.fits.size();
      history += " [rhs=" + std::to_string(rhs) +
                 ",matvec=" + std::to_string(mv) +
                 ",probes=" + std::to_string(probes) + "]";
      ev.attempts = history;
      if (ev.reason.find("probe budget exhausted") == string::npos ||
          !scorable || ceiling > remaining / scorable / 2)
        break;
      ceiling *= 2;
      size_t cost = scorable * ceiling;
      remaining -= cost;
      auto next_options = fold_options;
      next_options.max_probes = ceiling;
      auto next = evaluate(a, b, next_options);
      next.retry_spent = ev.retry_spent + cost;
      next.fold_attempts += ev.fold_attempts;
      next.successful_fold_fits += ev.successful_fold_fits;
      next.fold_seconds += ev.fold_seconds;
      next.timing.add(ev.timing);
      ev.fits.insert(ev.fits.end(), std::make_move_iterator(next.fits.begin()),
                     std::make_move_iterator(next.fits.end()));
      next.fits = std::move(ev.fits);
      ev = std::move(next);
    }
  };
  auto result = search(
      opt.has("strict-simplex"), rule,
      [&](double a, double b) { return evaluate(a, b, fold_options); },
      tm.weight_workers, retry);

  std::vector<FitAudit> audits;
  for (auto &t : result.trials)
    for (auto &a : t.result.fits)
      audits.push_back(std::move(a));
  write_audits(dir, audits, fold_options, "cv_");
  PdTimes pd;
  double folds_s = 0;
  for (const auto &t : result.trials) {
    folds_s += t.result.fold_seconds;
    pd.add(t.result.timing);
  }
  tm.times.emplace_back("cv_fold_fit_sum", folds_s);
  tm.times.emplace_back("cv_copy_subset_sum", pd.copy);
  tm.times.emplace_back("cv_eigen_sum", pd.eigen);
  tm.times.emplace_back("cv_reconstruction_sum", pd.reconstruction);
  tm.times.emplace_back("cv_matrix_work_sum", pd.matrix_work);
  Ids columns;
  for (const auto &f : folds)
    columns.push_back("r" + std::to_string(f.repeat) + "f" +
                      std::to_string(f.fold));
  return {std::move(result),
          std::move(columns),
          metric,
          l.counts(),
          opt.has("strict-simplex") ? "simplex" : "historical",
          score_policy,
          rule,
          scorable,
          budget};
}
// Validate complete-cohort covariance in CV score order. Candidate phenotypes
// never enter this check; the common case constructs just one full kernel.
Fusion select_fusion(const Tuning &t, const Layers &layers, const fs::path &dir,
                     double &alpha, double &beta, PdPolicy policy, Cache *cache,
                     std::optional<PopulationKernel> &selected_operator,
                     double cap, FitOptions &options) {
  const auto &result = t.search;
  const auto &trials = result.trials;
  const size_t count = trials.size();
  Ids rejected(count);
  bool partial = std::any_of(trials.begin(), trials.end(), [](const Trial &v) {
    return v.result.outcome == "uncertified";
  });
  std::vector<bool> uncertified(count, false);
  size_t selected = count, failures = 0;
  Fusion kernel;
  std::optional<FusionPlan> full;
  if (!cache)
    full.emplace(layers);
  for (;;) {
    size_t next = count;
    for (size_t i = 0; i < count; ++i) {
      const auto &e = trials[i].result;
      if (!e.reason.empty() || !std::isfinite(e.value) || !rejected[i].empty())
        continue;
      if (next == count || t.rule.better(e.value, trials[next].result.value))
        next = i;
    }
    if (next == count)
      break;
    const auto &v = trials[next];
    try {
      if (cache) {
        selected_operator =
            cache->prepare(layers.a.ids, v.alpha, v.beta, policy, cap);
        kernel.correction = pd_summary(selected_operator->diagnostics);
        kernel.pd_blocks = selected_operator->diagnostics;
      } else
        kernel = full->combine(v.alpha, v.beta, nullptr, policy);
    } catch (const CertificationUnavailable &ex) {
      partial = true;
      uncertified[next] = true;
      rejected[next] = ex.what();
      ++failures;
      continue;
    } catch (const std::runtime_error &ex) {
      rejected[next] = string("full coverage: ") + ex.what();
      ++failures;
      continue;
    }
    selected = next;
    if (v.result.probe_limit)
      options.max_probes = v.result.probe_limit;
    alpha = v.alpha;
    beta = v.beta;
    break;
  }
  write_file(dir / "search.tsv", [&](std::ostream &trace) {
    trace << "stage\talpha\tbeta\tmean\tvalid\treason\tcorrection";
    for (const auto &column : t.fold_columns)
      trace << '\t' << column;
    trace
        << "\treference_best\tfull_status\tfull_reason\tpd_policy\tnearpd_"
           "blocks\tmax_relative_normF\tfull_nearpd_blocks\tfull_max_relative_"
           "normF\trequested_domain\tbackend\tcandidate_status\tsearch_"
           "coverage\tprobe_attempts\tretry_reserved_probes\tfolds_scored_last_"
           "attempt\tfold_attempts\tfolds_completed\n";
    for (size_t i = 0; i < count; ++i) {
      const auto &v = trials[i];
      const auto &e = v.result;
      trace << v.stage << '\t' << v.alpha << '\t' << v.beta << '\t' << e.value
            << '\t' << e.reason.empty() << '\t' << field(e.reason) << '\t'
            << e.correction;
      for (auto score : e.scores)
        trace << '\t' << score;
      const bool ref_best = std::isfinite(result.best) &&
                            v.alpha == result.alpha && v.beta == result.beta;
      const char *status = i == selected          ? "selected"
                           : !rejected[i].empty() ? "rejected"
                                                  : "not_checked";
      trace << '\t' << ref_best << '\t' << status << '\t' << field(rejected[i])
            << '\t' << policy_name(policy) << '\t' << e.nearpd_blocks << '\t'
            << e.max_relative_norm_f;
      if (i == selected) {
        size_t projected = 0;
        double relative = 0;
        for (const auto &block : kernel.pd_blocks) {
          projected += block.projected;
          relative = std::max(relative, block.relative_norm_f);
        }
        trace << '\t' << projected << '\t' << relative;
      } else
        trace << "\tNA\tNA";
      trace << '\t' << t.domain << '\t' << (cache ? "operator" : "dense")
            << '\t' << (uncertified[i] ? "uncertified" : e.outcome) << '\t'
            << (partial ? "partial" : "complete") << '\t' << field(e.attempts)
            << '\t' << e.retry_spent << '\t'
            << std::count_if(e.scores.begin(), e.scores.end(),
                             [](double x) { return std::isfinite(x); })
            << '\t' << e.fold_attempts << '\t' << e.successful_fold_fits
            << '\n';
    }
  });
  size_t scored = 0, completed = 0, probe_failed = 0, numerical = 0,
         invalid = 0, unavailable = 0, spent = 0, attempted = 0,
         physical_completed = 0;
  for (const auto &v : trials) {
    const auto &e = v.result;
    scored += e.reason.empty() && std::isfinite(e.value);
    completed += std::count_if(e.scores.begin(), e.scores.end(),
                               [](double s) { return std::isfinite(s); });
    bool exhausted = e.reason.find("probe budget exhausted") != string::npos;
    probe_failed += exhausted;
    numerical += e.outcome == "fit_failed" && !exhausted;
    invalid += e.outcome == "invalid";
    unavailable += e.outcome == "uncertified";
    spent += e.retry_spent;
    attempted += e.fold_attempts;
    physical_completed += e.successful_fold_fits;
  }
  write_file(dir / "tuning.tsv", [&](std::ostream &f) {
    f << "metric\tselection_score\tstop\tcoverage\treference_best_alpha"
         "\treference_best_beta\treference_best_score\tselected_"
         "alpha\tselected_beta"
         "\tfull_rejections\trequested_domain\tbackend\tsearch_coverage\tscore_"
         "policy\tscore_scale\tcomparison_tie\tcomparison_stop\tpd_"
         "coverage\tcandidates_visited\tcandidates_scored\tscorable_folds_per_"
         "point\tfolds_completed\tprobe_exhausted\tnumerical_failures\tinvalid_"
         "kernel\tuncertified\tretry_budget\tretry_reserved\tfolds_scored_last_"
         "attempt\tfold_attempts\tfull_invalid\tfull_uncertified\n"
      << t.metric << '\t'
      << (selected < count ? trials[selected].result.value : NAN) << '\t'
      << result.stop << '\t' << field(t.coverage) << '\t' << result.alpha
      << '\t' << result.beta << '\t' << result.best << '\t'
      << (selected < count ? alpha : NAN) << '\t'
      << (selected < count ? beta : NAN) << '\t' << failures << '\t' << t.domain
      << '\t' << (cache ? "operator" : "dense") << '\t'
      << (partial ? "partial" : "complete") << '\t' << t.score_policy << '\t'
      << t.rule.scale << "\t1e-12\t1e-4\t" << (partial ? "partial" : "complete")
      << '\t' << count << '\t' << scored << '\t' << t.scorable << '\t'
      << physical_completed << '\t' << probe_failed << '\t' << numerical << '\t'
      << invalid << '\t' << unavailable << '\t' << t.budget << '\t' << spent
      << '\t' << completed << '\t' << attempted << '\t'
      << failures - std::count(uncertified.begin(), uncertified.end(), true)
      << '\t' << std::count(uncertified.begin(), uncertified.end(), true)
      << '\n';
  });
  if (!std::isfinite(result.best))
    throw TuningFailure(result.stop + ": " + trials.front().result.reason);
  if (selected == count)
    throw TuningFailure(
        "no reference-eligible weights produced a valid full covariance; " +
        std::to_string(failures) + " rejected (see search.tsv)");
  return kernel;
}
} // namespace
void run(const string &cmd, const Options &opt, const fs::path &dir, double cap,
         Timing &tm) {
  if (cmd == "prepare") {
    auto type = opt.get("omics-type", "generic");
    require(type == "generic" || type == "fpkm", "unknown omics type");
    auto format = opt.get("snp-format", "f64");
    require(format == "f64" || format == "2bit", "invalid --snp-format");
    prepare_cache(opt.get("ped"), opt.get("snp"), opt.get("omics"),
                  type == "fpkm", dir, cap, format == "2bit");
    tm.mark("prepare");
    return;
  }
  if (cmd == "build") {
    build(opt, dir, cap, tm);
    return;
  }
  const auto policy = read_policy(opt);
  require(opt.has("cache") ||
              (!opt.has("cache-storage") && !opt.has("cache-backend")),
          "cache options require --cache");
  double output_estimate = 0, output_available = 0;
  auto check_write_space = [&](const Ids &ids) {
    if (cmd != "fuse" && !opt.has("write-k"))
      return;
    double need = 52. * ids.size() * ids.size() + 4096;
    for (auto &id : ids)
      need += 8 * (id.size() + 4);
    auto available = fs::space(dir).available;
    require(available > need, "insufficient staging disk space for --write-k");
    output_estimate = need;
    output_available = available;
  };
  Layers l;
  std::optional<Cache> cache;
  bool use_op = false;
  if (opt.has("cache")) {
    require(!opt.has("A") && !opt.has("G") && !opt.has("O"),
            "--cache conflicts with external kernels");
    cache.emplace(opt.get("cache"), cap);
    auto storage = opt.get("cache-storage", "auto");
    require(storage == "auto" || storage == "stream" || storage == "resident",
            "invalid cache storage");
    cache->streaming = storage == "stream";
    cache->resident = storage == "resident";
    check_write_space(cache->ped.ids);
    auto mode = cache->mode();
    require(!cache->resident || mode == "A",
            "resident storage requires common coverage");
    double a = number(opt.get("alpha", "0")), b = number(opt.get("beta", "0"));
    auto backend = opt.get("cache-backend", "auto");
    require(backend == "auto" || backend == "operator" || backend == "dense",
            "invalid cache backend");
    bool compatible =
        !opt.has("write-k") && opt.get("varcomp", "ai-reml") != "profile-reml";
    double full_need = (160. * cache->ped.ids.size() * cache->ped.ids.size() +
                        64. * cache->ped.ids.size() *
                            std::max(cache->g.columns, cache->o.columns)) /
                       1048576.;
    use_op = compatible && backend != "dense" &&
             (mode == "A" || full_need > cap || backend == "operator");
    if (backend == "auto" && !opt.has("alpha") && !opt.has("strict-simplex") &&
        full_need <= cap)
      use_op = false;
    require(backend != "operator" || use_op,
            "operator backend cannot materialize --write-k, profile-reml or "
            "uncertified weights; use dense backend");
    if (use_op)
      l = {{cache->ped.ids, {}}, {cache->g.ids, {}}, {cache->o.ids, {}}};
    else {
      double need = (160. * cache->ped.ids.size() * cache->ped.ids.size() +
                     64. * cache->ped.ids.size() *
                         std::max(cache->g.columns, cache->o.columns)) /
                    1048576.;
      require(need <= cap, "mode " + mode +
                               " requires full dense covariance "
                               "certification/materialization; estimated MiB=" +
                               std::to_string(need) +
                               "; increase --memory-mb or use common-coverage "
                               "simplex PCG without --write-k");
      l = cache->materialize(cap);
    }
  } else
    l = {read_kernel(opt.get("A"), cap), read_kernel(opt.get("G"), cap),
         read_kernel(opt.get("O"), cap)};
  if (!cache)
    check_write_space(l.a.ids);
  auto mode = l.mode();
  write_file(dir / "memory.tsv", [&](std::ostream &f) {
    f << "estimated_dense_mib\tlimit_mib\n"
      << (use_op ? (256. * cache->ped.ids.size() +
                    64. * std::max(cache->g.columns, cache->o.columns)) /
                       1048576.
                 : budget(l.a.ids.size(), 0, cap))
      << '\t' << cap << '\n';
  });
  write_file(dir / "coverage.tsv", [&](std::ostream &f) {
    f << "mode\tcounts\n" << mode << '\t' << l.counts() << '\n';
  });
  tm.mark("read_kernels");
  bool fixed = opt.has("alpha");
  require(fixed == opt.has("beta"), "supply both alpha and beta");
  double a = 0, b = 0;
  if (fixed) {
    a = number(opt.get("alpha"));
    b = number(opt.get("beta"));
    if (opt.has("strict-simplex"))
      require(a >= 0 && b >= 0 && a + b <= 1, "weights outside simplex");
  }
  Fusion k;
  std::optional<PopulationKernel> selected_operator;
  Phenotype p;
  FitOptions opts;
  auto save_usage = [&] {
    if (cache)
      write_file(dir / "cache_usage.tsv", [&](auto &f) {
        f << "mode\tbackend\tfeature_bytes\tread_calls\tread_seconds\tdecode_"
             "seconds\tverified_cache_bytes\tstorage\ttargets\tactive_"
             "component_ids\tcalibration_bytes\ttraining_preload_"
             "bytes\ttraining_bytes\tcross_bytes\n"
          << mode << '\t' << (use_op ? "operator" : "dense") << '\t'
          << cache->io.bytes.load() << '\t' << cache->io.reads.load() << '\t'
          << cache->io.read_ns.load() * 1e-9 << '\t'
          << cache->io.decode_ns.load() * 1e-9 << '\t' << cache->verified_bytes
          << '\t'
          << (cache->streaming  ? "stream"
              : cache->resident ? "resident"
                                : "auto")
          << '\t' << p.ids.size() << '\t'
          << (selected_operator ? selected_operator->active_count : 0);
        for (const auto &bytes : cache->io.phase_bytes)
          f << '\t' << bytes.load();
        f << '\n';
      });
  };
  if (cmd == "fit") {
    opts = fit_options(opt, tm);
    require(!opt.has("retry-probes") || (!fixed && opts.trace == "hutchinson"),
            "--retry-probes requires stochastic automatic tuning");
    require(!opt.has("score-policy") || !fixed,
            "--score-policy requires automatic tuning");
    int random = opt.has("folds") || opt.has("repeats") || opt.has("seed");
    require(random + opt.has("folds-file") + opt.has("year") + opt.has("ngen") +
                    fixed <=
                1,
            "fixed weights, random folds, external folds and time split are "
            "mutually exclusive");
    auto t = read_table(opt.get("pheno"));
    p = phenotype(opt, t, cap);
    opts.ids = &p.ids;
    positions(l.a.ids, p.ids);
    tm.mark("read_phenotype");
    if (!fixed) {
      try {
        const auto tuning = tune(opt, t, p, l, dir, policy, cap, tm, opts,
                                 use_op ? &*cache : nullptr);
        tm.mark("search");
        k = select_fusion(tuning, l, dir, a, b, policy,
                          use_op ? &*cache : nullptr, selected_operator, cap,
                          opts);
      } catch (const TuningFailure &) {
        save_usage();
        throw;
      }
    }
  } else
    require(fixed, "fuse requires alpha and beta");
  if (fixed && use_op) {
    selected_operator = cache->prepare(l.a.ids, a, b, policy, cap);
    k.correction = pd_summary(selected_operator->diagnostics);
    k.pd_blocks = selected_operator->diagnostics;
  } else if (fixed)
    k = fuse(l, a, b, nullptr, policy);
  tm.mark("fusion");
  PdTimes pd;
  for (const auto &b : k.pd_blocks)
    pd.add(b.timing);
  tm.times.emplace_back("nearPD_copy_subset", pd.copy);
  tm.times.emplace_back("nearPD_eigen", pd.eigen);
  tm.times.emplace_back("nearPD_reconstruction", pd.reconstruction);
  tm.times.emplace_back("nearPD_matrix_work", pd.matrix_work);
  write_file(dir / "fusion.tsv", [&](std::ostream &f) {
    f << "mode\talpha\tbeta\tcorrection\tpd_policy\tscope\n"
      << mode << '\t' << a << '\t' << b << '\t' << k.correction << '\t'
      << policy_name(policy) << "\tfull\n";
  });
  write_pd(dir, k, policy);
  if (tm.log)
    tm.log->info("scope=full pd=" + k.correction);
  if (cmd == "fuse" || opt.has("write-k")) {
    require(fs::space(dir).available > 52. * l.a.ids.size() * l.a.ids.size(),
            "insufficient disk space for --write-k");
    write_matrix(dir / "K_raw.tsv", l.a.ids, k.raw);
    write_matrix(dir / "K.tsv", l.a.ids, k.k);
    save_ids(dir / "ids.tsv", l.a.ids);
    write_file(dir / "matrix_output_budget.tsv", [&](auto &f) {
      f << "estimated_bytes\tavailable_bytes\n"
        << output_estimate << '\t' << output_available << '\n';
    });
    tm.mark("write_kernels");
  }
  if (cmd == "fit") {
    auto idx = positions(l.a.ids, p.ids);
    if (opts.trace == "hutchinson") {
      double need =
          (160. * l.a.ids.size() +
           p.ref.n_elem * (8. * (2. * opts.slq_depth + 12 * p.x.n_cols) +
                           80. * opts.batch) +
           opts.max_probes * (opts.slq_depth * 96. + 64.)) /
              1048576. +
          9;
      require(need < cap, "stochastic solver workspace exceeds --memory-mb");
    }
    FitAudit audit;
    audit.context =
        "final alpha=" + std::to_string(a) + " beta=" + std::to_string(b);
    Model m;
    try {
      if (use_op) {
        auto op = selected_operator->restrict(
            p.ids, p.ref, cap,
            solver_workspace(p.ref.n_elem, p.ids.size(), p.x.n_cols, opts) /
                1048576.,
            feature_panel(opts, p.ref.n_elem));
        m = fit_operator(op, p.x.rows(p.ref), p.x, p.y.elem(p.ref), opts,
                         audit);
      } else
        m = fit_dense(k.k, idx, p.x, p.y, p.ref, opts, audit);
    } catch (const std::runtime_error &error) {
      if (fixed && audit.steps.empty() && audit.rhs == 0)
        throw;
      save_usage();
      audit.status = error.what();
      write_audits(dir, {audit}, opts, "");
      throw TuningFailure("final fit failed: " + audit.status);
    }
    save_usage();
    tm.mark("final_reml_blup");
    if (tm.log && tm.log->info_enabled())
      tm.log->info("stage=model alpha=" + std::to_string(a) +
                   " beta=" + std::to_string(b) +
                   " n_train=" + std::to_string(p.ref.n_elem) +
                   " status=" + m.status + " RHS=" + std::to_string(audit.rhs));
    write_file(dir / "predictions.tsv", [&](std::ostream &f) {
      f << "id\trole\tu\tyhat\n";
      for (size_t i = 0; i < p.ids.size(); ++i)
        f << field(p.ids[i]) << '\t'
          << (std::isfinite(p.y[i]) ? "reference" : "candidate") << '\t'
          << m.u[i] << '\t' << m.yhat[i] << '\n';
    });
    write_audits(dir, {audit}, opts, "");
    write_file(dir / "model.tsv", [&](std::ostream &g) {
      g << "Vu\tVe\tlambda\tstatus\tprofile_"
           "objective\treference\tcandidate\tsolver\tvarcomp\ttrace\tcontrast_"
           "loglik\n"
        << m.vu << '\t' << m.ve << '\t' << m.lambda << '\t' << m.status << '\t';
      if (std::isfinite(m.objective))
        g << m.objective;
      else
        g << "NA";
      g << '\t' << p.ref.n_elem << '\t' << p.ids.size() - p.ref.n_elem << '\t'
        << opts.solver << '\t' << opts.varcomp << '\t'
        << ((opts.varcomp == "fixed" || opts.varcomp == "profile-reml")
                ? "none"
                : opts.trace)
        << '\t' << audit.final_lik << '\n';
    });
    write_file(dir / "fixed.tsv", [&](std::ostream &h) {
      h << "term\testimate\n";
      for (size_t i = 0; i < p.effects.size(); ++i)
        h << field(p.effects[i]) << '\t' << m.b[i] << '\n';
    });
    write_file(dir / "reference_levels.tsv", [&](std::ostream &levels) {
      levels << "factor_reference\n";
      for (auto &v : p.levels)
        levels << field(v) << '\n';
    });
  }
  tm.mark("write");
}
} // namespace fsb
