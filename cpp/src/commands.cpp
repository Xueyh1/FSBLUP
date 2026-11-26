#include "cli.hpp"
#include "cache.hpp"
namespace fsb {
namespace {
FitOptions fit_options(const Options& opt,Timing& tm){
 FitOptions o;o.solver=opt.get("solver","pcg");o.varcomp=opt.get("varcomp","ai-reml");o.trace=opt.get("trace","exact");
 require(o.solver=="pcg"||o.solver=="direct","unknown solver");
 require(o.varcomp=="ai-reml"||o.varcomp=="em-reml"||o.varcomp=="profile-reml"||o.varcomp=="fixed","unknown varcomp");
 require(o.trace=="exact","trace backend unavailable: "+o.trace);
 require(!opt.has("trace")||o.varcomp=="ai-reml"||o.varcomp=="em-reml","--trace applies to AI/EM");
 require((o.varcomp=="fixed")==opt.has("lambda"),"--varcomp fixed requires --lambda; other estimators estimate lambda");
 if(opt.has("lambda")){o.lambda=number(opt.get("lambda"));require(o.lambda>0,"lambda must be positive");}
 o.rtol=number(opt.get("pcg-rtol","1e-11"));o.atol=number(opt.get("pcg-atol","1e-13"));
 require(o.rtol>0 && o.atol>=0,"invalid PCG tolerances");
 o.max_pcg=integer(opt.get("pcg-maxiter","2000"),1);o.max_reml=integer(opt.get("reml-maxiter",o.varcomp=="em-reml"?"20000":"200"),1);
 o.score_tol=number(opt.get("reml-tol","1e-8"));require(o.score_tol>0,"invalid score tolerance");
 o.preconditioner=opt.get("preconditioner","jacobi");require(o.preconditioner=="jacobi"||o.preconditioner=="identity","unknown preconditioner");
 if(o.solver=="direct")for(auto key:{"pcg-rtol","pcg-atol","pcg-maxiter","preconditioner"})require(!opt.has(key),string("--")+key+" requires PCG");
 if(o.varcomp=="fixed"||o.varcomp=="profile-reml")for(auto key:{"reml-maxiter","reml-tol","init-vu","init-ve"})require(!opt.has(key),string("--")+key+" requires AI/EM");
 require(opt.has("init-vu")==opt.has("init-ve"),"supply both initial variances");
 if(opt.has("init-vu")){o.initial_g=number(opt.get("init-vu"));o.initial_e=number(opt.get("init-ve"));require(o.initial_g>0&&o.initial_e>0,"positive initial variances required");}
 o.log_every=integer(opt.get("log-every","10"),1);
 if(tm.log){tm.log->info("solver="+o.solver+" varcomp="+o.varcomp+" trace="+o.trace);if(tm.log->debug_enabled())o.event=[log=tm.log](const string& s){log->debug(s);};}
 return o;
}
void write_audits(const fs::path& dir,const std::vector<FitAudit>& audits,const FitOptions& o,const string& prefix){
 write_file(dir/(prefix+"pcg_summary.tsv"),[&](std::ostream& f){
 f<<"context\tsolver\tvarcomp\tn_train\trhs\tmatvec\tmax_iterations\tmax_true_residual_over_tolerance\tprojected_residual_over_tolerance\torthogonality_over_tolerance\tstatus\n";
 for(auto& a:audits)f<<a.context<<'\t'<<o.solver<<'\t'<<o.varcomp<<'\t'<<a.n<<'\t'<<a.rhs<<'\t'<<a.matvec<<'\t'<<a.max_iterations<<'\t'<<a.residual_ratio<<'\t'<<a.projected_residual<<'\t'<<a.orthogonality<<'\t'<<a.status<<'\n';});
 write_file(dir/(prefix+"reml_trace.tsv"),[&](std::ostream& f){
 f<<"context\testimator\ttrace\titeration\tVu\tVe\tcontrast_loglik\tscore_g\tscore_e\tstep\tbacktracks\tAI_rcond\tstatus\n";
 for(auto& a:audits)for(auto& r:a.steps)f<<a.context<<'\t'<<o.varcomp<<'\t'<<o.trace<<'\t'<<r.iteration<<'\t'<<r.g<<'\t'<<r.e<<'\t'<<r.lik<<'\t'<<r.sg<<'\t'<<r.se<<'\t'<<r.step<<'\t'<<r.backtracks<<'\t'<<r.condition<<'\t'<<r.status<<'\n';});
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
         "\tmin_eigen_used\tnormF\trelative_normF\tmax_abs_change\trel_tol\n";
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
      f << '\n';
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
Phenotype phenotype(const Options &opt, const Table &t) {
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
  p.x.ones(n, 1);
  for (auto &c : names(opt, "covar")) {
    auto v = t.column(c);
    vec z(n);
    for (size_t i = 0; i < n; ++i)
      z[i] = number(v[i]);
    p.x.insert_cols(p.x.n_cols, z);
    p.effects.push_back(c);
  }
  for (auto &c : names(opt, "factor")) {
    auto v = t.column(c);
    std::set<string> levels(v.begin(), v.end());
    require(!levels.empty() && !missing(*levels.begin()) && !levels.count("NA"),
            "missing factor: " + c);
    p.levels.push_back(c + "=" + *levels.begin());
    for (auto iter = std::next(levels.begin()); iter != levels.end(); ++iter) {
      vec z(n);
      for (size_t i = 0; i < n; ++i)
        z[i] = (v[i] == *iter);
      p.x.insert_cols(p.x.n_cols, z);
      p.effects.push_back(c + "=" + *iter);
    }
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
  string metric, coverage;
};
Tuning tune(const Options &opt, const Table &t, const Phenotype &p,
            const Layers &layers, const fs::path &dir, PdPolicy policy,
            double cap, Timing &tm,const FitOptions& opts,Cache* cache) {
  Ids refs;
  for (auto i : p.ref)
    refs.push_back(p.ids[i]);
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
    require(cap >= shared + per,
            "--memory-mb insufficient for one candidate workspace");
    int requested = std::min(15, integer(opt.get("threads", "1"), 1));
    tm.weight_workers = static_cast<size_t>(
        std::min(double(requested), std::floor((cap - shared) / per)));
    write_file(dir / "worker_memory.tsv", [&](std::ostream &f) {
      f << "shared_mib\tper_worker_mib\tworkers\testimated_mib\n"
        << shared << '\t' << per << '\t' << tm.weight_workers << '\t'
        << shared + tm.weight_workers * per << '\n';
    });
  }
  Layers l;
  if(cache)l={{refs,{}},{refs,{}},{refs,{}}};else l=layers.subset(refs);
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
  FusionPlan reference(l);
  auto result = search(
      opt.has("strict-simplex"), metric == "rmse",
      [&](double a, double b) {
        if(cache){
          Evaluation ev;ev.scores.resize(folds.size(),NAN);
          try{
            require(a>=0 && b>=0 && a+b<=1,"operator covariance certification requires simplex weights");
            size_t count=0; double sum=0;
            for(size_t j=0;j<folds.size();++j){
              const auto& f=folds[j];if(!f.reason.empty())continue;
              FitAudit& audit=ev.fits.emplace_back();audit.context="alpha="+std::to_string(a)+" beta="+std::to_string(b)+" repeat="+std::to_string(f.repeat)+" fold="+std::to_string(f.fold);
              auto op=cache->kernel(refs,f.train,a,b,cap/tm.weight_workers);
              auto m=fit_operator(op,x.rows(f.train),x,y.elem(f.train),opts,audit);
              ev.scores[j]=score(truth.elem(f.valid),m.u.elem(f.valid),metric);
              require(std::isfinite(ev.scores[j]),"nonfinite score");sum+=ev.scores[j];++count;
            }
            require(count>0,"no scorable fold");ev.value=sum/count;ev.correction="certified-PSD,jitter=1e-6";
          }catch(const std::runtime_error& ex){ev.reason=ex.what();if(!ev.fits.empty())ev.fits.back().status=ev.reason;}
          return ev;
        }
        return evaluate_weight(reference, a, b, x, y, truth, folds, metric,
                               policy,&opts);
      },
      tm.weight_workers);
  std::vector<FitAudit> audits;for(auto& t:result.trials)for(auto& a:t.result.fits)audits.push_back(std::move(a));
  write_audits(dir,audits,opts,"cv_");
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
  return {std::move(result), std::move(columns), metric, l.counts()};
}
// Validate complete-cohort covariance in CV score order. Candidate phenotypes
// never enter this check; the common case constructs just one full kernel.
Fusion select_fusion(const Tuning &t, const Layers &layers, const fs::path &dir,
                     double &alpha, double &beta, PdPolicy policy,bool certified=false) {
  const auto &result = t.search;
  const auto &trials = result.trials;
  const size_t count = trials.size();
  Ids rejected(count);
  size_t selected = count, failures = 0;
  Fusion kernel;
  FusionPlan full(layers);
  for (;;) {
    size_t next = count;
    for (size_t i = 0; i < count; ++i) {
      const auto &e = trials[i].result;
      if (!e.reason.empty() || !std::isfinite(e.value) || !rejected[i].empty())
        continue;
      if (next == count ||
          (t.metric == "rmse" ? e.value < trials[next].result.value - 1e-12
                              : e.value > trials[next].result.value + 1e-12))
        next = i;
    }
    if (next == count)
      break;
    const auto &v = trials[next];
    try {
      if(certified){require(v.alpha>=0&&v.beta>=0&&v.alpha+v.beta<=1,"full operator requires simplex");kernel.correction="certified-PSD,jitter=1e-6";kernel.pd_blocks={{"K"}};}
      else kernel = full.combine(v.alpha, v.beta, nullptr, policy);
    } catch (const std::runtime_error &ex) {
      rejected[next] = string("full coverage: ") + ex.what();
      ++failures;
      continue;
    }
    selected = next;
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
           "normF\n";
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
      trace << '\n';
    }
  });
  if (!std::isfinite(result.best))
    throw TuningFailure(result.stop + ": " + trials.front().result.reason);
  if (selected == count)
    throw TuningFailure(
        "no reference-eligible weights produced a valid full covariance; " +
        std::to_string(failures) + " rejected (see search.tsv)");
  write_file(dir / "tuning.tsv", [&](std::ostream &f) {
    f << "metric\tselection_score\tstop\tcoverage\treference_best_alpha"
         "\treference_best_beta\treference_best_score\tselected_"
         "alpha\tselected_beta"
         "\tfull_rejections\n"
      << t.metric << '\t' << trials[selected].result.value << '\t'
      << result.stop << '\t' << field(t.coverage) << '\t' << result.alpha
      << '\t' << result.beta << '\t' << result.best << '\t' << alpha << '\t'
      << beta << '\t' << failures << '\n';
  });
  return kernel;
}
} // namespace
void run(const string &cmd, const Options &opt, const fs::path &dir, double cap,
         Timing &tm) {
  if(cmd=="prepare"){
    auto type=opt.get("omics-type","generic");require(type=="generic"||type=="fpkm","unknown omics type");
    prepare_cache(opt.get("ped"),opt.get("snp"),opt.get("omics"),type=="fpkm",dir,cap);
    tm.mark("prepare");return;
  }
  if (cmd == "build") {
    build(opt, dir, cap, tm);
    return;
  }
  const auto policy = read_policy(opt);
  Layers l; std::optional<Cache> cache;
  bool use_op=false;
  if(opt.has("cache")){
    require(!opt.has("A")&&!opt.has("G")&&!opt.has("O"),"--cache conflicts with external kernels");
    cache.emplace(opt.get("cache"));auto mode=cache->mode();
    double a=number(opt.get("alpha","0")),b=number(opt.get("beta","0"));
    use_op=mode=="A"&&a>=0&&b>=0&&a+b<=1&&!opt.has("write-k") && opt.get("varcomp","ai-reml")!="profile-reml";
    if(use_op)l={{cache->ped.ids,{}},{cache->g.ids,{}},{cache->o.ids,{}}};
    else{
      double need=(160.*cache->ped.ids.size()*cache->ped.ids.size()+64.*cache->ped.ids.size()*std::max(cache->g.columns,cache->o.columns))/1048576.;
      require(need<=cap,"mode "+mode+" requires full dense covariance certification/materialization; estimated MiB="+std::to_string(need)+"; increase --memory-mb or use common-coverage simplex PCG without --write-k");
      l=cache->materialize(cap);
    }
  }else l={read_kernel(opt.get("A"), cap),read_kernel(opt.get("G"),cap),read_kernel(opt.get("O"),cap)};
  auto mode = l.mode();
  write_file(dir / "memory.tsv", [&](std::ostream &f) {
    f << "estimated_dense_mib\tlimit_mib\n"
      << (use_op?0:budget(l.a.ids.size(), 0, cap)) << '\t' << cap << '\n';
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
  Phenotype p; FitOptions opts;
  if (cmd == "fit") {
    opts=fit_options(opt,tm);
    int random = opt.has("folds") || opt.has("repeats") || opt.has("seed");
    require(random + opt.has("folds-file") + opt.has("year") + opt.has("ngen") +
                    fixed <=
                1,
            "fixed weights, random folds, external folds and time split are "
            "mutually exclusive");
    auto t = read_table(opt.get("pheno"));
    p = phenotype(opt, t);
    positions(l.a.ids, p.ids);
    tm.mark("read_phenotype");
    if (!fixed) {
      const auto tuning = tune(opt, t, p, l, dir, policy, cap, tm,opts,use_op?&*cache:nullptr);
      tm.mark("search");
      k = select_fusion(tuning, l, dir, a, b, policy,use_op);
    }
  } else
    require(fixed, "fuse requires alpha and beta");
  if(fixed&&use_op){k.correction="certified-PSD,jitter=1e-6";k.pd_blocks={{"K"}};}
  else if (fixed)
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
    f << "mode\talpha\tbeta\tcorrection\tpd_policy\n"
      << mode << '\t' << a << '\t' << b << '\t' << k.correction << '\t'
      << policy_name(policy) << '\n';
  });
  write_pd(dir, k, policy);
  if (cmd == "fuse" || opt.has("write-k")) {
    write_matrix(dir / "K_raw.tsv", l.a.ids, k.raw);
    write_matrix(dir / "K.tsv", l.a.ids, k.k);
    save_ids(dir / "ids.tsv", l.a.ids);
    tm.mark("write_kernels");
  }
  if (cmd == "fit") {
    auto idx = positions(l.a.ids, p.ids);
    FitAudit audit;audit.context="final alpha="+std::to_string(a)+" beta="+std::to_string(b);
    Model m;
    if(use_op){
      auto op=cache->kernel(p.ids,p.ref,a,b,cap);
      m=fit_operator(op,p.x.rows(p.ref),p.x,p.y.elem(p.ref),opts,audit);
    }else m=fit_model(k.k.submat(idx,idx),p.x,p.y,p.ref,opts,audit);
    if(cache)write_file(dir/"cache_usage.tsv",[&](auto& f){f<<"mode\tbackend\tfeature_bytes\n"<<mode<<'\t'<<(use_op?"operator":"dense")<<'\t'<<cache->bytes.load()<<'\n';});
    write_audits(dir,{audit},opts,"");
    tm.mark("final_reml_blup");
    write_file(dir / "predictions.tsv", [&](std::ostream &f) {
      f << "id\trole\tu\tyhat\n";
      for (size_t i = 0; i < p.ids.size(); ++i)
        f << field(p.ids[i]) << '\t'
          << (std::isfinite(p.y[i]) ? "reference" : "candidate") << '\t'
          << m.u[i] << '\t' << m.yhat[i] << '\n';
    });
    write_file(dir / "model.tsv", [&](std::ostream &g) {
      g << "Vu\tVe\tlambda\tstatus\tprofile_objective\treference\tcandidate\n"
        << m.vu << '\t' << m.ve << '\t' << m.lambda << '\t' << m.status << '\t'
        << m.objective << '\t' << p.ref.n_elem << '\t'
        << p.ids.size() - p.ref.n_elem << '\n';
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
