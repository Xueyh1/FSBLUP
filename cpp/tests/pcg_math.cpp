#include "cache.hpp"
#include "core.hpp"
#include "linear.hpp"
#include <iostream>
using namespace fsb;
void close(double x, double y, double tol) {
  require(std::abs(x - y) < tol,
          "mismatch " + std::to_string(x) + " vs " + std::to_string(y));
}
int main() {
  try {
    Table ped{{"ID", "Sire", "Dam"},
              {{"kid", "sib1", "sib2"},
               {"sib2", "pa", "ma"},
               {"sib1", "pa", "ma"},
               {"ma", "0", "0"}}};
    PedigreeOperator sparse(ped);
    auto dense =
        pedigree(ped.column("ID"), ped.column("Sire"), ped.column("Dam"), 2)
            .subset(sparse.ids);
    close(arma::abs(
              sparse.apply(arma::eye(sparse.ids.size(), sparse.ids.size())) -
              dense.values)
              .max(),
          0, 1e-13);
    close(sparse.diag[positions(sparse.ids, {"kid"})[0]], 1.25, 1e-14);
    mat q = {{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}};
    q /= 2;
    vec ev = {2, 1, 2, 4}, z = {0, std::sqrt(3.), std::sqrt(5.), 3};
    mat k = q * arma::diagmat(ev) * q.t(), x(4, 1, arma::fill::ones);
    vec y = q * z;
    uvec t = {0, 1, 2, 3};
    Contrast c{ev.tail(3), z.tail(3)};
    double g = 1.4, e = .9;
    auto d = reml_derivatives(c, g, e);
    mat v = g * k + e * arma::eye(4, 4), vi = arma::inv_sympd(v), w = vi * x;
    mat P = vi - w * arma::inv_sympd(x.t() * w) * w.t();
    vec r = P * y;
    mat dr = arma::join_rows(k * r, r);
    close(arma::abs(d.information - .5 * dr.t() * P * dr).max(), 0, 1e-10);
    for (size_t j = 0; j < 2; ++j) {
      double h = 1e-5;
      double diff =
          (reml_derivatives(c, g + (j == 0) * h, e + (j == 1) * h).lik -
           reml_derivatives(c, g - (j == 0) * h, e - (j == 1) * h).lik) /
          (2 * h);
      close(d.score[j], diff, 1e-9);
    }
    {
      FitOptions extreme;
      extreme.initial_g = extreme.initial_e = 1e300;
      FitAudit audit;
      auto fit = fit_model(k, x, 1e-100 * y, t, extreme, audit);
      close(fit.vu / 1e-200, 2., 1e-5);
      close(fit.ve / 1e-200, 1., 1e-5);
    }
    auto legacy = solve(k, x, y, t);
    close(legacy.vu, 2, 1e-5);
    close(legacy.ve, 1, 1e-5);
    for (auto method : {"ai-reml", "em-reml"})
      for (auto solver : {"pcg", "direct"})
        for (double start : {.5, 3.}) {
          FitOptions o;
          o.solver = solver;
          o.varcomp = method;
          o.initial_g = start;
          o.initial_e = .7;
          o.score_tol = 1e-10;
          o.max_reml = 30000;
          FitAudit audit;
          auto m = fit_model(k, x, y, t, o, audit);
          close(m.vu, 2, 3e-7);
          close(m.ve, 1, 4e-7);
          close(m.lambda, .5, 2e-7);
          close(arma::abs(m.u - 2 * k * arma::solve(2 * k + arma::eye(4, 4), y))
                    .max(),
                0, 2e-7);
          for (size_t j = 1; j < audit.steps.size(); ++j)
            require(audit.steps[j].lik >= audit.steps[j - 1].lik - 1e-12,
                    "nonmonotone");
        }
    {
      mat design = arma::join_rows(x, 2 * q.col(1) + 20000);
      uvec swap = {1, 0};
      FitOptions fixed;
      fixed.varcomp = "fixed";
      fixed.lambda = .5;
      FitAudit state;
      auto left = fit_model(k, design, y, t, fixed, state);
      auto right = fit_model(k, design.cols(swap), y, t, fixed, state);
      close(arma::abs(left.u - right.u).max(), 0, 1e-10);
      close(arma::abs(left.yhat - right.yhat).max(), 0, 1e-10);
      close(arma::abs(left.b - right.b.elem(swap)).max(), 0, 1e-8);
    }
    FitOptions o;
    o.varcomp = "em-reml";
    {
      FitAudit invalid;
      try {
        fit_model(k, x, y, uvec{5}, o, invalid);
        throw std::logic_error("unvalidated training index");
      } catch (const std::runtime_error &err) {
        require(string(err.what()).find("indices") != string::npos, err.what());
      }
    }
    o.max_reml = 1;
    FitAudit a;
    try {
      fit_model(k, x, y, t, o, a);
      throw std::logic_error("false EM convergence");
    } catch (const std::runtime_error &err) {
      require(string(err.what()).find("iteration-limit") != string::npos,
              err.what());
    }
    o.varcomp = "ai-reml";
    o.max_reml = 100;
    auto zero = fit_model(k, x, q.col(1), t, o, a);
    require(zero.vu == 0 && arma::norm(zero.u) == 0, "g=0 KKT");
    auto e0 = fit_model(k, x, q.col(3), t, o, a);
    require(e0.ve == 0 && e0.status == "residual-zero", "e=0 KKT");
    TrainKernel kernel{k.diag(), [&](const mat &b) -> mat { return k * b; },
                       [&](const mat &b) -> mat { return k * b; },
                       [&] { return k; }};
    System sys(kernel, g, e, o, a);
    Projection proj(sys, x);
    close(arma::abs(proj.apply_P(y) - P * y).max(), 0, 1e-10);
    require(arma::norm(sys.solve_one(arma::zeros<vec>(4), "zero")) == 0,
            "zero RHS");
    kernel.apply = [](const mat &b) -> mat { return -2 * b; };
    try {
      sys.solve_one(y, "negative");
      throw std::logic_error("missed curvature failure");
    } catch (const std::runtime_error &ex) {
      require(string(ex.what()).find("curvature") != string::npos, ex.what());
    }
    auto flat = fit_model(arma::eye(4, 4), x, y, t, o, a);
    require(flat.status.find("unidentifiable") != string::npos,
            "unidentifiable");
    o.varcomp = "fixed";
    o.lambda = .5;
    o.max_pcg = 1;
    try {
      fit_model(k, x, y, t, o, a);
      throw std::logic_error("false PCG convergence");
    } catch (const std::runtime_error &err) {
      require(string(err.what()).find("PCG iteration-limit") != string::npos,
              err.what());
    }
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
