#include "core.hpp"
namespace fsb {
namespace {
struct System {
  const TrainKernel& k;
  double g,e;
  const FitOptions& opt;
  FitAudit& audit;
  vec diag;
  mat lower;
  System(const TrainKernel& kernel,double genetic,double noise,const FitOptions& o,FitAudit& a):
    k(kernel),g(genetic),e(noise),opt(o),audit(a),diag(g*k.diagonal+e){
    require(diag.is_finite()&&diag.min()>0,"nonpositive covariance diagonal");
    if(opt.solver=="direct"){
      mat v=g*k.dense();v.diag()+=e;require(arma::chol(lower,v,"lower"),"direct factorization failed");
    }
  }
  vec apply(const vec& v){++audit.matvec; return g*k.apply(v)+e*v;}
  vec solve_one(const vec& b,const std::string& rhs){
    ++audit.rhs;double norm=arma::norm(b);
    require(b.is_finite()&&std::isfinite(norm),"nonfinite RHS "+rhs);
    vec x(b.n_elem,arma::fill::zeros);
    if(norm==0)return x;
    const double tol=opt.atol+opt.rtol*norm;
    auto verify=[&](const vec& r){return arma::norm(r)<=tol;};
    auto success=[&](const vec& r,int it){
      audit.residual_ratio=std::max(audit.residual_ratio,arma::norm(r)/tol);
      audit.max_iterations=std::max(audit.max_iterations,size_t(it));
    };
    if(opt.solver=="direct"){
      x=arma::solve(arma::trimatu(lower.t()),arma::solve(arma::trimatl(lower),b));
      vec r=b-apply(x);require(verify(r),"direct true residual failed rhs="+rhs);success(r,1);return x;
    }
    vec r=b,z=r;
    if(opt.preconditioner=="jacobi")z/=diag;
    vec p=z;double rho=arma::dot(r,z);
    for(int it=1;it<=opt.max_pcg;++it){
      vec ap=apply(p);double curvature=arma::dot(p,ap);
      require(ap.is_finite()&&std::isfinite(curvature)&&std::isfinite(rho),"PCG nonfinite recurrence rhs="+rhs);
      require(curvature>0&&rho>0,"PCG nonpositive curvature rhs="+rhs);
      double step=rho/curvature;vec next=x+step*p;
      bool stalled=arma::all(next==x);x=std::move(next);r-=step*ap;
      require(x.is_finite()&&r.is_finite(),"PCG nonfinite iterate rhs="+rhs);
      if(opt.event && it%opt.log_every==0)opt.event(audit.context+" rhs="+rhs+" pcg="+std::to_string(it)+" residual="+std::to_string(arma::norm(r)));
      if(verify(r)||stalled||it==opt.max_pcg){
        r=b-apply(x);
        if(verify(r)){success(r,it);return x;}
        require(!stalled,"PCG stagnation rhs="+rhs);
        require(it<opt.max_pcg,"PCG iteration-limit rhs="+rhs+" true_residual="+std::to_string(arma::norm(r)));
        z=r;if(opt.preconditioner=="jacobi")z/=diag;p=z;rho=arma::dot(r,z);continue;
      }
      z=r;if(opt.preconditioner=="jacobi")z/=diag;
      double nr=arma::dot(r,z);p=z+(nr/rho)*p;rho=nr;
    }
    throw std::runtime_error("invalid PCG iteration limit");
  }
};
struct Projection {
  System& sys;
  const mat& x;
  mat w,lower;
  Projection(System& s,const mat& design):sys(s),x(design),w(x.n_rows,x.n_cols){
    for(size_t j=0;j<x.n_cols;++j)w.col(j)=sys.solve_one(x.col(j),"X"+std::to_string(j));
    mat c=x.t()*w;c=.5*c+.5*c.t();
    require(arma::rcond(c)>1e-14&&arma::chol(lower,c,"lower"),"ill-conditioned fixed-effect system");
  }
  vec small(const vec& rhs)const{return arma::solve(arma::trimatu(lower.t()),arma::solve(arma::trimatl(lower),rhs));}
  vec project(const vec& h,vec& b)const{b=small(x.t()*h);return h-w*b;}
  vec apply_P(const vec& v){vec b;return project(sys.solve_one(v,"P"),b);}
};
}
Model fit_operator(const TrainKernel& kernel,const mat& xt,const mat& x,const vec& y,const FitOptions& o,FitAudit& audit){
  audit.n=y.n_elem;
  size_t n=y.n_elem,p=x.n_cols;
  require(xt.n_rows==n&&xt.n_cols==p&&n>p&&p&&arma::rank(xt)==p,"training design rank deficient or no residual degrees of freedom");
  require(y.is_finite()&&xt.is_finite()&&x.is_finite(),"nonfinite training phenotype or design");
  int initial=0,center=0;
  vec yt=y;std::frexp(arma::abs(yt).max(),&initial);
  yt.transform([&](double v){return std::scalbn(v,-initial);});
  double offset=arma::all(x.col(0)==1)?yt[0]:0;yt-=offset;
  std::frexp(arma::abs(yt).max(),&center);yt.transform([&](double v){return std::scalbn(v,-center);});
  int exponent=initial+center;
  vec ols;require(arma::solve(ols,xt,yt,arma::solve_opts::no_approx),"fixed effect rank");
  require(arma::norm(yt-xt*ols)>64*arma::datum::eps*arma::norm(yt),"constant response or exact fixed-effect fit");
  double g=1,e=o.lambda,lik=NAN,objective=NAN;string status="fixed";
  if(o.varcomp!="fixed"){
    require(o.trace=="exact","unsupported trace backend");
    Contrast c;residual_spectrum(kernel.dense(),xt,yt,c.eigen,c.z);
    double v=arma::var(yt);
    if(v==0)v=arma::dot(yt,yt)/n;
    double g0=o.initial_g?std::scalbn(o.initial_g,-2*exponent):v/2;
    double e0=o.initial_e?std::scalbn(o.initial_e,-2*exponent):v/2;
    require(g0>0&&e0>0&&std::isfinite(g0+e0),"invalid initial variances");
    auto est=estimate_exact(c,n,g0,e0,o,audit);g=est.g;e=est.e;status=est.status;
    lik=est.lik-(n-p)*exponent*std::log(2.);
    // Historical profile objective = -2 contrast likelihood - df at a profiled scale.
    objective=-2*lik-(n-p);
  }
  System sys(kernel,g,e,o,audit);Projection P(sys,xt);
  vec b,r=P.project(sys.solve_one(yt,"y"),b),res=yt-xt*b;
  auto check=[&]{
    audit.projected_residual=arma::norm(sys.apply(r)-res)/(o.atol+o.rtol*arma::norm(res));
    audit.orthogonality=arma::norm(xt.t()*r)/(o.atol+o.rtol*arma::norm(xt,"fro")*arma::norm(r));
    return audit.projected_residual<=1 && audit.orthogonality<=1;
  };
  if(!check()){
    r=sys.solve_one(res,"centered");vec correction=P.small(xt.t()*r);
    b+=correction;r-=P.w*correction;res=yt-xt*b;
    require(check(),"projected residual or orthogonality failed after centering");
  }
  vec u(x.n_rows,arma::fill::zeros);if(g>0)u=g*kernel.cross(r);
  if(o.varcomp=="fixed"){g=arma::dot(res,r)/(n-p);e=o.lambda*g;require(g>0,"nonpositive fixed residual variance");}
  vec yh=x*b+u;
  for(auto& v:yh)v=std::scalbn(std::scalbn(v,center)+offset,initial);
  for(auto& v:u)v=std::scalbn(v,exponent);
  for(size_t j=0;j<p;++j)b[j]=std::scalbn(std::scalbn(b[j],center)+(j==0?offset:0.),initial);
  double vu=std::scalbn(g,2*exponent),ve=std::scalbn(e,2*exponent);
  require(std::isfinite(vu)&&std::isfinite(ve)&&(!g||vu>0)&&(!e||ve>0),"variance components outside the representable range in phenotype units");
  require(b.is_finite()&&u.is_finite()&&yh.is_finite(),"nonfinite predictions in phenotype units");
  for(auto& it:audit.steps){
    it.g=std::scalbn(it.g,2*exponent);it.e=std::scalbn(it.e,2*exponent);
    it.lik-=(n-p)*exponent*std::log(2.);it.sg=std::scalbn(it.sg,-2*exponent);it.se=std::scalbn(it.se,-2*exponent);
  }
  audit.status=status;
  return {b,u,yh,g?e/g:INFINITY,vu,ve,objective,status};
}
Model fit_model(const mat& k,const mat& x,const vec& y,const uvec& t,const FitOptions& o,FitAudit& a){
  if(o.varcomp=="profile-reml"){
    auto m=solve(k,x,y,t);
    if(o.solver=="direct"){a.n=t.n_elem;a.status=m.status;return m;}
    FitOptions fixed=o;fixed.varcomp="fixed";fixed.lambda=m.lambda;
    auto result=fit_model(k,x,y,t,fixed,a);result.objective=m.objective;result.status=m.status;a.status=m.status;return result;
  }
  mat kt=k.submat(t,t);
  TrainKernel op{kt.diag(),[&](const mat& v)->mat{return kt*v;},
    [&](const mat& v)->mat{return k.cols(t)*v;},[&]{return kt;}};
  return fit_operator(op,x.rows(t),x,y.elem(t),o,a);
}
}
