#include "cache.hpp"
#include <fstream>
#include <queue>
#include <openssl/evp.h>
#include <iomanip>
namespace fsb {
namespace {
string digest(const fs::path& path){
  std::ifstream in(path,std::ios::binary);require(bool(in),"cannot read "+path.string());
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  require(ctx&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"SHA256 init");
  char buf[65536];while(in){in.read(buf,sizeof(buf));require(EVP_DigestUpdate(ctx.get(),buf,in.gcount())==1,"SHA256 update");}
  require(in.eof(),"cache read error");unsigned char bytes[EVP_MAX_MD_SIZE];unsigned len=0;
  require(EVP_DigestFinal_ex(ctx.get(),bytes,&len)==1,"SHA256 finalize");
  std::ostringstream out; for(unsigned j=0;j<len;++j)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(bytes[j]);return out.str();
}
using Weights=std::map<size_t,double,std::greater<size_t>>;
Weights ancestors(long parent,const std::vector<long>& sire,const std::vector<long>& dam){
  Weights result,pending;pending[parent]=1;
  while(!pending.empty()){
    auto [i,w]=*pending.begin();pending.erase(pending.begin());result[i]=w;
    for(long p:{sire[i],dam[i]})if(p>=0)pending[p]+=.5*w;
  }
  return result;
}
struct Moment{size_t count[3]{},missing=0,low=0;double mean=0,m2=0;};
void pack_features(const fs::path& input,const fs::path& dir,string name,bool snp,bool fpkm,double cap){
  Ids ids,header;std::vector<Moment> moments;
  scan_table(input,[&](auto& h){header=h;require(h.size()>1&&h.size()*160./1048576.<cap,"feature header exceeds memory budget");moments.resize(h.size()-1);unique_ids(Ids(h.begin()+1,h.end()));},
    [&](auto& row){ids.push_back(row[0]);size_t n=ids.size();
      for(size_t j=0;j<moments.size();++j){auto& m=moments[j];double v=missing(row[j+1])?NAN:number(row[j+1]);
        if(std::isnan(v)||(snp&&v==-9)){++m.missing;if(snp)continue;v=0;}
        if(snp){require(v==0||v==1||v==2,"invalid genotype");++m.count[int(v)];}
        else {m.low+=v<.1;double delta=v-m.mean;m.mean+=delta/n;m.m2+=delta*(v-m.mean);}
      }
    });
  unique_ids(ids);size_t n=ids.size();require(n>=2,"two or more feature rows required");
  std::vector<size_t> keep;std::vector<double> impute,center,divisor;double denom=0;
  write_file(dir/(name+".stats.tsv"),[&](auto& out){out<<"feature\tkeep\tmissing\timpute\tcenter\tdivisor\tp\tq\n";
    for(size_t j=0;j<moments.size();++j){auto& m=moments[j];bool retain=true;double fill=0,mu=m.mean,sd=1,p=NAN,q=NAN;
      if(snp){retain=m.missing<=n/2;if(retain){fill=(2.*m.count[2]+m.count[1])/(n-m.missing);if(fill==0||fill==1||fill==2)m.count[int(fill)]+=m.missing;
        p=(2.*m.count[2]+m.count[1])/(2*n);q=(2.*m.count[0]+m.count[1])/(2*n);mu=2*p;denom+=2*p*q;}}
      else{sd=std::sqrt(m.m2/(n-1));retain=sd>0&&(!fpkm||m.low*100<95*n);}
      if(retain){keep.push_back(j);impute.push_back(fill);center.push_back(mu);divisor.push_back(sd);}
      out<<field(header[j+1])<<'\t'<<retain<<'\t'<<m.missing<<'\t'<<fill<<'\t'<<mu<<'\t'<<sd<<'\t'<<p<<'\t'<<q<<'\n';
    }});
  if(!snp)denom=keep.size();require(!keep.empty()&&denom>0,"no usable "+name+" features");
  vec diag(n);size_t i=0;std::vector<double> packed(keep.size());
  // Store normalized row-major doubles. Linux binary output uses the same checked close as TSV.
  write_file(dir/(name+".bin"),[&](auto& out){
    scan_table(input,[&](auto& h){require(h==header,"feature header changed");},[&](auto& row){
      require(i<n&&row[0]==ids[i],"feature IDs changed during prepare");double sum=0;
      for(size_t j=0;j<keep.size();++j){auto& s=row[keep[j]+1];double v=missing(s)?NAN:number(s);
        if(std::isnan(v)||(snp&&v==-9))v=impute[j];
        v=(v-center[j])/(divisor[j]*std::sqrt(denom));require(std::isfinite(v),"nonfinite standardized feature");packed[j]=v;sum+=v*v;}
      diag[i++]=sum;out.write(reinterpret_cast<const char*>(packed.data()),packed.size()*sizeof(double));
    });require(i==n,"feature input truncated");});
  write_file(dir/(name+".ids.tsv"),[&](auto& f){f<<"ID\tdiagonal\n";for(size_t j=0;j<n;++j)f<<field(ids[j])<<'\t'<<diag[j]<<'\n';});
  write_file(dir/(name+".shape"),[&](auto& f){f<<n<<' '<<keep.size()<<' '<<denom<<'\n';});
}
const std::vector<string> files={"ped.tsv","G.bin","G.shape","G.ids.tsv","G.stats.tsv","O.bin","O.shape","O.ids.tsv","O.stats.tsv","sources.tsv"};
fs::path validate(const fs::path& dir){
 std::ifstream f(dir/"manifest");string line;require(std::getline(f,line)&&line=="fsblup-cache 1 little-endian f64","invalid cache version");
 for(auto& file:files){string hash,key;require(bool(f>>hash>>key)&&key==file&&hash==digest(dir/file),"cache integrity failure: "+file);}
 require(!(f>>line),"unexpected cache manifest entries");return dir/"ped.tsv";
}
FeatureFile load_features(const fs::path& dir,string layer){
 FeatureFile f;f.path=dir/(layer+".bin");std::ifstream shape(dir/(layer+".shape"));size_t n=0;double denom;
 require(bool(shape>>n>>f.columns>>denom)&&n&&f.columns&&denom>0&&fs::file_size(f.path)/sizeof(double)/n==f.columns&&fs::file_size(f.path)==double(n)*f.columns*8,"invalid cached dimensions");
 auto t=read_table(dir/(layer+".ids.tsv"));f.ids=t.column("ID");unique_ids(f.ids);require(f.ids.size()==n,"invalid cache ID count");f.diag.set_size(n);
 for(size_t j=0;j<n;++j)f.diag[j]=number(t.rows[j][1]);return f;
}
}
PedigreeOperator::PedigreeOperator(const Table& input){
 Table p=input;for(auto& h:p.header)for(char& c:h)c=std::tolower(static_cast<unsigned char>(c));
 auto names=p.column("id"),ss=p.column("sire"),dd=p.column("dam");unique_ids(names);
 std::map<string,size_t> index;for(size_t i=0;i<names.size();++i){require(names[i]!="0","0 reserved");index[names[i]]=i;}
 for(auto parents:{&ss,&dd})for(auto& id:*parents){require(!missing(id),"unknown parent must be 0");if(id!="0"&&!index.count(id)){index[id]=names.size();names.push_back(id);}}
 size_t n=names.size();std::vector<std::vector<size_t>> child(n);std::vector<int> deg(n);
 for(size_t i=0;i<ss.size();++i)for(auto parent:{ss[i],dd[i]})if(parent!="0"){child[index.at(parent)].push_back(i);++deg[i];}
 std::queue<size_t> ready;for(size_t i=0;i<n;++i)if(!deg[i])ready.push(i);
 std::vector<long> topo(n,-1);diag.set_size(n);d.set_size(n);
 while(!ready.empty()){size_t orig=ready.front();ready.pop();size_t i=ids.size();topo[orig]=i;ids.push_back(names[orig]);
   long s=-1,m=-1;if(orig<ss.size()){if(ss[orig]!="0")s=topo[index.at(ss[orig])];if(dd[orig]!="0")m=topo[index.at(dd[orig])];}
   sire.push_back(s);dam.push_back(m);d[i]=1-.25*((s>=0?diag[s]:0)+(m>=0?diag[m]:0));diag[i]=1;
   if(s>=0&&m>=0){auto u=ancestors(s,sire,dam),v=ancestors(m,sire,dam);double r=0;for(auto [j,w]:u){auto it=v.find(j);if(it!=v.end())r+=w*d[j]*it->second;}diag[i]+=.5*r;}
   require(d[i]>=0,"negative Mendelian sampling variance");
   for(auto c:child[orig])if(--deg[c]==0)ready.push(c);
 }
 require(ids.size()==n,"cycle in pedigree");
}
mat PedigreeOperator::apply(mat x)const{
 for(size_t col=0;col<x.n_cols;++col){auto v=x.colptr(col);
 for(size_t i=ids.size();i-->0;){double h=.5*v[i];if(sire[i]>=0)v[sire[i]]+=h;if(dam[i]>=0)v[dam[i]]+=h;}
 for(size_t i=0;i<ids.size();++i){v[i]*=d[i];if(sire[i]>=0)v[i]+=.5*v[sire[i]];if(dam[i]>=0)v[i]+=.5*v[dam[i]];}}
 return x;
}
mat FeatureFile::apply(const mat& x,size_t block,std::atomic<uint64_t>& bytes)const{
 std::ifstream f(path,std::ios::binary);require(bool(f),"unreadable cache block");mat coeff(columns,x.n_cols,arma::fill::zeros),result(x.n_rows,x.n_cols);
 for(int pass=0;pass<2;++pass){f.seekg(0);for(size_t i=0;i<x.n_rows;i+=block){size_t len=std::min(block,size_t(x.n_rows)-i);mat w(columns,len,arma::fill::none);size_t count=w.n_elem*8;
 f.read(reinterpret_cast<char*>(w.memptr()),count);require(bool(f),"truncated feature cache");bytes.fetch_add(count,std::memory_order_relaxed);
 if(pass==0)coeff+=w*x.rows(i,i+len-1);else result.rows(i,i+len-1)=w.t()*coeff;}}
 return result;
}
void prepare_cache(const fs::path& pedigree,const fs::path& snp,const fs::path& omics,bool fpkm,const fs::path& dir,double cap){
 require(cap>0,"positive memory budget required");auto p=read_table(pedigree);PedigreeOperator op(p);
 auto orig=std::vector<fs::path>{pedigree,snp,omics};std::vector<string> hashes;for(auto& f:orig)hashes.push_back(digest(f));
 write_file(dir/"ped.tsv",[&](auto& f){f<<"ID\tSire\tDam\n";for(size_t i=0;i<op.ids.size();++i)f<<field(op.ids[i])<<'\t'<<(op.sire[i]<0?"0":field(op.ids[op.sire[i]]))<<'\t'<<(op.dam[i]<0?"0":field(op.ids[op.dam[i]]))<<'\n';});
 pack_features(snp,dir,"G",true,false,cap);pack_features(omics,dir,"O",false,fpkm,cap);
 write_file(dir/"sources.tsv",[&](auto& f){f<<"path\tsha256\tomics_type\n";for(size_t i=0;i<orig.size();++i){require(digest(orig[i])==hashes[i],"source changed during prepare");f<<field(orig[i].string())<<'\t'<<hashes[i]<<'\t'<<(fpkm?"fpkm":"generic")<<'\n';}});
 write_file(dir/"manifest",[&](auto& f){f<<"fsblup-cache 1 little-endian f64\n";for(auto& file:files)f<<digest(dir/file)<<' '<<file<<'\n';});
}
Cache::Cache(const fs::path& dir):ped(read_table(validate(dir))),g(load_features(dir,"G")),o(load_features(dir,"O")){}
string Cache::mode()const{return Layers{{ped.ids,{}},{g.ids,{}},{o.ids,{}}}.mode();}
Layers Cache::materialize(double cap){
 budget(ped.ids.size(),std::max(g.columns,o.columns),cap);
 return {{ped.ids,ped.apply(arma::eye(ped.ids.size(),ped.ids.size()))},{g.ids,g.apply(arma::eye(g.ids.size(),g.ids.size()),64,bytes)},{o.ids,o.apply(arma::eye(o.ids.size(),o.ids.size()),64,bytes)}};
}
TrainKernel Cache::kernel(const Ids& pop,const uvec& train,double a,double b,double cap){
 require(mode()=="A" && a>=0&&b>=0&&a+b<=1,"feature operator requires common coverage and simplex weights");
 auto pa=positions(ped.ids,pop),pg=positions(g.ids,pop),po=positions(o.ids,pop);
 auto ta=pa.elem(train).eval(),tg=pg.elem(train).eval(),tt=po.elem(train).eval();
 double remaining=cap*1048576.-(ped.ids.size()*128.+pop.size()*128.+std::max(g.columns,o.columns)*32.);
 require(remaining>=8.*std::max(g.columns,o.columns),"memory budget insufficient for feature operator");
 size_t block=std::min(4096.,std::floor(remaining/(8.*std::max(g.columns,o.columns))));
 auto multiply=[this,ta,tg,tt,pa,pg,po,a,b,block,train](const mat& x,bool cross)->mat{
   mat all(ped.ids.size(),x.n_cols,arma::fill::zeros);auto& ra=cross?pa:ta;auto& rg=cross?pg:tg;auto& ro=cross?po:tt;
   mat out(ra.n_elem,x.n_cols,arma::fill::zeros);
   if(1-a-b!=0){all.rows(ta)=x;mat v=ped.apply(all);out+=(1-a-b)*v.rows(ra);}
   if(b!=0){all.zeros();all.rows(tg)=x;mat v=g.apply(all,block,bytes);out+=b*v.rows(rg);}
   if(a!=0){all.zeros();all.rows(tt)=x;mat v=o.apply(all,block,bytes);out+=a*v.rows(ro);}
   if(cross)out.rows(train)+=1e-6*x;else out+=1e-6*x;return out;
 };
 vec diag=(1-a-b)*ped.diag.elem(ta)+b*g.diag.elem(tg)+a*o.diag.elem(tt)+1e-6;
 return {diag,[multiply](const mat& x){return multiply(x,false);},[multiply](const mat& x){return multiply(x,true);},
 [multiply,n=train.n_elem,cap,cols=std::max(g.columns,o.columns),total=ped.ids.size()]()->mat{
   require((total*6.+cols*2.)*n*8/1048576.+160.*n*n/1048576.<cap,"exact trace/direct training materialization exceeds memory budget; use fixed PCG or stochastic trace");
   // Materialize one column at a time to keep the feature workspace bounded.
   mat k(n,n);for(size_t j=0;j<n;++j){vec e(n,arma::fill::zeros);e[j]=1;k.col(j)=multiply(e,false);}return .5*k+.5*k.t();
 }};
}
}
