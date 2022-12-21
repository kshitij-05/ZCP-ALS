//
// Created by Kshitij Surjuse on 11/8/22.
//
#include <btas/btas.h>
#include <btas/tensor.h>
#include <random>
#include <btas/generic/gesvd_impl.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <complex>

using std::cout;
using std::endl;
using std::string;
typedef btas::Tensor<std::complex<double>> tensor;
using std::complex;
using namespace btas;

double fRand(double fMin, double fMax){
    double f = (double)std::rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


tensor rand_fact_mat(const unsigned long& n1, const int& rank){
    tensor fact (n1,rank);
    double lower = -1.0;
    double upper = 1.0;
    for(auto i=0;i<n1;i++){
        for(auto j=0;j<rank;j++){
            complex t(fRand(lower,upper) ,fRand(lower,upper));
            fact(i,j) = t;
        }
    }
    return fact;
}

tensor norm_fact_mat(const tensor& fact){
    auto rows = fact.extent(0);
    auto cols = fact.extent(1);
    double col_norm = 0.0;
    tensor norm_fact(rows,cols);
    for (auto j=0; j<cols;j++){
        for(auto i=0;i< rows;i++){
            col_norm+= norm(fact(i,j));
        }
        for(auto i=0;i< rows;i++){
            norm_fact(i,j) = fact(i,j)/ sqrt(col_norm);
        }
        col_norm = 0.0;
    }
    return norm_fact;
}

tensor matmul(const tensor& A, const tensor& B){
    tensor C(A.extent(0),B.extent(1));
    for (auto i=0; i < A.extent(0) ;i++){
        for(auto j=0; j < B.extent(1) ; j++){
            for(auto k=0; k< A.extent(1); k++){
                C(i,j) += A(i,k)*B(k,j);
            }
        }
    }
    return C;
}

tensor transpose_conj(const tensor& fact){
    tensor factH (fact.extent(1),fact.extent(0));
    for(int i=0;i< factH.extent(0);i++){
        for(int j=0;j<factH.extent(1);j++){
            factH(i,j) = conj(fact(j,i));
        }
    }
    return factH;
}

tensor transpose(const tensor& fact){
    tensor factH (fact.extent(1),fact.extent(0));
    for(int i=0;i< factH.extent(0);i++){
        for(int j=0;j<factH.extent(1);j++){
            factH(i,j) = fact(j,i);
        }
    }
    return factH;
}


tensor compute_w(const tensor& fact1,const tensor& fact2,const tensor& fact3){
    tensor f1hf1 = matmul(transpose_conj(fact1),fact1);
    tensor f2hf2 = matmul(transpose_conj(fact2),fact2);
    tensor f3hf3 = matmul(transpose_conj(fact3),fact3);
    auto rank = f2hf2.extent(1);
    tensor W(rank,rank);
    for(auto i=0;i<rank;i++){
        for(auto j=0;j<rank;j++){
            W(i,j) += f1hf1(i,j)*f2hf2(i,j)*f3hf3(i,j);
        }
    }
    return W;
}

tensor compute_v(const tensor& G,int indx,const unsigned long& firstf1,
                 const tensor& f2,const tensor& f3,const tensor& f4){
    auto rank = f2.extent(0);
    tensor V(rank,G.extent(indx));
        complex<double> alpha(1.0,0.0);
        complex<double> beta (0.0,0.0);
        for(auto p=0;p< G.extent(0);p++) {
            for (auto q = 0; q < G.extent(1); q++) {
                for (auto s = 0; s < G.extent(2); s++) {
                    for (auto t = 0; t < G.extent(3); t++) {
                        for (auto r = 0; r < rank; r++) {
                            //'pqst,rp,rs,rt->rq'   'pqst,rp,rq,rt->rs'   'pqst,rp,rq,rs->rt'
                            if( indx == 0) {
                                V(r, p) += G(p, q, s, t) * f2(r, q) * f3(r, s) * f4(r, t);
                            }
                            else if(indx==1){
                                V(r, q) += G(p, q, s, t) * f2(r, p) * f3(r, s) * f4(r, t);
                            }
                            else if(indx==2){
                                V(r, s) += G(p, q, s, t) * f2(r, p) * f3(r, q) * f4(r, t);
                            }
                            else if(indx==3){
                                V(r, t) += G(p, q, s, t) * f2(r, p) * f3(r, q) * f4(r, s);
                            }
                        }
                    }
                }
            }
        }
    return V;
}


tensor compute_s_inv(const Tensor<double>& S){
    tensor s_inv(S.extent(0) , S.extent(0));
    s_inv.fill(0.0);
    for(auto i=0;i<S.extent(0);i++){
        complex<double> val(1.0/S(i) , 0.0 );
        s_inv(i,i) = val;
    }
    return s_inv;
}


tensor compute_factm(const tensor& Vh, const tensor& U, const Tensor<double>& S, const tensor& V){
    tensor vhh = transpose_conj(Vh);
    tensor uh = transpose_conj(U);
    tensor s_inv = compute_s_inv(S);
    complex<double> alpha(1.0,0.0);
    complex<double> beta (0.0,0.0);
    tensor vhhs_inv(vhh.extent(0),s_inv.extent(1));
    tensor vhhs_invuh(vhh.extent(0),uh.extent(1));
    tensor fact( V.extent(0),V.extent(1));
    contract(alpha , vhh , {'i','j'} , s_inv ,{'j','k'} ,beta, vhhs_inv,{'i','k'});
    contract(alpha , vhhs_inv , {'i','k'} , uh, {'k','l'} ,beta, vhhs_invuh , {'i','l'});
    contract(alpha , vhhs_invuh , {'i','l'} , V , {'l','m'} , beta , fact , {'i','m'});
    return fact;
}

double compute_err(const tensor& G,const tensor& f1,const tensor& f2,const tensor& f3,const tensor& f4){
    tensor G_ (G.extent(0), G.extent(1),G.extent(2),G.extent(3));
    for(auto p=0;p< G.extent(0);p++) {
        for (auto q = 0; q < G.extent(1); q++) {
            for (auto s = 0; s < G.extent(2); s++) {
                for (auto t = 0; t < G.extent(3); t++) {
                    for (auto r = 0; r < f1.extent(1); r++) {
                        G_(p,q,s,t) += f1(p,r)*f2(q,r)*f3(s,r)*f4(t,r);
                    }
                }
            }
        }
    }
    double err = 0.0;
    for(auto p=0;p< G.extent(0);p++) {
        for (auto q = 0; q < G.extent(1); q++) {
            for (auto s = 0; s < G.extent(2); s++) {
                for (auto t = 0; t < G.extent(3); t++) {
                    err += norm(G(p,q,s,t) - G_(p,q,s,t));
                }
            }
        }
    }
    return err;
}


int main(int argc, char* argv[]){
    std::srand(42);
    tensor G(4,2,7,3);
    auto n1 = G.extent(0);
    std::ifstream inp ("4D_complex.txt");
    cout << std::setprecision(12);
    if (inp.is_open()){
        string line;
        int i,j,k,l;
        double rel,img;
        while(inp){
            inp >> i >> j >> k>>l>> rel>> img;
            complex t1(rel,img);
            G(i,j,k,l) = t1;
        }
    }
    int rank;
    std::cin >> rank;
    tensor f1 = rand_fact_mat(G.extent(0),rank);
    f1 = norm_fact_mat(f1);
    tensor f2 = rand_fact_mat(G.extent(1),rank);
    f2 = norm_fact_mat(f2);
    tensor f3 = rand_fact_mat(G.extent(2),rank);
    f3 = norm_fact_mat(f3);
    tensor f4 = rand_fact_mat(G.extent(0),rank);
    f4 = norm_fact_mat(f4);
    tensor W,V;
    double err = 0.0;

    for (int i=0;i<1000;i++) {
        W = compute_w(f2, f3, f4);
        V = compute_v(G, 0, rank, transpose_conj(f2), transpose_conj(f3), transpose_conj(f4));
        tensor U(W.extent(0), W.extent(1));
        tensor Vh(W.extent(0), W.extent(1));
        Tensor<double> S(W.extent(0));
        gesvd(lapack::Job::AllVec, lapack::Job::AllVec, W, S, U, Vh);
        tensor new_f1 = compute_factm(Vh, U, S, V);
        new_f1 = transpose(new_f1);
        new_f1 = norm_fact_mat(new_f1);

        W = compute_w(new_f1, f3, f4);
        V = compute_v(G, 1, rank, transpose_conj(new_f1), transpose_conj(f3), transpose_conj(f4));
        gesvd(lapack::Job::AllVec, lapack::Job::AllVec, W, S, U, Vh);
        tensor new_f2 = compute_factm(Vh, U, S, V);
        new_f2 = transpose(new_f2);
        new_f2 = norm_fact_mat(new_f2);

        W = compute_w(new_f1, new_f2, f4);
        V = compute_v(G, 2, rank, transpose_conj(new_f1), transpose_conj(new_f2), transpose_conj(f4));
        gesvd(lapack::Job::AllVec, lapack::Job::AllVec, W, S, U, Vh);
        tensor new_f3 = compute_factm(Vh, U, S, V);
        new_f3 = transpose(new_f3);
        new_f3 = norm_fact_mat(new_f3);

        W = compute_w(new_f1, new_f2, new_f3);
        V = compute_v(G, 3, rank, transpose_conj(new_f1), transpose_conj(new_f2), transpose_conj(new_f3));
        gesvd(lapack::Job::AllVec, lapack::Job::AllVec, W, S, U, Vh);
        tensor new_f4 = compute_factm(Vh, U, S, V);
        new_f4 = transpose(new_f4);

        double new_err = compute_err(G, new_f1, new_f2, new_f3, new_f4);
        err = new_err;
        cout << err << endl;
        new_f4 = norm_fact_mat(new_f4);
        f1 = new_f1;
        f2 = new_f2;
        f3 = new_f3;
        f4 = new_f4;
    }
    return 0;
}