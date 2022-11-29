//
// Created by Kshitij Surjuse on 11/8/22.
//
#include <btas/btas.h>
#include <btas/tensor.h>
#include <random>
#include <btas/generic/converge_class.h>
#include <btas/generic/gesvd_impl.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>
#include <libgen.h>
#include <cmath>
#include <complex>

using std::cout;
using std::endl;
using std::string;
typedef btas::Tensor<std::complex<double>> tensor;
using std::complex;


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


tensor compute_w(const tensor& fact1,const tensor& fact2,const tensor& fact3){
    tensor f1hf1 = matmul(transpose_conj(fact1),fact1);
    tensor f2hf2 = matmul(transpose_conj(fact2),fact2);
    tensor f3hf3 = matmul(transpose_conj(fact3),fact3);
    auto rank = f2hf2.extent(0);
    tensor W(rank,rank);
    for(auto i=0;i<rank;i++){
        for(auto j=0;j<rank;j++){
            W(i,j) += f1hf1(i,j)*f2hf2(i,j)*f3hf3(i,j);
        }
    }
    return W;
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

    int rank = 3;
    tensor f1 = rand_fact_mat(G.extent(0),rank);
    f1 = norm_fact_mat(f1);
    tensor f2 = rand_fact_mat(G.extent(1),rank);
    f2 = norm_fact_mat(f2);
    tensor f3 = rand_fact_mat(G.extent(2),rank);
    f3 = norm_fact_mat(f3);
    tensor f4 = rand_fact_mat(G.extent(0),rank);
    f4 = norm_fact_mat(f4);

    tensor W;
    W = compute_w(f2,f3,f4);
    tensor V(rank,G.extent(0));
    const char p ='p';
    const char q ='q';
    const char s ='s';
    const char t ='t';
    const char r ='r';
    //'pqst,rq,rs,rt->rp'
    tensor prst(G.extent(0),rank,G.extent(2),G.extent(3));
    btas::contract(1.0,G,{p,q,s,t}, transpose_conj(f2),{r,q},1.0,prst,{p,r,s,t});


    return 0;
}