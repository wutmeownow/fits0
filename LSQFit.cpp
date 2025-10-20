#include "TRandom2.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TGClient.h"
#include "TStyle.h"
#include "TText.h"


#include <iostream>
using namespace std;

using TMath::Log;

//parms
const double xmin=1;
const double xmax=20;
const int npoints=12;
const double sigma=0.2;
const int npar=3;
const int n_trial=100000;

double f(double x){
  const double a=0.5;
  const double b=1.3;
  const double c=0.5;
  return a+b*Log(x)+c*Log(x)*Log(x);
}

void getX(double *x){
  double step=(xmax-xmin)/npoints;
  for (int i=0; i<npoints; i++){
    x[i]=xmin+i*step;
  }
}

void getY(const double *x, double *y, double *ey){
  static TRandom2 tr(0);
  for (int i=0; i<npoints; i++){
    y[i]=f(x[i])+tr.Gaus(0,sigma);
    ey[i]=sigma;
  }
}


void leastsq(){
  double x[npoints];
  double y[npoints];
  double ey[npoints];
  getX(x);
  getY(x,y,ey);
  auto tg = new TGraphErrors(npoints,x,y,0,ey);
  tg->Draw("alp");
}

TMatrixD SolveLSQ(const TMatrixD &A, const TMatrixD &y){
  TMatrixD AT=(A);  
  AT.T();            // A transpose
  TMatrixD ATAi(AT,TMatrixD::kMult,A);
  ATAi.Invert();
  TMatrixD Adag(ATAi,TMatrixD::kMult,AT);  // (A^T A)^(-1) A^T
  TMatrixD theta(Adag,TMatrixD::kMult,y);  // (A^T A)^(-1) A^T * y
  return theta;
}

#include <vector>
#include <cmath>

// Compute chi-squared = Σ [ (y_i - f(x_i)) / σ_i ]²
double ChiSquared(const TVectorD& x,const TVectorD& y,const TVectorD& yerr,const TMatrixD& params){
  TF1 *fn1 = new TF1("fn1","[0]+[1]*log(x) + [2]*log(x)*log(x)",xmin,xmax);
  fn1->SetParameters(params[0][0], params[1][0], params[2][0]);
  double chi2 = 0.0;
  for (Int_t i = 0; i < x.GetNrows(); ++i) {
      double y_model = fn1->Eval(x[i]);
      double diff = (y[i] - y_model) / yerr[i];
      chi2 += diff * diff;
  }
  return chi2;
}


int main(int argc, char **argv){
  TApplication theApp("App", &argc, argv); // init ROOT App for displays

  // ******************************************************************************
  // ** this block is useful for supporting both high and std resolution screens **
  UInt_t dh = gClient->GetDisplayHeight()/2;   // fix plot to 1/2 screen height  
  //UInt_t dw = gClient->GetDisplayWidth();
  UInt_t dw = 1.1*dh;
  // ******************************************************************************

  // gStyle->SetOptStat(0); // turn off histogram stats box

  double lx[npoints];
  double ly[npoints];
  double ley[npoints];

  // *** modify and add your code here ***

  TVectorD par_a(n_trial);
  TVectorD par_b(n_trial);
  TVectorD par_c(n_trial);
  TVectorD chi2(n_trial);
  TVectorD chi2_red(n_trial);

  // perform many least squares fits on different pseudo experiments here
  for (int i=0;i<n_trial;i++) {
    getX(lx);
    getY(lx,ly,ley);

    // Define vectors and matrices.<br>
    // Make the vectors 'use' the data : they are not copied, 
    // the vector data pointer is just set appropriately
    TVectorD x; x.Use(npoints,lx);
    TVectorD y; y.Use(npoints,ly);
    TVectorD e; e.Use(npoints,ley);
    TVectorD logx(npoints);
    TVectorD logxlogx(npoints);

    for (int j=0; j<logx.GetNrows(); ++j) {
      logx[j] = std::log(x[j]);
      logxlogx[j] = logx[j]*logx[j];
    }


    TMatrixD A(npoints,npar);       // A matrix

    // Fill the A matrix
    TMatrixDColumn(A,0) = 1.0;
    TMatrixDColumn(A,1) = logx;
    TMatrixDColumn(A,2) = logxlogx;
    // cout << "A = ";
    // A.Print();

    // apply weights
    TMatrixD yw(A.GetNrows(),1);   // error-weighted data values 

    for (Int_t irow = 0; irow < A.GetNrows(); irow++) {
      TMatrixDRow(A,irow) *= 1/e(irow);
      TMatrixDRow(yw,irow) += y(irow)/e(irow);
    }
    // cout << "A weighted = ";
    // A.Print();
    // cout << "y weighted = ";
    // yw.Print();

    TMatrixD theta=SolveLSQ(A, yw);
    // cout << "Param vector = ";
    // theta.Print();

    // add parameters to histograms
    par_a[i] = theta[0][0];
    par_b[i] = theta[1][0];
    par_c[i] = theta[2][0];
    chi2[i] = ChiSquared(x, y, e, theta);
    chi2_red[i] = chi2[i]/(npoints-npar);

  }

  TH2F *h1 = new TH2F("h1","Parameter b vs a;a;b",100,par_a.Min(),par_a.Max(),100,par_b.Min(),par_b.Max());
  TH2F *h2 = new TH2F("h2","Parameter c vs a;a;c",100,par_a.Min(),par_a.Max(),100,par_c.Min(),par_c.Max());
  TH2F *h3 = new TH2F("h3","Parameter c vs b;b;c",100,par_b.Min(),par_b.Max(),100,par_c.Min(),par_c.Max());
  TH1F *h4 = new TH1F("h4","reduced chi^2;;frequency",100,chi2_red.Min(),chi2_red.Max());

  TH1F *h5 = new TH1F("h5","Parameter a;a;frequency",100,par_a.Min(),par_a.Max());
  TH1F *h6 = new TH1F("h6","Parameter b;b;frequency",100,par_b.Min(),par_b.Max());
  TH1F *h7 = new TH1F("h7","Parameter c;c;frequency",100,par_c.Min(),par_c.Max());
  TH1F *h8 = new TH1F("h8","chi^2;;frequency",100,chi2.Min(),chi2.Max());
  

  // fill histograms w/ required data
  for (int i=0; i<n_trial;++i) {
    h1->Fill(par_a[i], par_b[i]);
    h2->Fill(par_a[i], par_c[i]);
    h3->Fill(par_b[i], par_c[i]);
    h4->Fill(chi2_red[i]);

    h5->Fill(par_a[i]);
    h6->Fill(par_b[i]);
    h7->Fill(par_c[i]);
    h8->Fill(chi2[i]);
  }

  TText text;
  text.SetTextSize(0.03);
  
  TCanvas *tc1 = new TCanvas("c1","my study results 1",800,600);
  tc1->Divide(2,2);
  h5->SetStats(kTRUE);
  tc1->cd(1); h5->Draw();
  text.DrawTextNDC(0.15, 0.6, "Larger N -> mean goes to ");
  text.DrawTextNDC(0.15, 0.55, "the actual parameter value (ex: a=0.5)");
  h6->SetStats(kTRUE);
  tc1->cd(2); h6->Draw();
  h7->SetStats(kTRUE);
  tc1->cd(3); h7->Draw();
  h8->SetStats(kTRUE);
  tc1->cd(4); h8->Draw();

  text.DrawTextNDC(0.5, 0.6, "Mean approaches (N DoF)");
  text.DrawTextNDC(0.5, 0.5, "The Std Dev approaches (N DoF)*sigma");

  tc1->Update();
  tc1->SaveAs("LSQFit.pdf(");

  TCanvas *tc2 = new TCanvas("c2","my study results 2",800,600);
  tc2->Divide(2,2);
  h1->SetStats(kFALSE);
  tc2->cd(1); h1->Draw("colz");
  h2->SetStats(kFALSE);
  tc2->cd(2); h2->Draw("colz");
  h3->SetStats(kFALSE);
  tc2->cd(3); h3->Draw("colz");
  h4->SetStats(kTRUE);
  tc2->cd(4); h4->Draw();
  
  text.DrawTextNDC(0.5, 0.6, "Mean approaches 1.0");
  text.DrawTextNDC(0.5, 0.5, "The Std Dev approaches sigma");

  tc2->Update();
  tc2->SaveAs("LSQFit.pdf)");

  // **************************************
  
  cout << "Press ^c to exit" << endl;
  theApp.SetIdleTimer(30,".q");  // set up a failsafe timer to end the program  
  theApp.Run();
}
