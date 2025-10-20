import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import ROOT as r

xmin=1.0
xmax=20.0
npoints=40
sigma=0.5
lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)
pars=[0.5,1.3,0.5]


def f(x,par):
    return par[0]+par[1]*np.log(x)+par[2]*np.log(x)*np.log(x)

from random import gauss
def getX(x):  # x = array-like
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
def getY(x,y,ey):  # x,y,ey = array-like
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

def chi_squared(x, y, y_err, model_func, params):
    y_model = model_func(x, params)
    residuals = (y - y_model) / y_err
    chi2 = np.sum(residuals**2)
    return chi2

def reduced_chi_squared(x, y, y_err, model_func, params):
    """
    Calculate the reduced chi-squared of a fit.

    Parameters
    ----------
    x : array_like
        Independent variable data.
    y : array_like
        Observed dependent variable data.
    y_err : array_like or float
        Uncertainties in y data. Can be a single number or an array.
    model_func : callable
        Model function f(x, *params) used for fitting.
    params : list or tuple
        Best-fit parameters [A, B, C, ...].
    
    Returns
    -------
    float
        Reduced chi-squared value.
    """
    dof = len(y) - len(params)
    return chi_squared(x,y,y_err,model_func,params) / dof


# *** modify and add your code here ***
n_trials = 10000
n_par = len(pars)

a = np.zeros(n_trials)
b = np.zeros(n_trials)
c = np.zeros(n_trials)
chi2 = np.zeros(n_trials)

for i in range(n_trials):
    # print(i)
    # get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below
    getX(lx)
    getY(lx,ly,ley)


    A=np.matrix(np.zeros((npoints, n_par)))
    #Fill the A matrix
    for nr in range(npoints):
        for nc in range(n_par):
            if nc==0: A[nr,nc] = 1.
            elif nc==1: A[nr,nc] = np.log(lx[nr])
            else: A[nr,nc] = np.log(lx[nr])*np.log(lx[nr])


    # apply weights from error to measurements y and matrix A
    for j in range(npoints):
        A[j] = A[j] / ley[j]
    yw = (ly/ley).reshape(npoints,1)

    # solve for the parameters
    theta =  inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(yw)
    # print(theta)

    # add the parameters to the vectors
    a[i] = theta[0,0]
    b[i] = theta[1,0]
    c[i] = theta[2,0]
    chi2[i] = chi_squared(lx, ly, ley, f, [a[i],b[i],c[i]])

reduced_chi2 = chi2/(len(ly)-n_par)


# perform many least squares fits on different pseudo experiments here
# fill histograms w/ required data

n_bins = 40

# Create a canvas for both pages
canvas = r.TCanvas("c", "Parameters", 800, 600)
canvas.Divide(2,2)

canvas2 = r.TCanvas("c2", "Parameters", 800, 600)
canvas2.Divide(2,2)

# Create four histograms for each parameter and the chi_squared
ha = r.TH1F("ha", "Parameter a;a;Counts", n_bins, a.min(), a.max())
ha.SetDirectory(0)
hb = r.TH1F("hb", "Parameter b;b;Counts", n_bins, b.min(), b.max())
hb.SetDirectory(0)
hc = r.TH1F("hc", "Parameter c;c;Counts", n_bins, c.min(), c.max())
hc.SetDirectory(0)
hchi2 = r.TH1F("hchi2", "Chi Squared;Chi Squared;Counts", n_bins, chi2.min(), chi2.max())
hchi2.SetDirectory(0)

# Create four histograms for ratios of parameters and the reduced chi_squared
hba = r.TH2F("hba", "Parameter b vs a;a;b", n_bins, a.min(), a.max(), n_bins, b.min(), b.max())
hba.SetDirectory(0)
hca = r.TH2F("hca", "Parameter c vs a;a;c", n_bins, a.min(), a.max(), n_bins, c.min(), c.max())
hca.SetDirectory(0)
hcb = r.TH2F("hcb", "Parameter c vs b;b;c", n_bins, b.min(), b.max(), n_bins, c.min(), c.max())
hcb.SetDirectory(0)
hchi2_red = r.TH1F("hchi2_red", "Reduced Chi Squared;Reduced Chi Squared;Counts", n_bins, reduced_chi2.min(), reduced_chi2.max())
hchi2_red.SetDirectory(0)

# Fill the histograms
for i in range(n_trials):
    ha.Fill(a[i])
    hb.Fill(b[i])
    hc.Fill(c[i])
    hchi2.Fill(chi2[i])

    hba.Fill(a[i], b[i])
    hca.Fill(a[i], c[i])
    hcb.Fill(b[i], c[i])
    hchi2_red.Fill(reduced_chi2[i])


# Draw each histogram in its own pad
canvas.cd(1)
ha.Draw()

canvas.cd(2)
hb.Draw()

canvas.cd(3)
hc.Draw()

canvas.cd(4)
hchi2.Draw()

# Update and save
canvas.Update()
canvas.SaveAs("LSQFit.pdf(") # save this canvas in the first page of the pdf


# Draw each histogram in its own pad
canvas2.cd(1)
hba.Draw("COLZ")

canvas2.cd(2)
hca.Draw("COLZ")

canvas2.cd(3)
hcb.Draw("COLZ")

canvas2.cd(4)
hchi2_red.Draw()

canvas2.Update()
canvas2.SaveAs("LSQFit.pdf)") # save this canvas to the next page and close the pdf

# **************************************

# Clean up
del ha
del hb
del hc
del hchi2
del hba
del hca
del hcb
del hchi2_red

canvas.Close()
canvas2.Close()
del canvas
del canvas2



input("hit Enter to exit")
