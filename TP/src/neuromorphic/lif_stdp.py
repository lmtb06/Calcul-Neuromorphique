import math

var_gaussienne=28

def gauss(x,std_dev):
    return math.exp(-x*x/(std_dev*std_dev))

def entree(timing,n_neurone):
    centre = 2+3*((int(timing)//10)%3)
    if n_neurone>centre:
        dist=n_neurone-centre
    else:
        dist=centre-n_neurone
    return gauss(dist,math.sqrt(var_gaussienne))

def W(deltaT,theta):
    dw=-deltaT*math.exp(-deltaT*deltaT)*theta/10
    return dw

