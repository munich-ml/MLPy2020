# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:32:25 2020

@author: holge
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
   

def create_csv(plot=False):
    np.random.seed(23)
    
    def string(x, make_ohms=False):
        if type(x) in (float, np.float64):
            if(abs(x) < 1):
                return "{:.0f} mOhms".format(1000*x)
            return "{:.2f} Ohms".format(x)
            
        else:
            s = str(x)
        return s.replace(",", "_")
    
    
    # Create header
    cal_factors = [0.55, 1, 1.88]
    
    header = {"measurement date": dt.date(2020, 5, 6),
              "measurement time": dt.time(8, 0, 0)}
    
    for i, cal_factor in enumerate(cal_factors):
        if cal_factor != 1:
            header["calibration factor s{}".format(i)] = cal_factor
    
    s = "MLPy2020 logfile challenge\n"
    
    for key, value in header.items():
        s += "\n" + string(key) + "," + string(value)
    
    # Create content
    s += "\n\nmeasurements"
    s += "\nx,sig0,sig1,sig2"
    
    x = 10 * np.random.rand(30)    # sig x-axis
    signals = [x]

    for i, (sig_period, cal_factor) in enumerate(zip([6, 8, 10], cal_factors)):
        signal = 10 * np.sin(2*np.pi* x / sig_period)
        signals.append(signal / cal_factor)
        if plot:
            plt.plot(x, signal, "x", label="sig"+str(i))
    
    if plot:
        plt.xlabel("x"), plt.ylabel("sig [ohms]")
        plt.grid(), plt.legend()

    # write lines
    for values in zip(*signals):
        newline = "\n"
        for val in values:
            newline += string(val) + ","
        s += newline[:-1]
    
    with open("logfile.csv", "w") as file:
        file.write(s)
    

if __name__ == "__main__":
    create_csv(plot=True)