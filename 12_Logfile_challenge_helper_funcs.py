# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:47:08 2020

@author: holge
"""

def parse_logfile_string(s):
    # split the input string on "\n" new line
    lines = s.split("\n")

    # create a look-up table of sections and line numbers
    idxs = dict()
    for lineNo, line in enumerate(lines):
        if line in ['measurements', "header"]:
            idxs[line] = lineNo 
    idxs["names"] = idxs["measurements"] + 1
    idxs["params_begin"] = idxs["header"] + 1
    idxs["params_end"] = idxs["measurements"] - 1
    idxs["data"] = idxs["names"] + 1

    # parse the column 
    names = lines[idxs["names"]].split(",")

    # parse the params_lines list(str) into params dict{param: value}
    params = dict()
    for line in lines[idxs["params_begin"] : idxs["params_end"]]:
        key, value = line.split(",")
        params[key] = value

    # converts str to float incl. "Ohms" removal
    def string_to_float(s):
        idx = s.find("Ohms")
        if idx > 0:
            number = s.split(" ")[0]
            prefix = s[idx-1]
            return float(number) * {" ": 1, "m": 0.001}[prefix]
        return float(s)

    # parse data_lines list(str) into data list(list(floats))
    data = list()
    for data_line in lines[idxs["data"] :]:
        row = list()
        for item in data_line.split(","):
            row.append(string_to_float(item))
        data.append(row)

    return {"params": params, "names": names, "data":data}