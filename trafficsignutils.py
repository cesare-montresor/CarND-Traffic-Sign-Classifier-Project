#utility functions

import time, datetime, json, os
from collections import OrderedDict 
from json import dumps
from hashlib import sha256
import pickle


def sortOD(od):
    res = collections.OrderedDict()
    for k, v in sorted(od.items()):
        if isinstance(v, dict):
            res[k] = sortOD(v)
        else:
            res[k] = v
    return res


def hashParams(params):
    sortedParams = sortOD(params)
    serializedParams = json.dumps(sortedParams).encode('utf8')
    hashedParams = hashlib.sha256(serializedParams).hexdigest()
    return hashedParams

def saveModelInfo(filename, epoch, hyperparams, train_time, validation_accuracy,time_history=[],accuracy_history=[]):
    data = {
        'hyperparams' : hyperparams,
        'info': {
            'epoch': epoch,
            'train_time': train_time,
            'validation_accuracy': validation_accuracy,
            'history':{
                'time': time_history,
                'accuracy': accuracy_history,
            }
        }
    }
    with open(filename+'.p', mode='wb') as f:
        pickle.dump(data,f)
    with open(filename+'.json', mode='wb') as f:
        d = json.dumps(data).encode('utf8')
        f.write(d)

def loadModelInfo(basedir,restore_name):
    info_name_load = basedir + restore_name +'/info-' + restore_name+'.p'
    if not os.path.exists(info_name_load):
        print('load param failed, file not found'+info_name_load)
        return None
    else:
        with open(info_name_load,mode='rb') as f:
            #print('Loading Params')
            data = pickle.load(f)
        return  data
    return None
    
        
def formatLine(texts,col_size, default_size=7):
    col_num = len(texts)
    line = "|"
    for i in range(col_num):
        pad = ''
        size = col_size[i] if i < len(col_size) else default_size
        text = str(texts[i])
        spaces = max(size - len(text),0)
        if spaces >= 2:
            pad = ' '
            spaces -= 2
        line+= pad + text + pad + (" " * spaces) + "|"
    return line
