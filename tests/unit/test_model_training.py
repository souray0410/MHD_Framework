import json
import numpy as np
import pytest
from mhd_framework.models.artifacts import file_sha256
from mhd_framework.models.training import ArrayDataset, epoch_indices, macro_f1

def test_sharding_is_complete_disjoint_and_epoch_specific():
    a=epoch_indices(1264,0,2,3416,1);b=epoch_indices(1264,1,2,3416,1)
    assert set(a).isdisjoint(b) and sorted(a+b)==list(range(1264))
    assert a!=epoch_indices(1264,0,2,3416,2)
    assert a==epoch_indices(1264,0,2,3416,1)
    with pytest.raises(ValueError):epoch_indices(3,0,2,3416,1)

def test_array_manifest_hash_role_and_tensor_contract(tmp_path):
    f=tmp_path/'input.npy';np.save(f,np.full((2,3,4,4),255,dtype=np.uint8))
    m={'role':'train','samples':[{'id':'s1','label':1,'file':f.name,'sha256':file_sha256(f)}]}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    pre={'shape':[2,3,4,4],'mean':[.5]*3,'std':[.25]*3}
    d=ArrayDataset(p,pre,file_sha256(p));x,y,i=d[0]
    assert x.shape==(2,3,4,4) and (x==2).all() and (y,i)==(1,0)
    np.save(f,np.zeros((2,3,4,4),dtype=np.uint8))
    with pytest.raises(ValueError,match='array changed'):ArrayDataset(p,pre,file_sha256(p))
    m['role']='test';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='train/development'):ArrayDataset(p,pre,file_sha256(p))

def test_macro_f1_counts_absent_classes_as_zero():
    assert macro_f1(np.array([0,1]),np.array([[.8,.2],[.3,.7]]))==1
    assert macro_f1(np.array([0,0]),np.array([[.8,.2],[.8,.2]]))==.5
