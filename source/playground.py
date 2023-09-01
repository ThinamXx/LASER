from numpy import dot
from numpy.linalg import norm
import numpy as np

eng_emb_path = "/root/nllb/LASER/tasks/embed/out_eng_latn.bin"
hi_emb_path = "/root/nllb/LASER/tasks/embed/out_hin_Deva.bin"
dim = 1024
eng_embs = np.fromfile(eng_emb_path, dtype=np.float32, count=-1)
hi_embs = np.fromfile(eng_emb_path, dtype=np.float32, count=-1)
eng_embs.resize(eng_embs.shape[0] // dim, dim)    
hi_embs.resize(hi_embs.shape[0] // dim, dim)
print(hi_embs.shape)

hi_emb = hi_embs[0]

for emb in eng_embs:
    cos_sim = dot(emb, hi_emb)/(norm(emb)*norm(hi_emb))
    print(cos_sim)