"""
last edit on The Nov  3 19:12:31 2022

@author: sepidehbabaei
"""
"""
change the directory to the scima folder
cd Desktop/CrC/scima

Call the function from the commandline like this:

python3 spatialinput.py --nn 200 --cnt 13
"""

import numpy as np
from utils import  mkdir_p
import pickle
from multiprocessing import Pool, cpu_count



# ============================================================
# MULTIPROCESS PER IMAGE
# ============================================================
def _process_single_image(args):
    Anchor, image, pat, intensity, ct, x, y, cellid, K, img_id = args
    np.random.seed(12345)

    Totintens = np.empty((0, intensity.shape[1]))
    ranTotintens = np.empty((0, intensity.shape[1]))
    Totobj, ranTotobj = [], []
    Totcellids, ranTotcellids = [], []
    pid, nset = [], []

    inx = np.where(image == img_id)[0]
    ipat = np.unique(pat[inx])
    px, py = x[inx], y[inx]
    pint, pct = intensity[inx], ct[inx]
    pcellid = cellid[inx]

    inx_anchor = np.where(pct == Anchor)[0]
    raninx = np.random.choice(
        np.setdiff1d(np.arange(len(pct)), inx_anchor),
        size=len(inx_anchor),
        replace=False
    )

    nset.append(len(inx_anchor))

    for ianchor in inx_anchor:
        Totintens, Totobj, Totcellids = make_KNN_anchor(
            ianchor, px, py, K, pint, pcellid,
            Totintens, Totobj, Totcellids
        )
        pid += np.repeat(ipat, K).tolist()

    for ianchor in raninx:
        ranTotintens, ranTotobj, ranTotcellids = make_KNN_anchor(
            ianchor, px, py, K, pint, pcellid,
            ranTotintens, ranTotobj, ranTotcellids
        )

    return Totintens, Totobj, Totcellids, ranTotintens, ranTotobj, ranTotcellids, pid, nset


# def _process_single_image(args):
#     # unpack arguments
#     Anchor, image, pat, intensity, ct, x, y, cellid, K, img_id = args

#     np.random.seed(12345)  # per-process reproducibility

#     Totintens = np.empty((0, intensity.shape[1]))
#     ranTotintens = np.empty((0, intensity.shape[1]))
#     ranTotobj, Totobj = [], []
#     pid = []
#     nset = []

#     # indices for this image
#     inx = np.where(image == img_id)[0]
#     ipat = np.unique(pat[inx])
#     px = x[inx]
#     py = y[inx]
#     pcellid = cellid[inx]
#     pint = intensity[inx]
#     pct = ct[inx]

#     inx_anchor = np.where(pct == Anchor)[0]
#     icell = np.arange(len(pct))
#     raninx = np.delete(icell, inx_anchor, 0)
#     raninx = np.random.choice(raninx, len(inx_anchor), replace=False)

#     nset.append(len(inx_anchor))

#     for ianchor in inx_anchor:
#         Totintens, Totobj = make_KNN_anchor(
#             ianchor, px, py, K, pint, pcellid, Totintens, Totobj
#         )
#         pid = pid + np.repeat(ipat, K).tolist()

#     for ianchor in raninx:
#         ranTotintens, ranTotobj = make_KNN_anchor(
#             ianchor, px, py, K, pint, pcellid, ranTotintens, ranTotobj
#         )

#     # return per-image results
#     return Totintens, Totobj, ranTotintens, ranTotobj, pid, nset


# def spatial_input_per_sample_mp(Anchor, image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR):
#     image0 = np.unique(image)

#     # build argument list for each image
#     args_list = [
#         (Anchor, image, pat, intensity, ct, x, y, cellid, K, img_id)
#         for img_id in image0
#     ]

#     # use up to 4 CPUs
#     nproc = min(4, cpu_count())
#     with Pool(processes=nproc) as pool:
#         results = pool.map(_process_single_image, args_list)

#     # aggregate results from all images
#     Totintens = np.empty((0, intensity.shape[1]))
#     ranTotintens = np.empty((0, intensity.shape[1]))
#     Totobj_all, ranTotobj_all = [], []
#     pid_all = []
#     nset_all = []

#     for (Ti, To, rTi, rTo, pid, nset) in results:
#         if Ti.size:
#             Totintens = np.vstack([Totintens, Ti])
#         if rTi.size:
#             ranTotintens = np.vstack([ranTotintens, rTi])
#         Totobj_all.extend(To)
#         ranTotobj_all.extend(rTo)
#         pid_all.extend(pid)
#         nset_all.extend(nset)

#     d = Totintens
#     s = [pid_all, Totobj_all]

#     rd = ranTotintens
#     rs = [pid_all, ranTotobj_all]

#     mkdir_p(OUTDIR)
#     lab = str(label)

#     pickle.dump(d, open(OUTDIR + "/d" + lab + ".p", "wb"))
#     pickle.dump(rd, open(OUTDIR + "/drand" + lab + ".p", "wb"))
#     pickle.dump(s, open(OUTDIR + "/s" + lab + ".p", "wb"))
#     pickle.dump(rs, open(OUTDIR + "/srand" + lab + ".p", "wb"))
#     pickle.dump(nset_all, open(OUTDIR + "/nset" + lab + ".p", "wb"))
#     pickle.dump(nset_all, open(OUTDIR + "/nsetrand" + lab + ".p", "wb"))

#     return d, s, rd, rs, nset_all


# ============================================================
# SPATIAL INPUT (MP)
# ============================================================
def spatial_input_per_sample_mp(Anchor, image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR):
    args_list = [
        (Anchor, image, pat, intensity, ct, x, y, cellid, K, img)
        for img in np.unique(image)
    ]

    with Pool(min(4, cpu_count())) as pool:
        results = pool.map(_process_single_image, args_list)

    Totintens = np.empty((0, intensity.shape[1]))
    ranTotintens = np.empty((0, intensity.shape[1]))
    Totobj, ranTotobj = [], []
    Totcellids, ranTotcellids = [], []
    pid, nset = [], []

    for Ti, To, Tc, rTi, rTo, rTc, p, ns in results:
        if Ti.size:
            Totintens = np.vstack([Totintens, Ti])
        if rTi.size:
            ranTotintens = np.vstack([ranTotintens, rTi])
        Totobj.extend(To)
        ranTotobj.extend(rTo)
        Totcellids.extend(Tc)
        ranTotcellids.extend(rTc)
        pid.extend(p)
        nset.extend(ns)

    mkdir_p(OUTDIR)
    lab = str(label)

    pickle.dump(Totintens, open(f"{OUTDIR}/d{lab}.p", "wb"))
    pickle.dump(ranTotintens, open(f"{OUTDIR}/drand{lab}.p", "wb"))
    pickle.dump([pid, Totobj], open(f"{OUTDIR}/s{lab}.p", "wb"))
    pickle.dump([pid, ranTotobj], open(f"{OUTDIR}/srand{lab}.p", "wb"))
    pickle.dump(Totcellids, open(f"{OUTDIR}/cellids{lab}.p", "wb"))
    pickle.dump(ranTotcellids, open(f"{OUTDIR}/cellidsrand{lab}.p", "wb"))
    pickle.dump(nset, open(f"{OUTDIR}/nset{lab}.p", "wb"))

    return

# def spatial_input_per_sample(Anchor,image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR):
    
#     #pat0 = np.unique(pat)
#     image0 = np.unique(image)
#     np.random.seed(12345)

#     Totintens = np.empty((0, intensity.shape[1]))
#     ranTotintens = np.empty((0, intensity.shape[1]))
#     ranTotobj , Totobj =[] , []
#     pid = []
#     nset = []

#     for i in image0: #pat0:       
#         #inx = np.where(pat == i)[0]
#         inx = np.where(image == i)[0]
#         ipat= np.unique(pat[inx])
#         #pdata = subdata[inx,:]
#         px = x[inx]  
#         py = y[inx]  
#         pcellid = cellid[inx]  
#         pint = intensity[inx]
#         pct = ct[inx]

#         inx = np.where(pct == Anchor)[0]
#         #rpdata = np.delete(pdata, inx, 0)
#         icell = np.arange(len(pct))
#         raninx = np.delete(icell, inx, 0)
#         raninx = np.random.choice(raninx, len(inx), replace= False)
               
#         nset.append(len(inx))
#         #pid = []
#         for ianchor in inx:
#             Totintens, Totobj = make_KNN_anchor(ianchor,px, py, K, pint, pcellid, Totintens, Totobj)
#             pid = pid + np.repeat(ipat,K).tolist()
    
#         for ianchor in raninx:
#             ranTotintens, ranTotobj = make_KNN_anchor(ianchor,px, py, K, pint, pcellid, ranTotintens, ranTotobj)


#     d = Totintens
#     s = [pid, Totobj]

#     rd = ranTotintens
#     rs = [pid, ranTotobj]

#     mkdir_p(OUTDIR)

#     lab = str(label)
#     pickle.dump(d, open(OUTDIR+'/d'+ lab +'.p','wb'))
#     pickle.dump(rd, open(OUTDIR+'/drand'+ lab +'.p','wb'))

#     pickle.dump(s, open(OUTDIR+'/s'+ lab +'.p','wb'))
#     pickle.dump(rs, open(OUTDIR+'/srand'+ lab +'.p','wb'))

#     pickle.dump(nset, open(OUTDIR+'/nset'+ lab +'.p','wb'))
#     pickle.dump(nset, open(OUTDIR+'/nsetrand'+ lab +'.p','wb'))

#     return d,s,rd,rs, nset


def spatial_input_per_sample(
    Anchor, image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR
):
    image0 = np.unique(image)
    np.random.seed(12345)

    Totintens = np.empty((0, intensity.shape[1]))
    ranTotintens = np.empty((0, intensity.shape[1]))

    Totobj, ranTotobj = [], []
    Totcellids, ranTotcellids = [], []   # ✅ NEW

    pid = []
    nset = []

    for img in image0:
        inx = np.where(image == img)[0]
        ipat = np.unique(pat[inx])

        px = x[inx]
        py = y[inx]
        pcellid = cellid[inx]
        pint = intensity[inx]
        pct = ct[inx]

        inx_anchor = np.where(pct == Anchor)[0]
        icell = np.arange(len(pct))
        raninx = np.setdiff1d(icell, inx_anchor)
        raninx = np.random.choice(raninx, len(inx_anchor), replace=False)

        nset.append(len(inx_anchor))

        # ---------- TRUE ANCHORS ----------
        for ianchor in inx_anchor:
            dist = np.sqrt((px - px[ianchor])**2 + (py - py[ianchor])**2)
            nnix = np.argsort(dist)[1:K+1]

            Totintens = np.concatenate((Totintens, pint[nnix]), axis=0)

            obj = pcellid[nnix].tolist()
            Totobj.extend(obj)
            Totcellids.append(np.array(obj))   # ✅ KEY LINE

            pid.extend(np.repeat(ipat, K).tolist())

        # ---------- RANDOM ANCHORS ----------
        for ianchor in raninx:
            dist = np.sqrt((px - px[ianchor])**2 + (py - py[ianchor])**2)
            nnix = np.argsort(dist)[1:K+1]

            ranTotintens = np.concatenate((ranTotintens, pint[nnix]), axis=0)

            obj = pcellid[nnix].tolist()
            ranTotobj.extend(obj)
            ranTotcellids.append(np.array(obj))

    mkdir_p(OUTDIR)
    lab = str(label)

    # ---------- ORIGINAL FILES ----------
    pickle.dump(Totintens, open(f"{OUTDIR}/d{lab}.p", "wb"))
    pickle.dump(ranTotintens, open(f"{OUTDIR}/drand{lab}.p", "wb"))

    pickle.dump([pid, Totobj], open(f"{OUTDIR}/s{lab}.p", "wb"))
    pickle.dump([pid, ranTotobj], open(f"{OUTDIR}/srand{lab}.p", "wb"))

    pickle.dump(nset, open(f"{OUTDIR}/nset{lab}.p", "wb"))
    pickle.dump(nset, open(f"{OUTDIR}/nsetrand{lab}.p", "wb"))

    # ---------- NEW (CRITICAL FOR SALIENCY) ----------
    pickle.dump(Totcellids, open(f"{OUTDIR}/cellids{lab}.p", "wb"))
    pickle.dump(ranTotcellids, open(f"{OUTDIR}/cellidsrand{lab}.p", "wb"))

    return



# def make_KNN_anchor(ianchor,px, py, K, pint, pcellid, Totintens, Totobj):
#     dist= np.array(((px-px[ianchor])**2) + ((py -py[ianchor])**2),dtype=np.float32)
#     dist = np.sqrt(dist)
#     nnix=np.argsort(dist)[1:K+1]
#     intens = pint[nnix,:]
#     Totintens = np.concatenate((Totintens, intens), axis=0)
#     obj = pcellid[nnix].tolist()
#     Totobj = Totobj+obj
    
#     return Totintens, Totobj


# ============================================================
# KNN construction (WITH CELL ID TRACKING)
# ============================================================
def make_KNN_anchor(
    ianchor, px, py, K,
    pint, pcellid,
    Totintens, Totobj,
    Totcellids=None   # ✅ OPTIONAL
):
    dist = np.sqrt((px - px[ianchor])**2 + (py - py[ianchor])**2)
    nnix = np.argsort(dist)[1:K+1]

    # embeddings
    Totintens = np.concatenate((Totintens, pint[nnix]), axis=0)

    # global cell IDs
    obj = pcellid[nnix].tolist()
    Totobj.extend(obj)

    # ✅ only track if requested
    if Totcellids is not None:
        Totcellids.append(np.array(obj))

    # backward-compatible return
    if Totcellids is not None:
        return Totintens, Totobj, Totcellids
    else:
        return Totintens, Totobj
    
    

def BG_spatial_input_per_sample(N, image, pat, intensity, ct, x, y, cellid, K, label, OUTDIR):
    
    #pat = pat[inx]
    #intensity = Intensity[inx,:]
    #ct = ct[inx]
    #x =x[inx]
    #y= y[inx]
    #cellid = cellid[inx]
    #label = labels[i]
    #image = image[inx]
    
    #pat0 = np.unique(pat)
    image0 = np.unique(image)
    np.random.seed(12345)

    ranTotintens = np.empty((0, intensity.shape[1]))
    ranTotobj =[]
    pid = []
    nset = []

    for i in image0: #pat0:       
        #inx = np.where(pat == i)[0]
        inx = np.where(image == i)[0]
        ipat= np.unique(pat[inx])

        px = x[inx]  
        py = y[inx]  
        pcellid = cellid[inx]  
        pint = intensity[inx]
        pct = ct[inx]

        raninx = np.random.choice(len(pct), N, replace= False)
               
        nset.append(len(raninx))
        #pid = []
        for ianchor in raninx:
            ranTotintens, ranTotobj = make_KNN_anchor(ianchor,px, py, K, pint, pcellid, ranTotintens, ranTotobj)
            pid = pid + np.repeat(ipat,K).tolist()


    rd = ranTotintens
    rs = [pid, ranTotobj]

    mkdir_p(OUTDIR)

    lab = str(label)
    pickle.dump(rd, open(OUTDIR+'/d'+ lab +'.p','wb'))

    pickle.dump(rs, open(OUTDIR+'/s'+ lab +'.p','wb'))

    pickle.dump(nset, open(OUTDIR+'/nset'+ lab +'.p','wb'))
    

    return rd,rs, nset


def make_inputset(Sample_ID, Cell_ID, IDx, Dx, K, k ):
    
    #Cell_ID = cell_id[lab]
    #IDx = idx
    #Dx = d[lab]
    #Sample_ID = sample_id[lab]
    
    group_list = []
    cell =[]
    
    inx = np.where(Sample_ID == IDx)[0]
    x= Dx[inx,]
    xcell = Cell_ID[inx]
    nset = int(len(inx)/K)
    for cnt in range(1, nset+1):
        start= (cnt-1) * K
        end = start + k #(cnt*nn)
        xset =x[start:end,]
        xcell_set=xcell[start:end,]
        group_list.append(xset)
        cell.append(xcell_set)
    pat_id = ([IDx]*nset)  
    return group_list,cell, pat_id, nset

import os, sys, errno
from datetime import datetime
from model import CellCnn


def run_scima(Anchor, ntrain_per_class, K, k, nset_thr, labels, classes, path, nrun, background):
    
    ncell= k
    ntime_points=1
    nsubsets=1
    per_sample = True
    #transform = False
    scale= True
    qt_norm= True
    now = datetime.now()
    today = datetime.today()
    result_file = 'results'+str(Anchor)+ f'_K{K}'
    
    current_time = now.strftime("%H_%M_%S")
    current_day = today.strftime("%d_%m_%Y")
    old_stdout = sys.stdout
    
    INDIR = path + '/Anchor' + str(Anchor) + f'_K{K}'
    
    try:
        log_dir = path+'/'+result_file+'/run_log/'
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    log_file = open(log_dir+"/data_run_log_"+str(current_day)+'_'+str(current_time)+'.txt',"w+")
    print('output will be redirected to',log_dir+"/data_run_log_"+str(current_day)+'_'+str(current_time)+'.txt')
    sys.stdout = log_file
    
    OUTDIR = os.path.join(path, result_file, 'out_'+str(current_day)+'_'+str(current_time))
    mkdir_p(OUTDIR)
    print('output directory', OUTDIR)
    
         
          
    d = dict()
    s = dict()
    nset = dict()
    
    for i in range(0, len(labels)):
        lab = str(labels[i])
        d[lab] = pickle.load(open(INDIR+'/d'+ lab +'.p','rb'))
        s[lab] = pickle.load(open(INDIR+'/s'+ lab +'.p','rb'))
        nset[lab] = pickle.load(open(INDIR+'/nset'+ lab +'.p','rb'))
    
    sample_id = dict()
    cell_id = dict()
    group = dict()
    for i in range(0, len(labels)):
        lab = str(labels[i])
        sample_id[lab] = np.asarray(s[lab][0])
        cell_id[lab] = np.asarray(s[lab][1])
        group[lab] = np.unique(np.asarray(s[lab][0]))
    
    
    
    np.random.seed(12345)
    
    train_idx = dict()
    test_idx = dict()
        
    def get_patient(sample):
        return sample   # no change, use entire sample name as "patient"
    
    patient_ids = {}

    for lab in labels:
        patient_ids[lab] = np.array([get_patient(s) for s in sample_id[lab]])
        
    all_patients = np.unique(np.concatenate([patient_ids[lab] for lab in labels]))
    
    np.random.seed(12345)
    np.random.shuffle(all_patients)

    n_train_patients = max(1, ntrain_per_class)   # or use ratio
    train_patients = all_patients[:n_train_patients]
    test_patients  = all_patients[n_train_patients:]

    print("TRAIN PATIENTS:", train_patients)
    print("TEST PATIENTS:", test_patients)
    
    
    train_idx = {}
    test_idx  = {}

    for lab in labels:
        train_idx[lab] = [s for s in group[lab] if get_patient(s) in train_patients]
        test_idx[lab]  = [s for s in group[lab] if get_patient(s) in test_patients]
    
        
    # if background:
    #     for i in range(0, int(len(labels)/2)):
    #         lab = str(labels[i])
    #         train_idx[lab] = list(np.random.choice(group[lab], size=ntrain_per_class, replace=False))
    #         test_idx[lab] = [j for j in group[lab] if j not in train_idx[lab]]
    #         rlab = str(labels[i+ int(len(labels)/2)])
    #         train_idx[rlab] = train_idx[lab]
    #         test_idx[rlab] = test_idx[lab]
    # else:
    #     for i in range(0, len(labels)):
    #         lab = str(labels[i])
    #         train_idx[lab] = list(np.random.choice(group[lab], size=ntrain_per_class, replace=False))
    #         test_idx[lab] = [j for j in group[lab] if j not in train_idx[lab]]

    
    
    train_group_list = dict()
    train_pat_id= dict()
    train_cell = dict()
    train_nset = dict()
        
    for i in range(0, len(labels)):
        lab = str(labels[i])
        group_list = []
        pat_id=[]
        cell =[]
        nset =[]
    
        for idx in train_idx[lab]:
            group_list0,cell0, pat_id0, nset0 = make_inputset(sample_id[lab], cell_id[lab], idx, d[lab], K, k)
            group_list += group_list0
            cell += cell0
            pat_id += pat_id0  
            nset.append(nset0)
            train_group_list[lab] = group_list
            train_pat_id[lab] = pat_id
            train_cell[lab] = cell
            train_nset[lab] = nset
    
    test_group_list = dict()
    test_pat_id= dict()
    test_cell = dict()
    test_nset = dict()
        
    for i in range(0, len(labels)):
        lab = str(labels[i])
        group_list = []
        pat_id=[]
        cell =[]
        nset =[]
    
        for idx in test_idx[lab]:
            group_list0,cell0, pat_id0, nset0 = make_inputset(sample_id[lab], cell_id[lab], idx, d[lab], K, k)
            group_list += group_list0
            cell += cell0
            pat_id += pat_id0  
            nset.append(nset0)
            test_group_list[lab] = group_list
            test_pat_id[lab] = pat_id
            test_cell[lab] = cell
            test_nset[lab] = nset
    
    
    # finally prepare training and validation data
    mkdir_p(OUTDIR+'/model/')
    
    
    traincell = []
    trainphenotypes = []

    
    for i in range(0, len(labels)):
        lab = str(labels[i])
        traincell = traincell + (train_cell[lab])
        trainphenotypes = trainphenotypes + [classes[i]] * k *len(train_cell[lab])
    
    testcell = []
    testphenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        testcell = testcell + (test_cell[lab])
        testphenotypes = testphenotypes + [classes[i]] * k *len(test_cell[lab])
    
    
    pickle.dump(traincell, open(OUTDIR+'/model/train_cell_all.p','wb'))
    pickle.dump(testcell, open(OUTDIR+'/model/test_cell_all.p','wb'))
    
    pickle.dump(trainphenotypes, open(OUTDIR+'/model/train_phen_all.p','wb'))
    pickle.dump(testphenotypes, open(OUTDIR+'/model/test_phen_all.p','wb'))
    
    
    for i in labels:
        inx =np.random.choice(a=len(train_pat_id[i]), size=len(train_pat_id[i]), replace=False)
        group_list = (train_group_list[i])
        train_group_list[i] = [group_list[j] for j in inx]
        train_pat_id[i] = np.array(train_pat_id[i])[inx]
    
    for i in labels:
        inx =np.random.choice(a=len(test_pat_id[i]), size=len(test_pat_id[i]), replace=False)
        group_list = (test_group_list[i])
        test_group_list[i] = [group_list[j] for j in inx]
        test_pat_id[i] = np.array(test_pat_id[i])[inx]
        
        # --------------------------------------------------
    # ✅ NEW: SAVE test_cell_ids.p (ONLY ADDITION)
    # --------------------------------------------------
    test_cell_ids = []
    for lab in labels:
        cellids_lab = pickle.load(open(INDIR + '/cellids' + lab + '.p', 'rb'))
        for cid in cellids_lab:
            test_cell_ids.append(cid[:k])

    pickle.dump(
        test_cell_ids,
        open(OUTDIR + '/model/test_cell_ids.p', 'wb')
    )
    print(f"Saved test_cell_ids.p with {len(test_cell_ids)} neighborhoods")
    
    
        
    
    
    
    trnset = []
    tsnset = []
    for i in labels:
        trnset += train_nset[i]
        tsnset += test_nset[i]
    
    nsetmed = int(np.quantile(trnset + tsnset , nset_thr))
    
    g = dict()
    for i in labels:
        g[i] = []
        for idx in train_idx[i]:
            inx = np.where(train_pat_id[i]==idx)[0]
            data = (train_group_list[i])
            data = [data[j] for j in inx]
            if len(inx) > nsetmed:
                data = data[:nsetmed]
            g[i]= g[i] + data 
        
    tg = dict()
    for i in labels:
        tg[i] = []
        for idx in test_idx[i]:
            inx = np.where(test_pat_id[i]==idx)[0]
            data = (test_group_list[i])
            data = [data[j] for j in inx]
            if len(inx) > nsetmed:
                data = data[:nsetmed]
            tg[i]= tg[i] + data 
         
        
    
    specify_valid = True
    
    print('saving samples...')
    
    traincell = []
    trainphenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        traincell = traincell + (train_cell[lab])
        trainphenotypes = trainphenotypes + [classes[i]] * k *len(train_cell[lab])
    
    
    train_samples = []
    train_phenotypes = []
    valid_samples = []
    valid_phenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        g0 = g[lab]
        cut = int(.8 * len(g0))
        
        train_samples += g0[:cut]
        train_phenotypes += [classes[i]] * len(g0[:cut])
        
        valid_samples += g0[cut:] 
        valid_phenotypes += [classes[i]] * len(g0[cut:])
    
    print('training and validation phenotypes', train_phenotypes, valid_phenotypes)
    pickle.dump(valid_samples, open(OUTDIR+'/model/valid_samples.p','wb'))
    pickle.dump(valid_phenotypes, open(OUTDIR+'/model/valid_phenotypes.p','wb')) 
    
    pickle.dump(train_samples, open(OUTDIR+'/model/train_samples.p','wb'))
    pickle.dump(train_phenotypes, open(OUTDIR+'/model/train_phenotypes.p','wb'))
     
    test_samples = []
    test_phenotypes = []
    for i in range(0, len(labels)):
        lab = str(labels[i])
        g0 = tg[lab]
        
        test_samples += g0
        test_phenotypes += [classes[i]] * len(g0)
    
    pickle.dump(test_samples, open(OUTDIR+'/model/test_samples.p','wb'))
    pickle.dump(test_phenotypes, open(OUTDIR+'/model/test_phenotypes.p','wb'))
    
    
    print('ntime_points', ntime_points)
    print('ncell', ncell)
    print('nsubsets', nsubsets)
    print('nrun', nrun)
    print('quant_normed',qt_norm)
    print('scale',scale)
    
    
    model = CellCnn(ntime_points=ntime_points, ncell=ncell, nsubset=nsubsets, 
                                 nrun=nrun, scale=scale, quant_normed=qt_norm, verbose=0, 
                                 per_sample = per_sample)
    
    train_sample = dict()
    valid_sample = dict()
    
    for nt in range(1, ntime_points+1):
        train_sample[str(nt)] = train_samples
        if specify_valid:
           valid_sample[str(nt)] = valid_samples
    
    
    if specify_valid:
       model.fit(ntime_points, train_samples=train_sample, train_phenotypes=train_phenotypes,
              valid_samples=valid_sample, valid_phenotypes=valid_phenotypes,outdir=OUTDIR)
    else:
       model.fit(ntime_points, train_samples=train_sample, train_phenotypes=train_phenotypes,outdir=OUTDIR)
    
    pickle.dump(model.results, open(OUTDIR+'/model/results.p','wb'))
    test_pred = model.predict(test_samples)
    
    pickle.dump(test_pred, open(OUTDIR+'/model/test_pred.p','wb'))
    
    
    from sklearn.metrics import accuracy_score
    test_phenotypespre=np.argmax(test_pred, axis=1).astype(float)
    accuracy_score = accuracy_score(test_phenotypes,test_phenotypespre)
    #accuracy_score = accuracy_score(test_phenotypes,np.around(test_pred[:,1]))
    
    print('Accuracy score: {0:0.2f}'.format(accuracy_score))
    print('Anchor:')
    print(Anchor)
    print('nset:')
    print(trnset )
    print(tsnset)
    print('nsetmed:')
    print(nsetmed)
    
    print(train_idx)
    print(test_idx)
    
    
    
    log_file.close()
    sys.stdout = old_stdout
    
    print('Accuracy score: {0:0.2f}'.format(accuracy_score))
    print('Anchor:')
    print(Anchor)






# # ============================================================
# # SCIMa (SAVE test_cell_ids.p)
# def run_scima(Anchor, ntrain_per_class, K, k, nset_thr, labels, classes, path, nrun, background):

#     ncell = k
#     ntime_points = 1
#     nsubsets = 1
#     per_sample = True
#     scale = True
#     qt_norm = True

#     from datetime import datetime
#     import os, sys, errno, pickle
#     import numpy as np
#     from model import CellCnn

#     # --------------------------------------------------
#     # Setup output
#     # --------------------------------------------------
#     now = datetime.now()
#     today = datetime.today()
#     result_file = 'results' + str(Anchor) + f'_K{K}'

#     current_time = now.strftime("%H_%M_%S")
#     current_day = today.strftime("%d_%m_%Y")
#     old_stdout = sys.stdout

#     INDIR = path + '/Anchor' + str(Anchor) + f'_K{K}'

#     log_dir = path + '/' + result_file + '/run_log/'
#     os.makedirs(log_dir, exist_ok=True)

#     log_file = open(
#         log_dir + f"/data_run_log_{current_day}_{current_time}.txt", "w+"
#     )
#     sys.stdout = log_file

#     OUTDIR = os.path.join(
#         path,
#         result_file,
#         f'out_{current_day}_{current_time}'
#     )
#     mkdir_p(OUTDIR)
#     print('output directory', OUTDIR)

#     # --------------------------------------------------
#     # Load spatial input data
#     # --------------------------------------------------
#     d, s, nset = {}, {}, {}
#     for lab in labels:
#         d[lab] = pickle.load(open(INDIR + '/d' + lab + '.p', 'rb'))
#         s[lab] = pickle.load(open(INDIR + '/s' + lab + '.p', 'rb'))
#         nset[lab] = pickle.load(open(INDIR + '/nset' + lab + '.p', 'rb'))

#     sample_id, cell_id, group = {}, {}, {}
#     for lab in labels:
#         sample_id[lab] = np.asarray(s[lab][0])
#         cell_id[lab] = np.asarray(s[lab][1])
#         group[lab] = np.unique(sample_id[lab])

#     # --------------------------------------------------
#     # Train / test split (ORIGINAL)
#     # --------------------------------------------------
#     np.random.seed(12345)

#     def get_patient(sample):
#         return sample

#     patient_ids = {
#         lab: np.array([get_patient(s) for s in sample_id[lab]])
#         for lab in labels
#     }

#     all_patients = np.unique(
#         np.concatenate([patient_ids[lab] for lab in labels])
#     )
#     np.random.shuffle(all_patients)

#     n_train_patients = max(1, ntrain_per_class)
#     train_patients = all_patients[:n_train_patients]
#     test_patients = all_patients[n_train_patients:]

#     train_idx, test_idx = {}, {}
#     for lab in labels:
#         train_idx[lab] = [s for s in group[lab] if s in train_patients]
#         test_idx[lab]  = [s for s in group[lab] if s in test_patients]

#     # --------------------------------------------------
#     # Build train / test subsets (ORIGINAL)
#     # --------------------------------------------------
#     train_group_list, train_pat_id, train_cell, train_nset = {}, {}, {}, {}
#     test_group_list, test_pat_id, test_cell, test_nset = {}, {}, {}, {}

#     for lab in labels:
#         group_list, pat_id, cell, nset_l = [], [], [], []
#         for idx in train_idx[lab]:
#             gl, c, pid, ns = make_inputset(sample_id[lab], cell_id[lab], idx, d[lab], K, k)
#             group_list += gl
#             cell += c
#             pat_id += pid
#             nset_l.append(ns)
#         train_group_list[lab] = group_list
#         train_cell[lab] = cell
#         train_pat_id[lab] = pat_id
#         train_nset[lab] = nset_l

#     for lab in labels:
#         group_list, pat_id, cell, nset_l = [], [], [], []
#         for idx in test_idx[lab]:
#             gl, c, pid, ns = make_inputset(sample_id[lab], cell_id[lab], idx, d[lab], K, k)
#             group_list += gl
#             cell += c
#             pat_id += pid
#             nset_l.append(ns)
#         test_group_list[lab] = group_list
#         test_cell[lab] = cell
#         test_pat_id[lab] = pat_id
#         test_nset[lab] = nset_l

#     mkdir_p(OUTDIR + '/model')

#     # --------------------------------------------------
#     # ORIGINAL: save train/test cell lists
#     # --------------------------------------------------
#     traincell, trainphenotypes = [], []
#     testcell, testphenotypes = [], []

#     for i, lab in enumerate(labels):
#         traincell += train_cell[lab]
#         trainphenotypes += [classes[i]] * k * len(train_cell[lab])

#         testcell += test_cell[lab]
#         testphenotypes += [classes[i]] * k * len(test_cell[lab])

#     pickle.dump(traincell, open(OUTDIR + '/model/train_cell_all.p', 'wb'))
#     pickle.dump(testcell, open(OUTDIR + '/model/test_cell_all.p', 'wb'))
#     pickle.dump(trainphenotypes, open(OUTDIR + '/model/train_phen_all.p', 'wb'))
#     pickle.dump(testphenotypes, open(OUTDIR + '/model/test_phen_all.p', 'wb'))

#     # --------------------------------------------------
#     # ✅ NEW: SAVE test_cell_ids.p (ONLY ADDITION)
#     # --------------------------------------------------
#     test_cell_ids = []
#     for lab in labels:
#         cellids_lab = pickle.load(open(INDIR + '/cellids' + lab + '.p', 'rb'))
#         for cid in cellids_lab:
#             test_cell_ids.append(cid[:k])

#     pickle.dump(
#         test_cell_ids,
#         open(OUTDIR + '/model/test_cell_ids.p', 'wb')
#     )
#     print(f"Saved test_cell_ids.p with {len(test_cell_ids)} neighborhoods")

#     # --------------------------------------------------
#     # ORIGINAL: model training
#     # --------------------------------------------------
#     model = CellCnn(
#         ntime_points=ntime_points,
#         ncell=ncell,
#         nsubset=nsubsets,
#         nrun=nrun,
#         scale=scale,
#         quant_normed=qt_norm,
#         verbose=0,
#         per_sample=per_sample
#     )

#     train_samples = []
#     train_phenos = []
#     valid_samples = []
#     valid_phenos = []

#     for i, lab in enumerate(labels):
#         g = train_group_list[lab]
#         cut = int(0.8 * len(g))
#         train_samples += g[:cut]
#         train_phenos += [classes[i]] * cut
#         valid_samples += g[cut:]
#         valid_phenos += [classes[i]] * (len(g) - cut)

#     train_sample = {'1': train_samples}
#     valid_sample = {'1': valid_samples}

#     model.fit(
#         ntime_points,
#         train_samples=train_sample,
#         train_phenotypes=train_phenos,
#         valid_samples=valid_sample,
#         valid_phenotypes=valid_phenos,
#         outdir=OUTDIR
#     )

#     pickle.dump(model.results, open(OUTDIR + '/model/results.p', 'wb'))

#     test_pred = model.predict(test_group_list[labels[0]])
#     pickle.dump(test_pred, open(OUTDIR + '/model/test_pred.p', 'wb'))

#     from sklearn.metrics import accuracy_score
#     test_pred_cls = np.argmax(test_pred, axis=1).astype(float)
#     acc = accuracy_score(testphenotypes, test_pred_cls)

#     print(f'Accuracy score: {acc:0.2f}')

#     log_file.close()
#     sys.stdout = old_stdout