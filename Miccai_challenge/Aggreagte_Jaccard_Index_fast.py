# fast version of Aggregated Jaccrd Index
def agg_jc_index(mask, pred):
    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0

    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance

    Returns: Aggregated Jaccard index for GT & mask 
    """
    
    c = 0   # intersection
    u = 0   # unio
    pred_mark_used = []
    pred_mark_used_flag=np.zeros(len(mask[0,0,:]),dtype=bool)
    for idx_m in tqdm_notebook(range(len(mask[0,0,:]))):
        m = mask[:,:,idx_m]
        intersect_list = []
        union_list = []
        for idx_pred in range(1, int(np.max(pred))+1):
            if pred_mark_used_flag[idx_pred-1] == False:
                p = (pred==idx_pred)
                intersect = np.count_nonzero(m & p)
                union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
            #print(intersect, union)
            else:
                intersect=0
                union=1
            intersect_list.append(intersect)
            union_list.append(union)
            
        hit_idx = np.argmax(np.array(intersect_list)/np.array(union_list))
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        #print(pred_mark_used)
        
    pred_mark_used = [x+1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

    u += pred_fp_pixel
    print(c / u)
    return c / u