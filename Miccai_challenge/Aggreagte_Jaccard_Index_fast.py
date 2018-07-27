import numpy as np
import tqdm
from tqdm import tqdm_notebook
import numexpr as ne

def compute_intersect_union(m, pred, pred_mark_isused, idx_pred):
    # check the prediction has been used or not
    if pred_mark_isused[idx_pred]:
        intersect = 0
        union = np.count_nonzero(m)
    else:
        p = (pred == idx_pred)
        # replace multiply with bool operation
        s = ne.evaluate("m&p")
        intersect = np.count_nonzero(s)
        #intersect = np.count_nonzero(m & p)
        #u1 = np.count_nonzero(m)
        #u2 = np.count_nonzero(p)
        #union = ne.evaluate("u1+u2-intersect")
        union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
    return (intersect, union)

# fast version of Aggregated Jaccrd Index
def agg_jc_index(mask, pred):
    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0

    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance

    Returns: Aggregated Jaccard index for GT & mask 
    """
    
    mask=mask.astype(np.bool)
    c = 0 # count intersection
    u = 0 # count union
    tqdm.monitor_interval = 0 # disable tqdm monitor to prevent warning message
    pred_instance = pred.max() # predcition instance number
    pred_mark_used = [] # mask used
    pred_mark_isused = np.zeros((pred_instance+1), dtype=bool)
    
    for idx_m in tqdm_notebook(range(len(mask[0,0,:]))):
        # m = mask[:,:,idx_m]
        m = np.take(mask, idx_m, axis=2)
        #intersect_list = []
        #union_list = []
        #iou_list = []
        
        intersect_list, union_list = zip(*[compute_intersect_union(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance+1)])
        #print(intersect_list)
        """
        for idx_pred in range(1, pred_instance+1):
            # check the prediction has been used or not
            if pred_mark_isused[idx_pred] == True:
                intersect = 0
                union = np.count_nonzero(m)
            else:
                p = (pred == idx_pred)
                
                # replace multiply with bool operation 
                s = ne.evaluate("m&p")
                intersect = np.count_nonzero(s)
                union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
            
            intersect_list.append(intersect)
            union_list.append(union)
            #print(intersect_list)
        """
        iou_list = np.array(intersect_list) / np.array(union_list)    
        hit_idx = np.argmax(iou_list)
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        pred_mark_isused[hit_idx+1] = True
        
    pred_mark_used = [x+1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

    u += pred_fp_pixel
    print (c / u)
    return (c / u)