import numpy as np
from rknnlite.api import RKNNLite

IMG_SIZE=640; OBJ_THRESH=0.01; NMS_THRESH=0.45
ANCHORS=np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]],np.float32)
MASKS=[[0,1,2],[3,4,5],[6,7,8]]
def sig(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
def nms(ds):
    ds=np.array(ds,np.float32); ds=ds[np.argsort(ds[:,4])[::-1]][:500]; out=[]
    for c in sorted(set(ds[:,5].astype(int).tolist())):
        d=ds[ds[:,5].astype(int)==c]; order=np.argsort(d[:,4])[::-1]
        while len(order):
            i=order[0]; out.append(d[i]); rest=order[1:]
            xx1=np.maximum(d[i,0],d[rest,0]); yy1=np.maximum(d[i,1],d[rest,1])
            xx2=np.minimum(d[i,2],d[rest,2]); yy2=np.minimum(d[i,3],d[rest,3])
            inter=np.maximum(0,xx2-xx1+1)*np.maximum(0,yy2-yy1+1)
            area=(d[i,2]-d[i,0]+1)*(d[i,3]-d[i,1]+1)
            area2=(d[rest,2]-d[rest,0]+1)*(d[rest,3]-d[rest,1]+1)
            order=rest[inter/(area+area2-inter+1e-9)<=NMS_THRESH]
    return out
def main(model,raw):
    inp=np.fromfile(raw,dtype=np.uint8).reshape(1,640,640,3)
    rk=RKNNLite(); assert rk.load_rknn(model)==0; assert rk.init_runtime()==0; outs=rk.inference(inputs=[inp]); rk.release()
    raw=[]
    for o,mask in zip(outs,MASKS):
        o=np.asarray(o).reshape(3,85,o.shape[-2],o.shape[-1]); gh,gw=o.shape[-2:]; stride=640//gh
        grid=np.stack(np.meshgrid(np.arange(gw),np.arange(gh)),0)
        box=sig(o[:,:4]); obj=sig(o[:,4])[:,None]; cls=sig(o[:,5:]); xy=(box[:,:2]*2-.5+grid)*stride; wh=(box[:,2:]*2)**2*ANCHORS[mask,:,None,None]
        score=obj*cls.max(1,keepdims=True); cid=cls.argmax(1)
        for a in range(3):
            m=score[a,0]>OBJ_THRESH
            for x1,y1,x2,y2,s,c in zip((xy[a,0]-wh[a,0]/2)[m],(xy[a,1]-wh[a,1]/2)[m],(xy[a,0]+wh[a,0]/2)[m],(xy[a,1]+wh[a,1]/2)[m],score[a,0][m],cid[a][m]): raw.append([x1,y1,x2,y2,s,c])
    out=nms(raw)
    print('python_lite_detections=',len(out))
    for d in out[:10]: print('%.6f %.6f %.6f %.6f %.6f %d'%(*d[:5],int(d[5])))
main(__import__('sys').argv[1],__import__('sys').argv[2])
