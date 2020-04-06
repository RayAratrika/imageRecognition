import torch;
from torch.autograd import Variable;
import cv2;
from data import BaseTransform, VOC_CLASSES as labelmap;
from ssd import build_ssd;
import imageio;
def detect(frame,net,transform):
    h,w = frame.shape[:2]
    frame_t=transform(frame)[0]
    x=torch.from_numpy(frame_t).permute(2,0,1)
    x=Variable(x.unsqueeze(0))
    y=net(x)
    det = y.data
    print(type(det))
    scale=torch.Tensor([w,h,w,h])
    print(scale)
    
    for i in range(det.size(1)):
        j=0
        while det[0,i,j,0]>=0.6:
            pt = (det[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame, (int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(0,0,0),3)
            cv2.putText(frame ,labelmap[i-1],(int(pt[0]),int(pt[1])),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),3,cv2.LINE_AA)
            j+=1
        
    return frame

net=build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location=lambda storage, loc:storage))
transform = BaseTransform(net.size,(100/256.0,100/256.0,150/256.0))

reader=imageio.get_reader('funny_dog.mp4')
fps=reader.get_meta_data()['fps']
writer=imageio.get_writer('doggiestyle.mp4',fps=fps)
for i, frame in enumerate(reader):
    frame=detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()
