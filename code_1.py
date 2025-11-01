DATA_PATH=os.path.join("data")
actions=np.array(['Anxiety','Hit','Tantrum','Normal'])
personas=np.array(['gabriel','sebastian','mariana','bibian','david','luis','leidy','ivon','analopez','camilo','zharick',
                   'alexandra','paulac','andres','duban','karlareal','juanjose','richi','angelh','mariad','sofiam',
                   'mariapeña','sebastianin','carolina','camila'])


no_vids_pp=15
gente={nom:range(num*no_vids_pp,(num+1)*no_vids_pp) for num,nom in enumerate(personas)}
no_vids=len(gente)*no_vids_pp
vid_length=16
label_map = {label: num for num, label in enumerate(actions)}

# label_map -> {'Anxiety': 0, 'Hit': 1, ...}

# ['Anxiety']
# enumerate -> 0, 'Anxiety'
# num, label = enumerate(actions) -> num = 0, label = 'Anxiety'
# {label: num} -> {'Anxiety': 0}

 
vids_training,vids_val,vids_test,labels_training,labels_val, labels_test=[],[],[],[],[],[]
no_vid_for_training=round(375*0.7)
no_vid_for_val=round(375*0.1)
no_vid_for_test=round(375*0.2)

for vid in range(no_vid_for_training):
    for action in actions:
        toma=[]
        for frame_num in range(vid_length):
            arr=np.load(os.path.join(DATA_PATH,action,str(vid),"{}.npy".format(frame_num)))
            toma.append(arr)
            
        vids_training.append(toma)
        labels_training.append(label_map[action])
        
vids_val, labels=[],[]
for vid in range(no_vid_for_training,no_vid_for_training+no_vid_for_val):
    for action in actions:
        toma=[]
        for frame_num in range(vid_length):
            arr=np.load(os.path.join(DATA_PATH,action,str(vid),"{}.npy".format(frame_num)))
            toma.append(arr)
            
        vids_val.append(toma)
        labels_val.append(label_map[action])

vids_test, labels=[],[]
for vid in range(no_vid_for_training+no_vid_for_val,no_vids):
    for action in actions:
        toma=[]
        for frame_num in range(vid_length):
            arr=np.load(os.path.join(DATA_PATH,action,str(vid),"{}.npy".format(frame_num)))
            toma.append(arr)
            
        vids_test.append(toma)
        labels_test.append(label_map[action])
