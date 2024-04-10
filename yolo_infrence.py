from ultralytics import YOLO

model = YOLO('models/yolo5_last.pt')

result = model.predict('input_videos/input_video2.mp4', save=True, conf=.25)
print(result)
#print("boxes: ")
#for box in result[0].boxes:
#    print(box)