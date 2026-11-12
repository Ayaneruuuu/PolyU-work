Merged Pet + Household YOLOv8 Dataset

Training command:
  yolo detect train model=yolov8n.pt data=merged_pet_household_yolov8/data.yaml epochs=50 imgsz=640 batch=16

Classes:
  0: cat
  1: dog
  2: backpack
  3: book
  4: box
  5: chair
  6: fan
  7: fork
  8: glass
  9: plate
  10: spoon
  11: table
  12: vase

Image counts:
  train: 5553
  valid: 694
  test: 695
