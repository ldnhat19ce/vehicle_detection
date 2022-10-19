import cv2


# CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
#            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
#            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
#            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
#            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
#            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
#            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
#            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
#            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class VehicleDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)

        # Allow classes containing Vehicles only
        self.classes_allowed = [2, 3, 5, 6, 7]

    def detect_vehicles(self, img):
        # Detect Objects
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)

        for class_id, score, box in zip(class_ids, scores, boxes):
            # score: confidence
            if score < 0.5:
                # Skip detection with low confidence
                continue

            # it detects all objects as we only get objects of the class we declare
            if class_id in self.classes_allowed:
                vehicles_boxes.append(box)

        return vehicles_boxes
