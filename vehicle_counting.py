import glob
import random
import string

import cv2 as cv

from vehicle_detector import VehicleDetector


# generate random string
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# Load Veichle Detector
vd = VehicleDetector()

# Load images from a folder
images_folder = glob.glob("images/*.jpg")

total_vehicle = 0

# Loop through all the images
for img_path in images_folder:
    print("Img path", img_path)
    img = cv.imread(img_path)

    vehicle_boxes = vd.detect_vehicles(img)
    vehicle_count = len(vehicle_boxes)

    # Update total count
    total_vehicle += vehicle_count

    for box in vehicle_boxes:
        x, y, w, h = box

        # draw rectangles on vehicles
        cv.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)

        cv.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    cv.imwrite("output" + id_generator() + ".jpg", img)
    # cv.imshow("Cars", img)
    cv.waitKey(0)

print("Total vehicle", total_vehicle)
