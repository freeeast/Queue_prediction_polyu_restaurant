import csv
import cv2
import numpy as np
import time

def load_yolo():
    net = cv2.dnn.readNet("Source/yolov3.weights", "Source/yolov3.cfg")
    layer_names = net.getLayerNames()
    
    output_layer_indices = net.getUnconnectedOutLayers()
    if output_layer_indices.ndim == 1:
        output_layers = [layer_names[index - 1] for index in output_layer_indices]
    else:
        output_layers = [layer_names[index[0] - 1] for index in output_layer_indices]
    
    return net, output_layers

def detect_objects(img, net, outputLayers):            
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_people_count(outputs, height, width, conf_threshold=0.3, nms_threshold=0.6):
    boxes = []
    confidences = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, x+w, y+h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return len(indices)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    net, output_layers = load_yolo()
    frame_count = 0
    detection_interval = 5

    with open('guide_table.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Frame", "Time", "People_Count"]
        writer.writerow(headers)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            blob, outputs = detect_objects(frame, net, output_layers)
            people_count = get_people_count(outputs, height, width)

            if frame_count % detection_interval == 0:
                data_row = {'Frame': frame_count, 'Time': time.strftime("%H:%M:%S"), 'People_Count': people_count}
                writer.writerow([data_row[h] for h in headers])

            frame_count += 1

        cap.release()

if __name__ == "__main__":
    main('Dataset/7700_1714833871.mp4')