import csv
import cv2
import numpy as np
from sort.sort import Sort
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

def get_boxes(outputs, height, width, conf_threshold=0.3, nms_threshold=0.6):
    boxes = []
    confidences = []
    class_ids = []
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
    final_boxes = [boxes[i] for i in indices.flatten()]
    return final_boxes, confidences

#处理视频
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    net, output_layers = load_yolo()
    tracker = Sort(max_age=60, min_hits=1, iou_threshold=0.3)
    track_start_times = {}
    frame_count = 0
    detection_interval = 5 
    id_columns = {}
    data_row = {} 

    with open('queue_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Frame", "Time", "Queue_Length"]
        writer.writerow(headers)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            blob, outputs = detect_objects(frame, net, output_layers)
            boxes, confidences = get_boxes(outputs, height, width)
            trackers = tracker.update(np.array(boxes))
            current_queue_length = len(trackers) 

            for d in trackers:
                xmin, ymin, xmax, ymax, track_id = map(int, d)
                if track_id not in track_start_times:
                    track_start_times[track_id] = time.time()
                    if f"ID_{track_id}_Wait_Time" not in headers:
                        id_columns[track_id] = f"ID_{track_id}_Wait_Time"
                        headers.append(id_columns[track_id])
          
                waiting_time = time.time() - track_start_times[track_id]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                display_text = f"ID: {track_id} Time: {waiting_time:.2f}s"
                cv2.putText(frame, display_text, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                data_row[id_columns[track_id]] = f"{waiting_time:.2f}s"

            if frame_count % detection_interval == 0:
                data_row.update({'Frame': frame_count, 'Time': time.strftime("%H:%M:%S"), 'Queue_Length': current_queue_length})
                writer.writerow([data_row.get(h, '') for h in headers])

            frame_count += 1
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main('Dataset/7700_1714833871.mp4')


