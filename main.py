import cv2
import torch.cuda
from ultralytics import YOLO

path = "samples/crowd.jpg"


def checkDevices():
    print("############### SYSTEM CHECKS ###############")
    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("#############################################")


def yoloPredict():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Send model to CUDA aka GPU
    # model.to('cuda')

    # Export the model to ONNX format
    # return model.export(format="onnx")

    # Perform object detection on an image using the model
    results = model(path)

    # build output frame
    build_output(results, model)


def build_output(results, model):
    # Load the original image
    image = cv2.imread(path)

    # Iterate over each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f'{model.names[cls]}: {conf:.2f}'

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the image
    output_path = 'out/result.jpg'
    cv2.imwrite(output_path, image)

    print(f"Output image saved to {output_path}")


if __name__ == '__main__':
    checkDevices()
    yoloPredict()
