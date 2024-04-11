import cv2
import easyocr
import csv
import torch
import hydra
from datetime import datetime  # Import datetime module for getting current date and time
from pathlib import Path
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from Levenshtein import distance  # Import Levenshtein distance for string comparison

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Dictionary to store previously detected vehicle details across frames
previous_frames = {}

def getOCR(im):
    # Function to extract details from the number plate
    gray_plate = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rgb_plate = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(rgb_plate)
    text = ""
    for result in results:
        text += result[1] + " "  # Add a whitespace between each detail
    return text.strip()  # Strip leading and trailing whitespace

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(Path(self.save_dir) / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        
        # Initialize CSV writer outside the loop
        csv_file_path = f"{self.save_dir}/{self.data_path.stem}_number_plate_details.csv"
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Add headers to CSV if it's empty
            if csvfile.tell() == 0:
                csv_writer.writerow(["Vehicles Detected in Campus", "Date", "Time"])
            
            # Get current date and time
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Iterate over each detection and extract number plate details
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{self.model.names[c]} {conf:.2f}'
                # Get ROI from the frame
                roi = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                # Extract details from the number plate ROI
                plate_details = getOCR(roi)
                if plate_details:
                    # Check if the current vehicle is significantly different from previously detected ones across frames
                    skip = False
                    for prev_label, prev_details in previous_frames.items():
                        if distance(plate_details, prev_details) <= 2:
                            skip = True
                            break
                    if not skip:
                        # Update dictionary with current vehicle details
                        previous_frames[label] = plate_details
                        # Split multiline details and write each one on a separate row
                        details = plate_details.split('\n')
                        for detail in details:
                            # Write each detail along with the current date and time as a row in the CSV file
                            csv_writer.writerow([detail, current_date, current_time])
                        log_string += f"{label} OCR saved to {csv_file_path}, "
                    else:
                        log_string += f"Skipped saving {label} OCR as redundant, "
                else:
                    log_string += f"No text detected from number plate {label}, "
                
                # Add bounding box to image
                if self.args.save or self.args.save_crop or self.args.show:
                    self.annotator.box_label(xyxy, label, color=colors(c, True))
                if self.args.save_crop:
                    imc = im0.copy()
                    save_one_box(xyxy,
                                imc,
                                file=Path(self.save_dir) / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}_{c}.jpg',
                                BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()
