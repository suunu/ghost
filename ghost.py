import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("mps")
predictor = SamPredictor(sam)


cap = cv2.VideoCapture(1)


roi_x, roi_y, roi_w, roi_h = 100, 1000, 800, 400  
roi_selected = False
tracker = None 
tracking_active = False  
bbox = None  

def click_event(event, x, y, flags, param):
    """클릭 이벤트로 추적 대상 초기화"""
    global tracking_active, tracker, bbox

    if event == cv2.EVENT_LBUTTONDOWN:
       
        roi_click_x = x - roi_x
        roi_click_y = y - roi_y

        if 0 <= roi_click_x < roi_w and 0 <= roi_click_y < roi_h:
            print(f"Clicked Position in ROI: ({roi_click_x}, {roi_click_y})")

            
            input_point = np.array([[roi_click_x, roi_click_y]])
            input_label = np.array([1])  

            predictor.set_image(roi_frame)  # 잘라낸 ROI 이미지를 SAM 입력으로
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )

            # 마스크 선택
            best_mask = masks[np.argmax(scores)]
            mask_display = (best_mask * 255).astype(np.uint8)

            
            contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])

                bbox = (roi_x + x, roi_y + y, w, h)

             
                try:
                    tracker = cv2.legacy.TrackerCSRT_create()
                except AttributeError:
                    tracker = cv2.TrackerCSRT_create()

                tracker.init(frame, bbox)
                tracking_active = True
                print(f"Tracking initialized with Bounding Box: {bbox}")
            else:
                print("No contours found for segmentation.")
        else:
            print("Clicked position is outside the ROI.")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_selected = True

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if tracking_active and tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            print(f"Tracking successful: {bbox}")
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2) 
        else:
            print("Tracking failed. Resetting tracker.")
            tracking_active = False
            tracker = None

    # 화면 표시
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
