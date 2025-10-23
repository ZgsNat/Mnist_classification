import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('src/mnist_cnn_model_2.h5')

def preprocess_and_predict(canvas_image, model):

    gray_image = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(gray_image)

    _, thresh = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY_INV)

    # handle different OpenCV return signatures
    contours_info = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) == 0:
            print("Không tìm thấy chữ số nào để dự đoán.")
            return None

        x, y, w, h = cv2.boundingRect(main_contour)
        digit_roi = thresh[y:y+h, x:x+w]

        # Fit into 20x20 box while keeping aspect ratio
        new_size = 20
        if w > h:
            new_w = new_size
            new_h = int(round((new_size / w) * h))
        else:
            new_h = new_size
            new_w = int(round((new_size / h) * w))

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create 28x28 canvas and center the digit
        canvas_28x28 = np.zeros((28, 28), dtype=np.uint8)

        pad_x = (28 - new_w) // 2
        pad_y = (28 - new_h) // 2

        canvas_28x28[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_digit

        # Prepare final image for model
        final_image = canvas_28x28.astype('float32') / 255.0
        final_image = np.expand_dims(final_image, axis=-1)  # channel
        final_image = np.expand_dims(final_image, axis=0)   # batch

        prediction = model.predict(final_image)
        predicted_digit = int(np.argmax(prediction))

        print(f"Dự đoán: {predicted_digit}")
        return predicted_digit

    print("Không tìm thấy chữ số nào để dự đoán.")
    return None

# Below: simple drawable canvas and key handling ('p' to predict, 'c' to clear, 'q' to quit)

def run_draw_canvas():
    canvas_size = 500
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    drawing = False
    last_point = None
    color = (255, 255, 255)
    thickness = 20

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, last_point, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
            cv2.circle(canvas, (x, y), thickness // 2, color, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), color, thickness)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    win_name = "Draw digit - press 'p' to predict, 'c' to clear, 'q' to quit"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        display = canvas.copy()
        cv2.putText(display, "p: predict  c: clear  q: quit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('p'):
            # call prediction with a copy
            preprocess_and_predict(canvas.copy(), model)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_draw_canvas()

