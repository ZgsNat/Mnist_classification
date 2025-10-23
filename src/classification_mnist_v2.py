import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('src/mnist_cnn_model_2.h5')

def find_and_crop_digit(canvas_image: np.ndarray) -> tuple[np.ndarray | None, tuple[int,int] | None]:
    """
    find largest contour, extract the ROI area and binary
    Arg:
        canvas_image: image from canvas (BGR)
    Return:
        (digit_roi,(width, height)) or (None, None)
    """
    # 1. preprocess: Convert image to gray, invert and binary (get the white digit in black background)
    gray_image = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray_image)
    _, thresh = cv2.threshold(inverted_gray, 170, 255, cv2.THRESH_BINARY_INV)

    # 2. Find contour
    contour, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contour:
        return None, None
    
    # 3. Find max contour
    main_contour = max(contour, key=cv2.contourArea)
    # If contour is not large enough
    if cv2.contourArea(main_contour) < 100:
        return None, None
    
    # 4. Extract ROI Area
    x, y, w, h = cv2.boundingRect(main_contour)
    digit_roi = thresh[y : y + h, x : x + w]

    return digit_roi, (w, h)

def normalize_to_mnist_format(digit_roi: np.ndarray, original_size: tuple[int,int]) -> np.ndarray:
    """
    """
    w, h = original_size
    # Max size for digit inside frame 28x28
    MAX_MNIST_SIZE = 20

    # 1. Aspect Ratio

    if w > h:
        new_w = MAX_MNIST_SIZE
        new_h = int(round((new_w / w) * h))
    else:
        new_h = MAX_MNIST_SIZE
        new_w = int(round((new_h / h) * w))
    
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 2. Resizing
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Create Canvas 28x28 and center
    canvas_28x28 = np.zeros((28,28), dtype=np.uint8)

    pad_x = (28 - new_w) // 2
    pad_y = (28 - new_h) // 2

    # Put resized digit in the center of canvas

    canvas_28x28[pad_y:pad_y + new_h, pad_x : pad_x + new_w] = resized_digit

    # 4. Normalization and Dimension expansion
    
    #  0 - 255 to 0.0 - 0.1
    final_image = canvas_28x28.astype('float32') / 255.0

    # add channel: 1
    # (28,28) -> (28, 28, 1)
    final_image = np.expand_dims(final_image,axis = -1)

    # add batch: 1
    # (1, 28 , 28, 1)
    final_image = np.expand_dims(final_image, axis = 0)

    return final_image

def preprocess_and_predict(canvas_image: np.ndarray, model: object) -> int:
    digit_roi, original_size = find_and_crop_digit(canvas_image)

    if digit_roi is None:
        print("No digit detect!")
        return None

    final_image = normalize_to_mnist_format(digit_roi, original_size)

    prediction = model.predict(final_image)
    predicted_digit = int(np.argmax(prediction))

    print(f"Predict: {predicted_digit}")
    return predicted_digit

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
            cv2.circle(canvas, (x,y), thickness // 2, color, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), color, thickness)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None
    win_name = "Draw digit - press 'p' to predict, 'c' to clear, 'q' to quit"

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    result = None
    while True:
        display = canvas.copy()
        # Draw instructions on first line
        cv2.putText(display, "p: predict  c: clear  q: quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        # Draw prediction result on second line
        if result is not None:
            cv2.putText(display, f"Predict: {result}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
            result = None  # Clear the result when canvas is cleared
        elif key == ord('p'):
            # call prediction with a copy
            result = preprocess_and_predict(canvas.copy(), model)
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_draw_canvas()
