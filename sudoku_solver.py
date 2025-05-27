import cv2
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# Load the trained digit classifier model
model_path = "digit_classifier.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError("Trained model 'digit_classifier.h5' not found!")

# Sudoku Solver (Backtracking Algorithm)
def is_valid(board, row, col, num):
    for i in range(3):
        if board[row][i] == num or board[i][col] == num:
            return False
    return True

def solve_sudoku(board):
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                for num in range(1, 4):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

# Extract the Sudoku grid from an image
def extract_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        pts = np.array([p[0] for p in approx], dtype="float32")
        side = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts, dst)
        grid = cv2.warpPerspective(image, matrix, (int(side), int(side)))
        return grid, pts
    return None, None

# Recognize digits using the CNN model
def recognize_digits(grid):
    board = np.zeros((3, 3), dtype=int)
    side = grid.shape[0] // 3

    for i in range(3):
        for j in range(3):
            x, y = j * side, i * side
            cell = grid[y:y + side, x:x + side]
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            cell_resized = cv2.resize(cell_gray, (28, 28)) / 255.0  # Normalize
            cell_reshaped = cell_resized.reshape(1, 28, 28, 1)

            prediction = model.predict(cell_reshaped)
            board[i][j] = np.argmax(prediction) if np.max(prediction) > 0.5 else 0

    return board

# Draw the solved Sudoku grid on the original frame
def draw_solution(frame, board, pts):
    if pts is None:
        return frame

    side = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2])))
    step = side // 3

    for i in range(3):
        for j in range(3):
            x = int(pts[0][0] + j * step + step // 3)
            y = int(pts[0][1] + i * step + step // 1.5)
            num = board[i][j]

            if num != 0:
                cv2.putText(frame, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Main function to capture webcam image, extract and solve Sudoku, then overlay results
def main():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)  # Use Video4Linux2 (Linux only)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grid, pts = extract_grid(frame)
        if grid is not None:
            board = recognize_digits(grid)
            print("Extracted Board:")
            print(board)

            if solve_sudoku(board):
                print("Solved Board:")
                print(board)
                frame = draw_solution(frame, board, pts)
            else:
                print("No solution found.")

        cv2.imshow("Sudoku Solver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

