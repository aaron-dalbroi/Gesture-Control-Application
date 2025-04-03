# Gesture-Control-Application

Installation
----------
pip install opencv-python
pip install mediapipe
pip install pytorch

If you get the following error:
    Traceback (most recent call last):
    File "C:\Users\infer\Documents\GitHub\Gesture-Control-Application\VideoCapture.py", line 5, in <module>
        import mediapipe as mp
    File "C:\Users\infer\anaconda3\lib\site-packages\mediapipe\__init__.py", line 15, in <module>
        from mediapipe.python import *
    File "C:\Users\infer\anaconda3\lib\site-packages\mediapipe\python\__init__.py", line 17, in <module>
        from mediapipe.python._framework_bindings import model_ckpt_util
    ImportError: DLL load failed while importing _framework_bindings: A dynamic link library (DLL) initialization routine failed.

then run pip install msvc-runtime

Running
------------
make sure your webcam is turned on. It should automatically detect it if it is.

run the command "python VideoCapture.py"
Enter 0 for camera feed and press enter
press q to exit the video feed. Closing the window itself will not work

Training the model
------------------
There is a CNN model provided already within the repository (rotated_gesture_model.pth)
If you wish to make your own,
1. Download the ipynb file
2. Make any changes to the file
3. Run the script
4. Download the new rotated_gesture_model.pth file generated and replace the old rotated_gesture_model.pth in the repository.\ 