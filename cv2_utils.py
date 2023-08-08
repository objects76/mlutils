import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # To slow when cv2.VideoCapture(0)

import cv2
import time
import numpy as np

print('cv2 version:' + cv2.__version__ +", modified:", time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(__file__))))

class Cam:
    def __init__(self, file_path=None):
        super().__init__()
        method = cv2.CAP_MSMF # too slow to open device
        # method = cv2.CAP_DSHOW
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
        else:
            self.cap = cv2.VideoCapture(0, method)
        assert self.cap and self.cap.isOpened()
        self.fps_tick = 0


    def __del__(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            print('cv2 released...')

        cv2.destroyAllWindows()


    def get_frame(self, selfie_view=False):
        ret, frame = self.cap.read()

        if selfie_view:
            frame.flags.writeable = True
            frame = cv2.flip(frame, 1) # flip horizentally
        return frame


    def is_valid(self):
        if not self.cap.isOpened():
            return False
        if (cv2.waitKey(10) & 0xFF) == 27:  # ESC
            return False
        return True


    def test(self):
        while self.is_valid():
            ret, frame = self.cap.read()

            # Recolor Feed
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
            image = cv2.flip(frame, 1) # flip horizentally
            cv2.imshow('Test', image)

    def show(self, title, image, show_fps=False):
        if show_fps:
            fps = 1 / (time.time() - self.fps_tick)
            self.fps_tick = time.time()
            cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), thickness=3)
        cv2.imshow(title, image)


# https://bskyvision.com/1078
def cv2_imread(filePath, gray2color=False, toRGB=True):
    assert os.path.exists(filePath), f"No file {filePath}"

    # https://zzdd1558.tistory.com/228
    #img = cv2.imread(path)
    bytes = np.fromfile(filePath, np.uint8)
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)
    if toRGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(filePath, img.shape)

    if gray2color and len(img.shape) == 2:
        img = np.stack([img] * 3, 2)

    return img

def cv2_imwrite(dst_path, image, params=None):
    extension = os.path.splitext(dst_path)[1]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result, encoded_img = cv2.imencode(extension, image, params)

    if result:
        with open(dst_path, mode='w+b') as f:
            encoded_img.tofile(f)


if __name__ == '__main__':
    cam = Cam()
    cam.test()
    del cam


