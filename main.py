import torch
import cv2
from time import time
import numpy as np


def load_custom_yolov5_model(file, confidence):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=file, force_reload=True)
    model.conf = confidence
    return model


def detect_video_from_url(url, model):
    player = cv2.VideoCapture(url)

    assert player.isOpened()
    x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four_cc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("results/videos/detected.avi", four_cc, 20, (x_shape, y_shape))
    while True:
        start_time = time()
        ret, frame = player.read()
        assert ret
        frame_list = [frame]
        results = model(frame_list)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        results = labels, cord

        # plot boxes
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, model.names[0], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 3)
        print(f"Frames Per Second : {fps}")
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = load_custom_yolov5_model('model/drum.pt', 0.5)
    # result = model(["FILEPATH1","FILEPATH2"])
    # result.print()
    # result.show()
    # result.save("results/images")

    URL = "https://redirector.googlevideo.com/videoplayback?expire=1637289402&ei=WbmWYbi6O4T41gKGtIpQ&ip=165.231.71.177&id=o-AO123pTBgs3-YqmiWZ2RKZp224IhGFbgHUEOKoJiej3V&itag=22&source=youtube&requiressl=yes&mh=3K&mm=31%2C26&mn=sn-4g5lznle%2Csn-hpa7zn76&ms=au%2Conr&mv=u&mvi=1&pl=24&vprv=1&mime=video%2Fmp4&ns=IgZggtwvGcs0vX5jgspyy-gG&cnr=14&ratebypass=yes&dur=77.067&lmt=1608317941805408&mt=1637266741&fvip=1&fexp=24001373%2C24007246&c=WEB&txp=6211222&n=usWD2tdW9sWp4g&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cvprv%2Cmime%2Cns%2Ccnr%2Cratebypass%2Cdur%2Clmt&sig=AOq0QJ8wRQIhAOR_8_znYppUV5JZdcupSNRGXXTEdRoeTYja-ows-iX5AiBibsjeu2upC1GyP_mzVL-J-cLNYbwdbsJxzdv1sXlY2A%3D%3D&lsparams=mh%2Cmm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpl&lsig=AG3C_xAwRAIgLGeN9gcN4jnl-jz8-mpW7QA7BexbhHkkRoURrToj428CICm4cnc6tomPQcWATorQOAcJNwuV4uFMW4nRQTkAHqyC&title=Carro+Volcador+para+tambores"
    detect_video_from_url(URL, model)
